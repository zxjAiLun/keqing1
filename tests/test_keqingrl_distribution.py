from __future__ import annotations

import torch

from keqingrl.distribution import MaskedCategorical


def test_masked_categorical_never_samples_illegal_actions() -> None:
    torch.manual_seed(7)
    logits = torch.tensor([[0.0, 8.0, -1.0, 2.0], [1.0, -2.0, 0.5, 3.0]])
    mask = torch.tensor([[True, False, True, True], [False, False, True, True]])
    dist = MaskedCategorical(logits, mask)

    for _ in range(128):
        sample = dist.sample()
        assert mask[0, sample[0]]
        assert mask[1, sample[1]]


def test_masked_categorical_greedy_probs_and_entropy_respect_mask() -> None:
    logits = torch.tensor([[1.0, 9.0, 2.0], [5.0, 4.0, 3.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    dist = MaskedCategorical(logits, mask)

    greedy = dist.greedy()
    probs = dist.probs()
    entropy = dist.entropy()
    log_prob = dist.log_prob(torch.tensor([2, 1]))

    assert greedy.tolist() == [2, 1]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    assert probs[0, 1].item() == 0.0
    assert probs[1, 0].item() == 0.0
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(log_prob).all()


def test_masked_categorical_requires_at_least_one_legal_action_per_row() -> None:
    logits = torch.zeros((2, 3))
    mask = torch.tensor([[True, False, False], [False, False, False]])
    try:
        MaskedCategorical(logits, mask)
    except ValueError as exc:
        assert "at least one legal action" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for all-illegal row")


def test_masked_categorical_rejects_non_finite_logits_and_illegal_log_prob() -> None:
    logits = torch.tensor([[0.0, float("nan")]])
    mask = torch.tensor([[True, True]])
    try:
        MaskedCategorical(logits, mask)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for non-finite logits")

    dist = MaskedCategorical(torch.tensor([[0.0, 1.0]]), torch.tensor([[True, False]]))
    assert dist.greedy().tolist() == [0]
    assert dist.entropy().item() == 0.0
    try:
        dist.log_prob(torch.tensor([1]))
    except ValueError as exc:
        assert "legal" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for illegal action index")
