from __future__ import annotations

import torch

from keqingrl.buffer import PPOBatch
from keqingrl.contracts import ObsTensorBatch, PolicyInput, PolicyOutput
from keqingrl.learning_signal import (
    PpoDiagnosticConfig,
    batch_diagnostic_summary,
    loss_gradient_decomposition,
    ppo_update_probe,
    top1_margin_diagnostics,
)
from keqingrl.policy import RulePriorDeltaPolicy
from keqingrl.ppo import compute_ppo_loss
from training.state_features import C_TILE, N_SCALAR


class _IndependentLogitPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[0.2, 0.0, -0.5]], dtype=torch.float32))

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        batch_size = int(policy_input.legal_action_mask.shape[0])
        logits = self.logits.expand(batch_size, -1)
        return PolicyOutput(
            action_logits=logits,
            value=torch.zeros(batch_size),
            rank_logits=torch.zeros(batch_size, 4),
            aux={"prior_logits": torch.zeros_like(logits), "neural_delta": logits},
        )


def _independent_logit_batch(policy: _IndependentLogitPolicy, advantage: float, action_index_value: int) -> PPOBatch:
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros(1, 1, 1),
            scalar_obs=torch.zeros(1, 1),
            extras={},
        ),
        legal_action_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        legal_action_features=torch.zeros(1, 3, 1),
        legal_action_mask=torch.ones(1, 3, dtype=torch.bool),
        rule_context=torch.zeros(1, 6),
        prior_logits=torch.zeros(1, 3),
    )
    action_index = torch.tensor([action_index_value], dtype=torch.long)
    with torch.no_grad():
        dist = torch.distributions.Categorical(logits=policy(policy_input).action_logits)
    return PPOBatch(
        policy_input=policy_input,
        action_index=action_index,
        old_log_prob=dist.log_prob(action_index).detach(),
        old_value=torch.zeros(1),
        advantages=torch.tensor([advantage], dtype=torch.float32),
        returns=torch.tensor([advantage], dtype=torch.float32),
        final_rank_target=torch.tensor([0], dtype=torch.long),
    )


def _batch_and_policy() -> tuple[RulePriorDeltaPolicy, PPOBatch]:
    torch.manual_seed(7)
    batch_size = 2
    action_count = 3
    policy = RulePriorDeltaPolicy(hidden_dim=16, num_res_blocks=1, dropout=0.0)
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.randn(batch_size, C_TILE, 8),
            scalar_obs=torch.randn(batch_size, N_SCALAR),
            extras={},
        ),
        legal_action_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        legal_action_features=torch.randn(batch_size, action_count, 5),
        legal_action_mask=torch.ones(batch_size, action_count, dtype=torch.bool),
        rule_context=torch.zeros(batch_size, 6),
        prior_logits=torch.tensor([[0.2, 0.0, -0.5], [0.1, 0.0, -0.2]], dtype=torch.float32),
    )
    with torch.no_grad():
        output = policy(policy_input)
        dist = torch.distributions.Categorical(logits=output.action_logits)
        action_index = torch.tensor([1, 0], dtype=torch.long)
        old_log_prob = dist.log_prob(action_index)
        old_value = output.value.detach()
    batch = PPOBatch(
        policy_input=policy_input,
        action_index=action_index,
        old_log_prob=old_log_prob.detach(),
        old_value=old_value,
        advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=torch.tensor([1.0, -1.0], dtype=torch.float32),
        final_rank_target=torch.tensor([0, 3], dtype=torch.long),
    )
    return policy, batch


def test_learning_signal_batch_and_margin_stats() -> None:
    policy, batch = _batch_and_policy()
    rows = [
        {"reward": 0.0, "return": 1.0, "advantage_raw": 1.0, "selected_is_prior_top1": False, "delta_needed_to_flip_top1": 0.2},
        {"reward": -1.0, "return": -1.0, "advantage_raw": -1.0, "selected_is_prior_top1": True, "delta_needed_to_flip_top1": 0.1},
    ]

    summary = batch_diagnostic_summary(rows)
    margin = top1_margin_diagnostics(policy, batch)["summary"]

    assert summary["batch_size"] == 2
    assert summary["reward_nonzero_count"] == 1
    assert summary["selected_non_top1_positive_advantage_count"] == 1
    assert margin["selected_non_top1_positive_advantage_count"] == 1
    assert margin["mean_delta_needed_to_flip_top1"] >= 0.0


def test_loss_gradient_decomposition_reports_actor_gradients() -> None:
    policy, batch = _batch_and_policy()

    report = loss_gradient_decomposition(
        policy,
        batch,
        config=PpoDiagnosticConfig(value_coef=0.0, entropy_coef=0.0, rank_coef=0.0, rule_kl_coef=0.0),
    )

    total = next(row for row in report["rows"] if row["component"] == "total_loss")
    policy_only = next(row for row in report["rows"] if row["component"] == "policy_loss_only")
    assert total["actor_grad_norm_total"] > 0.0
    assert policy_only["policy_mlp.final_linear.weight_grad_norm"] > 0.0


def test_actor_only_update_moves_selected_logits_by_advantage_sign() -> None:
    for advantage, action_index_value, comparator in (
        (1.0, 1, torch.gt),
        (-1.0, 0, torch.lt),
    ):
        policy = _IndependentLogitPolicy()
        batch = _independent_logit_batch(policy, advantage, action_index_value)
        before = policy.logits[0, action_index_value].detach().clone()

        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_ppo_loss(
            policy,
            batch,
            clip_eps=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
            rank_coef=0.0,
            rule_kl_coef=0.0,
            normalize_advantages=False,
        )
        loss.total_loss.backward()
        optimizer.step()

        after = policy.logits[0, action_index_value].detach()
        assert bool(comparator(after, before))


def test_post_update_probe_measures_nonzero_kl_after_step() -> None:
    policy, batch = _batch_and_policy()

    rows = ppo_update_probe(policy, batch, lrs=(1e-2,), update_epochs=(1,), normalize_advantages=False)

    assert len(rows) == 1
    assert rows[0]["post_approx_kl_vs_old"] > 0.0
    assert rows[0]["post_ratio_std"] >= 0.0
