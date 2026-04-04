import numpy as np

from training.cached_dataset import V3AuxAdapter
from training.trainer import _format_nonfinite_debug, _is_finite_scalar, _meld_metric_from_stats


def test_v3_aux_adapter_permute_scalar_swaps_aka_and_suit_ratios():
    adapter = V3AuxAdapter()
    scalar = [0.0] * 56
    scalar[18] = 0.1  # aka_m
    scalar[19] = 0.2  # aka_p
    scalar[20] = 0.3  # aka_s
    scalar[30] = 0.4  # man ratio
    scalar[31] = 0.5  # pin ratio
    scalar[32] = 0.6  # sou ratio

    permuted = adapter.permute_scalar(scalar=np.array(scalar, dtype=np.float32), perm=(2, 0, 1))

    assert list(permuted[18:21]) == [0.3, 0.1, 0.2]
    assert list(permuted[30:33]) == [0.6, 0.4, 0.5]


def test_meld_metric_uses_only_response_window_types():
    stats = {
        "acc_by_type": {
            "none": 0.8,
            "chi": 0.5,
            "pon": 0.25,
            "daiminkan": 1.0,
            "dahai": 0.0,
            "reach": 0.0,
        },
        "total_by_type": {
            "none": 10,
            "chi": 4,
            "pon": 2,
            "daiminkan": 1,
            "dahai": 100,
        },
    }
    metric = _meld_metric_from_stats(stats)
    expected = (10 * 0.8 + 4 * 0.5 + 2 * 0.25 + 1 * 1.0) / (10 + 4 + 2 + 1)
    assert metric == expected


def test_is_finite_scalar_rejects_inf_and_nan():
    assert _is_finite_scalar(0.0) is True
    assert _is_finite_scalar(1.5) is True
    assert _is_finite_scalar(float("inf")) is False
    assert _is_finite_scalar(float("-inf")) is False
    assert _is_finite_scalar(float("nan")) is False


def test_format_nonfinite_debug_includes_reason_and_metrics():
    msg = _format_nonfinite_debug(
        tag="train",
        batch_idx=12,
        reason="grad_norm",
        loss_value=1.2,
        ce_value=0.8,
        val_loss_value=0.1,
        extra_loss_value=0.3,
        extra_metrics={"score_loss": 0.02, "win_loss": 0.4},
        lr=3e-4,
        grad_norm=float("inf"),
    )
    assert "reason=grad_norm" in msg
    assert "batch=   12" in msg
    assert "score_loss=0.0200" in msg
    assert "win_loss=0.4000" in msg
    assert "grad_norm=inf" in msg
