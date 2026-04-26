import pytest

pytest.importorskip("torch")

from training.trainer import _format_nonfinite_debug, _is_finite_scalar, _meld_metric_from_stats


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
