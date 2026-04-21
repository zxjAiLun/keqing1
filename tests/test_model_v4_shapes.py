import torch

from mahjong_env.action_space import ACTION_SPACE
from training.state_features import C_TILE, N_SCALAR
from training.cache_schema import (
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)
from keqingv4.model import KeqingV4Model


def test_keqingv4_forward_shapes():
    model = KeqingV4Model(hidden_dim=64, num_res_blocks=2, action_embed_dim=16, context_dim=12)
    tile = torch.randn(3, C_TILE, 34)
    scalar = torch.randn(3, N_SCALAR)
    event_history = torch.zeros(3, 48, 5, dtype=torch.long)
    discard_summary = torch.randn(3, 34, KEQINGV4_SUMMARY_DIM)
    call_summary = torch.randn(3, KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM)
    special_summary = torch.randn(3, KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM)
    logits, value = model(
        tile,
        scalar,
        event_history=event_history,
        discard_summary=discard_summary,
        call_summary=call_summary,
        special_summary=special_summary,
    )

    assert logits.shape == (3, ACTION_SPACE)
    assert value.shape == (3, 1)

    aux = model.get_last_aux_outputs()
    assert aux["global_value"].shape == (3, 1)
    assert aux["composed_ev"].shape == (3, 1)
    assert aux["win_prob"].shape == (3, 1)
    assert aux["dealin_prob"].shape == (3, 1)
    assert aux["pts_given_win"].shape == (3, 1)
    assert aux["pts_given_dealin"].shape == (3, 1)
    assert aux["opp_tenpai_logits"].shape == (3, 3)
    assert aux["rank_logits"].shape == (3, 4)
    assert aux["final_score_delta"].shape == (3, 1)


def test_keqingv4_forward_under_cuda_autocast_keeps_logit_assignment_safe():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    model = KeqingV4Model(hidden_dim=64, num_res_blocks=2, action_embed_dim=16, context_dim=12).to(device)
    tile = torch.randn(2, C_TILE, 34, device=device)
    scalar = torch.randn(2, N_SCALAR, device=device)
    event_history = torch.zeros(2, 48, 5, dtype=torch.long, device=device)
    discard_summary = torch.randn(2, 34, KEQINGV4_SUMMARY_DIM, device=device)
    call_summary = torch.randn(2, KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM, device=device)
    special_summary = torch.randn(2, KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM, device=device)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        logits, value = model(
            tile,
            scalar,
            event_history=event_history,
            discard_summary=discard_summary,
            call_summary=call_summary,
            special_summary=special_summary,
        )

    assert logits.shape == (2, ACTION_SPACE)
    assert value.shape == (2, 1)


def test_keqingv4_event_history_changes_state_encoding():
    model = KeqingV4Model(hidden_dim=64, num_res_blocks=2, action_embed_dim=16, context_dim=12, dropout=0.0)
    tile = torch.randn(2, C_TILE, 34)
    scalar = torch.randn(2, N_SCALAR)
    pad_history = torch.zeros(2, 48, 5, dtype=torch.long)
    pad_history[..., 0] = 4
    pad_history[..., 2] = -1
    active_history = pad_history.clone()
    active_history[:, -1, 0] = 0
    active_history[:, -1, 1] = 2
    active_history[:, -1, 2] = 13
    active_history[:, -1, 3] = 3
    active_history[:, -1, 4] = 1
    shared_pad, _ = model.encode_state(tile, scalar, event_history=pad_history)
    shared_active, _ = model.encode_state(tile, scalar, event_history=active_history)
    assert not torch.allclose(shared_pad, shared_active)
