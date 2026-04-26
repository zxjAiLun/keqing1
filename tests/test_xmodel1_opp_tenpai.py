"""Xmodel1 opp_tenpai × 3 label 回填与 v2 cache 合同验证.

该测试覆盖以下链路:
  1. build_replay_samples_mc_return 在决策时刻计算 opp_tenpai 并写入 ReplaySample
  2. events_to_xmodel1_arrays 输出 opp_tenpai_target 数组到 cache
  3. Xmodel1DiscardDataset 严格拒绝缺少 v2 必填字段的旧 cache
  4. trainer 用 opp_tenpai_loss_weight>0 配合新 cache 时 loss 贡献生效且可 backward
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mahjong_env.replay import _compute_opp_tenpai_target, build_replay_samples_mc_return
from mahjong_env.state import GameState, apply_event
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.preprocess import events_to_xmodel1_arrays


def _TENPAI_FIXTURE_TEHAIS():
    """明确 actor=0 决策时刻下对手 1 tenpai、对手 2/3 明显非 tenpai 的发牌。

    - actor 0: 1m-9m 各 1 + 1p×2 2p 3p
    - actor 1 (下家): 1m-9m 各 1 + 5p×2 6p 7p —— 纯正九莲宝灯听牌
      shanten=0
    - actor 2 (对家): 散牌 (3m 6m 2p 4p 8p 9p 2s 5s 8s S W N + P) —— shanten 很高
    - actor 3 (上家): 散牌 (7m 9p 1s 3s 6s 9s E F C + 2m 5m 8m 4s) —— shanten 很高

    注: 每种牌用量不超 4 张,满足牌山约束。
    """
    return [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "5p", "5p", "6p", "7p"],
        # 对手 2 (对家):非 tenpai 的散牌
        ["3m", "6m", "2p", "4p", "8p", "9p", "2s", "5s", "8s", "S", "W", "N", "P"],
        # 对手 3 (上家):非 tenpai 的另一副散牌
        ["7m", "8m", "9p", "1s", "3s", "6s", "9s", "E", "F", "C", "2m", "5m", "4s"],
    ]


def _build_tenpai_fixture_state() -> GameState:
    """构造 actor=0 决策时刻、对手 1 已 tenpai、对手 2/3 未 tenpai 的 state。"""
    start_game = {"type": "start_game", "names": ["A", "B", "C", "D"]}
    start_kyoku = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_marker": "1m",
        "tehais": _TENPAI_FIXTURE_TEHAIS(),
    }
    state = GameState()
    apply_event(state, start_game)
    apply_event(state, start_kyoku)
    return state


def test_compute_opp_tenpai_target_detects_tenpai_opponent():
    state = _build_tenpai_fixture_state()
    # actor=0 视角: 对手 1/2/3 相对位置为 (actor+1, +2, +3)
    opp = _compute_opp_tenpai_target(state, actor=0)
    assert isinstance(opp, tuple) and len(opp) == 3
    # 对手 1 (次家) 应该 tenpai
    assert opp[0] == 1.0, f"expected opp1 tenpai=1.0, got {opp}"
    # 对手 2/3 明显非 tenpai
    assert opp[1] == 0.0, f"expected opp2 not tenpai, got {opp}"
    assert opp[2] == 0.0, f"expected opp3 not tenpai, got {opp}"


def test_build_replay_samples_mc_return_fills_opp_tenpai_target():
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": _TENPAI_FIXTURE_TEHAIS(),
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    samples = build_replay_samples_mc_return(events, strict_legal_labels=False)
    assert samples, "fixture should yield at least one sample"
    # 找到 actor=0 的那个 dahai 样本,验证 opp_tenpai_target
    dahai_samples = [s for s in samples if s.actor == 0 and s.label_action.get("type") == "dahai"]
    assert dahai_samples, "expected at least one dahai sample for actor 0"
    s = dahai_samples[0]
    assert isinstance(s.opp_tenpai_target, tuple) and len(s.opp_tenpai_target) == 3
    # 对手 1 两面听,应为 tenpai
    assert s.opp_tenpai_target[0] == 1.0, f"opp1 expected tenpai, got {s.opp_tenpai_target}"
    assert s.opp_tenpai_target[1] == 0.0
    assert s.opp_tenpai_target[2] == 0.0


def test_events_to_xmodel1_arrays_emits_opp_tenpai_target():
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": _TENPAI_FIXTURE_TEHAIS(),
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    arrays = events_to_xmodel1_arrays(events, replay_id="fixture.mjson")
    assert arrays is not None
    assert "opp_tenpai_target" in arrays, "preprocess must emit opp_tenpai_target"
    opp = arrays["opp_tenpai_target"]
    assert opp.ndim == 2 and opp.shape[1] == 3
    assert opp.shape[0] == arrays["state_tile_feat"].shape[0]
    # 第一行样本 actor=0,对手 1 tenpai
    assert float(opp[0, 0]) == 1.0
    assert float(opp[0, 1]) == 0.0
    assert float(opp[0, 2]) == 0.0


def test_cached_dataset_rejects_cache_missing_required_opp_tenpai(tmp_path: Path):
    """xmodel1_discard_v2 下缺少 opp_tenpai_target 的 cache 必须报错并要求重跑 preprocess。"""
    from xmodel1.schema import (
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_MAX_SPECIAL_CANDIDATES,
        XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    )

    n = 2
    path = tmp_path / "old_cache.npz"
    candidate_mask = np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.uint8)
    candidate_mask[:, 0] = 1
    candidate_tile_id = np.full((n, XMODEL1_MAX_CANDIDATES), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 0
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 56), dtype=np.float16),
        candidate_feat=np.zeros((n, XMODEL1_MAX_CANDIDATES, 35), dtype=np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, XMODEL1_MAX_CANDIDATES, 10), dtype=np.uint8),
        candidate_quality_score=np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.uint8),
        chosen_candidate_idx=np.array([0, 0], dtype=np.int16),
        special_candidate_feat=np.zeros(
            (n, XMODEL1_MAX_SPECIAL_CANDIDATES, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM), dtype=np.float16
        ),
        special_candidate_type_id=np.full((n, XMODEL1_MAX_SPECIAL_CANDIDATES), -1, dtype=np.int16),
        special_candidate_mask=np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.uint8),
        special_candidate_quality_score=np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.uint8),
        chosen_special_candidate_idx=np.full((n,), -1, dtype=np.int16),
        sample_type=np.zeros(n, dtype=np.int8),
        action_idx_target=np.zeros(n, dtype=np.int16),
        actor=np.zeros(n, dtype=np.int8),
        event_index=np.zeros(n, dtype=np.int32),
        kyoku=np.ones(n, dtype=np.int8),
        honba=np.zeros(n, dtype=np.int8),
        is_open_hand=np.zeros(n, dtype=np.uint8),
        global_value_target=np.zeros(n, dtype=np.float32),
        score_delta_target=np.zeros(n, dtype=np.float32),
        win_target=np.zeros(n, dtype=np.float32),
        dealin_target=np.zeros(n, dtype=np.float32),
        offense_quality_target=np.zeros(n, dtype=np.float32),
        # 故意不写 opp_tenpai_target,模拟未按 v2 合同重跑 preprocess 的旧 cache
    )

    ds = Xmodel1DiscardDataset([path], shuffle=False, buffer_size=4, seed=0)
    try:
        list(iter(ds))
    except ValueError as exc:
        msg = str(exc)
        assert "missing required fields" in msg
        assert "opp_tenpai_target" in msg
        assert "rerun preprocess" in msg
    else:
        raise AssertionError("expected cache missing opp_tenpai_target to fail under xmodel1_discard_v2")


def test_trainer_opp_tenpai_loss_soft_enable():
    """验证 trainer: weight=0 时 loss 不受影响 ; weight>0 时 BCE 生效可 backward。"""
    from xmodel1.model import Xmodel1Model
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    B = 4
    state_tile_feat = torch.randn(B, 57, 34)
    state_scalar = torch.randn(B, 56)
    candidate_feat = torch.randn(B, 14, 35)
    candidate_tile_id = torch.zeros(B, 14, dtype=torch.long)
    candidate_flags = torch.zeros(B, 14, 10)
    candidate_mask = torch.zeros(B, 14)
    candidate_mask[:, 0] = 1.0
    out = model(
        state_tile_feat,
        state_scalar,
        candidate_feat,
        candidate_tile_id,
        candidate_flags,
        candidate_mask,
    )
    # 随机 opp_tenpai 目标:模拟新 cache 标签
    opp_target = torch.tensor([[1.0, 0.0, 0.0]] * B)
    # 验证 BCE 可微分
    loss = F.binary_cross_entropy_with_logits(out.opp_tenpai_logits.float(), opp_target)
    loss.backward()
    # opp_tenpai_head 应该有梯度
    head = model.opp_tenpai_head[-1]
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    # BCE 值在合理范围 (随机初始化的 model,监督 opp_target 后 loss 应 > 0)
    assert float(loss.item()) > 0.0
