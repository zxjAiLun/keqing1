"""Stage 3 E 回归:runtime 填 special_candidate 通道后

验证:
1. ``build_runtime_special_candidate_arrays`` 返回形状 / dtype / mask 与
   preprocess 路径一致。
2. ``KeqingModelAdapter`` (xmodel1) 的 forward 真的把 special 通道接上,
   表现为 action_logits 里 reach/pon/kan/none 这些 slot 的分数**不再只由
   ``misc_action_head`` 投影**;与"把 runtime special builder 强制置 None"
   时的分数对比差异可见。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.legal_actions import enumerate_legal_actions
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.features import build_runtime_special_candidate_arrays
from xmodel1.model import Xmodel1Model
from xmodel1.schema import XMODEL1_SPECIAL_TYPE_PON


def _pon_fixture() -> tuple[dict, int]:
    """构造一个轮到 actor 0 考虑 pon 的 snapshot(直接构造 snapshot,
    避免依赖完整事件流的重建逻辑)。"""

    snap: dict = {
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "hand": [
            "3p", "3p", "1m", "2m", "4m", "5m", "7m", "8m",
            "9m", "1s", "2s", "3s", "4s",
        ],
        "tsumo_pai": None,
        "melds": [[], [], [], []],
        "discards": [[], [{"pai": "3p", "tsumogiri": False}], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "pending_reach": [False, False, False, False],
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
        "last_discard": {"actor": 1, "pai": "3p", "pai_raw": "3p", "tsumogiri": False},
        "last_kakan": None,
        "actor_to_move": 0,
    }
    actor = 0
    legal = [a.to_mjai() if hasattr(a, "to_mjai") else dict(a) for a in enumerate_legal_actions(snap, actor)]
    if not any(a.get("type") == "pon" for a in legal):
        raise AssertionError(f"fixture 未产生 pon 候选;legal={legal}")
    snap["legal_actions"] = legal
    return snap, actor


def test_build_runtime_special_candidate_arrays_returns_expected_shapes():
    snap, actor = _pon_fixture()
    feat, type_id, mask = build_runtime_special_candidate_arrays(snap, actor)
    assert feat.shape == (XMODEL1_MAX_SPECIAL_CANDIDATES, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM)
    assert feat.dtype == np.float32
    assert type_id.shape == (XMODEL1_MAX_SPECIAL_CANDIDATES,)
    assert type_id.dtype == np.int16
    assert mask.shape == (XMODEL1_MAX_SPECIAL_CANDIDATES,)
    assert mask.dtype == np.uint8
    assert mask.sum() >= 2, "pon 场景至少应产生 pon + none 两个 special slot"
    active_types = set(int(type_id[i]) for i in range(XMODEL1_MAX_SPECIAL_CANDIDATES) if mask[i] > 0)
    assert XMODEL1_SPECIAL_TYPE_PON in active_types


def _make_xmodel1_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "xmodel1_runtime_special.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    torch.manual_seed(0)
    for param in model.parameters():
        param.data.uniform_(-0.3, 0.3)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "state_scalar_dim": 56,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "hidden_dim": 32,
                "num_res_blocks": 1,
                "schema_name": "xmodel1_discard_v3",
                "schema_version": 3,
            },
            "model_version": "xmodel1",
            "schema_name": "xmodel1_discard_v3",
            "schema_version": 3,
        },
        ckpt,
    )
    return ckpt


_REACH_IDX = 34
_PON_IDX = 38
_NONE_IDX = 44


def test_xmodel1_forward_uses_special_candidate_channel(tmp_path: Path):
    """打开 vs 关闭 runtime special builder,检验 forward 产生不同 flat logits。"""

    ckpt = _make_xmodel1_checkpoint(tmp_path)
    snap, actor = _pon_fixture()

    adapter_on = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    assert adapter_on._runtime_special_candidate_builder is not None
    result_on = adapter_on.forward(snap, actor)
    assert result_on.xmodel1 is not None

    adapter_off = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    adapter_off._runtime_special_candidate_builder = None
    result_off = adapter_off.forward(snap, actor)

    assert result_on.policy_logits.shape == result_off.policy_logits.shape

    pon_on = float(result_on.policy_logits[_PON_IDX])
    pon_off = float(result_off.policy_logits[_PON_IDX])
    assert not np.isclose(pon_on, pon_off), (
        "pon slot 在 special 通道 on/off 下应产生不同 logit;"
        f"on={pon_on} off={pon_off}"
    )

    none_on = float(result_on.policy_logits[_NONE_IDX])
    none_off = float(result_off.policy_logits[_NONE_IDX])
    assert not np.isclose(none_on, none_off), (
        "none slot 在 special 通道 on/off 下应产生不同 logit;"
        f"on={none_on} off={none_off}"
    )

    assert np.isfinite(result_on.policy_logits).all()
