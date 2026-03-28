"""Tests for v5model feature encoding (features.py).

Covers:
- tile_feat shape and dtype
- hand count planes (ch 0-3)
- meld presence planes (ch 4-7)
- shanten/waits from snap injection (training path)
- shanten/waits self-computed fallback (inference path, menzen)
- scalar shape and basic values
- aka (red5) flags (ch 56-58)
"""
import numpy as np
import pytest

from v5model.features import encode, C_TILE, N_SCALAR


def _base_snap(**kwargs):
    snap = {
        "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": ["1z"],
        "actor": 0,
        "hand": [],
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
    }
    snap.update(kwargs)
    return snap


def test_output_shapes():
    snap = _base_snap(hand=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"])
    tf, sc = encode(snap, 0)
    assert tf.shape == (C_TILE, 34)
    assert tf.dtype == np.float32
    assert sc.shape == (N_SCALAR,)
    assert sc.dtype == np.float32


def test_hand_count_planes():
    # 3张1m → ch0,ch1,ch2 的 1m位置=1, ch3=0
    snap = _base_snap(hand=["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"])
    tf, _ = encode(snap, 0)
    # 1m 是 tile index 0
    assert tf[0, 0] == 1.0  # >=1张
    assert tf[1, 0] == 1.0  # >=2张
    assert tf[2, 0] == 1.0  # >=3张
    assert tf[3, 0] == 0.0  # <4张
    # 2m 只有1张
    assert tf[0, 1] == 1.0
    assert tf[1, 1] == 0.0


def test_meld_presence_planes():
    # 碰了1m，consumed=[1m,1m]，pai=1m
    snap = _base_snap(
        hand=["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"],
        melds=[[{"type": "pon", "pai": "1m", "consumed": ["1m", "1m"], "target": 1}], [], [], []],
    )
    tf, sc = encode(snap, 0)
    # ch4: 第0个副露槽，1m在index 0
    assert tf[4, 0] == 1.0
    # ch5: 第1个副露槽（空）
    assert tf[5, :].sum() == 0.0
    # scalar[10]: 副露数/4 = 0.25
    assert abs(sc[10] - 0.25) < 1e-6


def test_shanten_snap_injection():
    """训练路径：snap 注入 shanten/waits_count，应直接使用不自算。"""
    snap = _base_snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p","5p"],
        shanten=0,
        waits_count=5,
    )
    _, sc = encode(snap, 0)
    assert abs(sc[7] - 0.0 / 8.0) < 1e-6   # shanten=0
    assert abs(sc[8] - 5.0 / 34.0) < 1e-6  # waits_count=5


def test_shanten_snap_injection_nonzero():
    """snap 注入 shanten=2，scalar[7] = 2/8."""
    snap = _base_snap(
        hand=["1m","3m","5m","6m","9m","1p","2p","3p","4p","5p","9p","3s","4s","1z"],
        shanten=2,
        waits_count=0,
    )
    _, sc = encode(snap, 0)
    assert abs(sc[7] - 2.0 / 8.0) < 1e-6
    assert sc[8] == 0.0


def test_shanten_self_computed_menzen_tenpai():
    """推理路径（无注入）：门前13张听牌，向听=0，进张>0。"""
    # 123m456m789m123p 听4p
    snap = _base_snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
    )
    _, sc = encode(snap, 0)
    assert sc[7] == 0.0        # shanten=0
    assert sc[8] > 0.0         # 有进张


def test_shanten_self_computed_not_tenpai():
    """推理路径（无注入）：非听牌，进张=0。"""
    snap = _base_snap(
        hand=["1m","3m","5m","9m","1p","3p","5p","9p","1s","3s","5s","9s","1z"],
    )
    _, sc = encode(snap, 0)
    assert sc[7] > 0.0   # shanten > 0
    assert sc[8] == 0.0  # 非听牌无进张


def test_aka_flags():
    """赤宝牌 flag：ch56=赤5m, ch57=赤5p, ch58=赤5s。"""
    snap = _base_snap(
        hand=["0m", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
    )
    tf, _ = encode(snap, 0)
    assert tf[56, :].max() == 1.0  # 赤5m flag
    assert tf[57, :].max() == 0.0  # 无赤5p
    assert tf[58, :].max() == 0.0  # 无赤5s


def test_no_aka_flags():
    """手牌无赤宝牌时 ch56-58 全 0。"""
    snap = _base_snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
    )
    tf, _ = encode(snap, 0)
    assert tf[56, :].max() == 0.0
    assert tf[57, :].max() == 0.0
    assert tf[58, :].max() == 0.0


def test_meld_reduces_hand():
    """副露后 hand 里已扣除 consumed 牌，ch0 的计数应反映剩余手牌。"""
    # 碰了3张1m，hand里不再有1m
    snap = _base_snap(
        hand=["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"],
        melds=[[{"type": "pon", "pai": "1m", "consumed": ["1m", "1m"], "target": 1}], [], [], []],
    )
    tf, _ = encode(snap, 0)
    # hand 里无1m，ch0 index 0 应为 0
    assert tf[0, 0] == 0.0
    # hand 里有1张2m
    assert tf[0, 1] == 1.0
    assert tf[1, 1] == 0.0
