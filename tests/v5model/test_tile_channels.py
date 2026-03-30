"""Tests for tile_feat channels: discards, riichi, dora, meld/discard counts."""
import numpy as np
import pytest
from v5model.features import encode


def _snap(**kw):
    s = {
        "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": ["1z"],
        "hand": ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
    }
    s.update(kw)
    return s


def test_opponent_discard_presence_ch8():
    """上家(pid=1)弃牌 1m → ch8 index 0 = 1.0。"""
    tf, _ = encode(_snap(discards=[[], ["1m"], [], []]), 0)
    assert tf[8, 0] == 1.0
    assert tf[8, 1] == 0.0


def test_opponent_discard_presence_ch12():
    """对家(pid=2)弃牌 2p → ch12 index 10 = 1.0。"""
    # 2p = index 10 (9m+1)
    tf, _ = encode(_snap(discards=[[], [], ["2p"], []]), 0)
    assert tf[12, 10] == 1.0


def test_self_discard_presence_ch20():
    """自家弃牌 9m → ch20 index 8 = 1.0。"""
    tf, _ = encode(_snap(discards=[["9m"], [], [], []]), 0)
    assert tf[20, 8] == 1.0


def test_riichi_broadcast_ch21():
    """pid=0 立直 → ch21 全列 = 1.0。"""
    tf, _ = encode(_snap(reached=[True, False, False, False]), 0)
    assert tf[21, :].min() == 1.0
    assert tf[22, :].max() == 0.0


def test_riichi_broadcast_ch23():
    """pid=2 立直 → ch23 全列 = 1.0，其余 0。"""
    tf, _ = encode(_snap(reached=[False, False, True, False]), 0)
    assert tf[23, :].min() == 1.0
    assert tf[21, :].max() == 0.0
    assert tf[22, :].max() == 0.0
    assert tf[24, :].max() == 0.0


def test_dora_marker_ch25():
    """第一个 dora_marker=1m → ch25 index 0 = 1.0。"""
    tf, _ = encode(_snap(dora_markers=["1m"]), 0)
    assert tf[25, 0] == 1.0
    assert tf[26, :].max() == 0.0


def test_dora_marker_multiple():
    """3 个 dora_marker 分别占 ch25/26/27。"""
    tf, _ = encode(_snap(dora_markers=["1m", "2p", "3s"]), 0)
    assert tf[25, 0] == 1.0   # 1m index 0
    assert tf[26, 10] == 1.0  # 2p index 10
    assert tf[27, 20] == 1.0  # 3s index 20 (9m+9p+2)
    assert tf[28, :].max() == 0.0


def test_meld_count_ch32():
    """pid=0 有2副露 → ch32 全列 = 2/4 = 0.5。"""
    melds = [[{"type":"pon","pai":"1m","consumed":["1m","1m"],"target":1},
              {"type":"chi","pai":"3p","consumed":["1p","2p"],"target":3}], [], [], []]
    tf, _ = encode(_snap(melds=melds), 0)
    assert abs(tf[32, 0] - 0.5) < 1e-6
    assert abs(tf[33, 0]) < 1e-6  # pid=1 no melds


def test_discard_count_ch36():
    """ch36-39 已删除，应全为 0。"""
    tf, _ = encode(_snap(discards=[[], ["1m","2m","3m"], [], []]), 0)
    assert tf[36:40, :].max() == 0.0


def test_bakaze_onehot_E_ch40():
    """East場 → ch40 全列 = 1.0，ch41-43 = 0。"""
    tf, _ = encode(_snap(bakaze="E"), 0)
    assert tf[40, :].min() == 1.0
    for ch in (41, 42, 43):
        assert tf[ch, :].max() == 0.0


def test_bakaze_onehot_S_ch41():
    """South場 → ch41 = 1.0，ch40 = 0。"""
    tf, _ = encode(_snap(bakaze="S"), 0)
    assert tf[41, :].min() == 1.0
    assert tf[40, :].max() == 0.0


def test_jikaze_onehot_ch91_actor0_oya0():
    """actor=0, oya=0 → jikaze=0(East) → ch91 = 1.0。"""
    tf, _ = encode(_snap(oya=0), 0)
    assert tf[91, :].min() == 1.0
    for ch in (92, 93, 94):
        assert tf[ch, :].max() == 0.0


def test_jikaze_onehot_ch92_actor1_oya0():
    """actor=1, oya=0 → jikaze=1(South) → ch92 = 1.0。"""
    tf, _ = encode(_snap(oya=0), 1)
    assert tf[92, :].min() == 1.0
    assert tf[91, :].max() == 0.0


def test_jikaze_wraps_actor0_oya2():
    """actor=0, oya=2 → jikaze=(0-2)%4=2(West) → ch93 = 1.0。"""
    tf, _ = encode(_snap(oya=2), 0)
    assert tf[93, :].min() == 1.0


def test_turn_estimate_ch47():
    """ch47 已删除，应全为 0。"""
    discs = [["1m","2m","3m","4m","5m","6m"], [], [], []]
    tf, _ = encode(_snap(discards=discs), 0)
    assert tf[47, :].max() == 0.0


def test_score_diff_ch48_equal():
    """各家分数相同 → ch48-51 全为 0。"""
    tf, _ = encode(_snap(scores=[25000,25000,25000,25000]), 0)
    for ch in range(48, 52):
        assert tf[ch, :].max() == 0.0


def test_score_diff_ch48_actor0_lower():
    """actor=0 score=20000, pid=1 score=30000 → ch49=(30000-20000)/30000。"""
    tf, _ = encode(_snap(scores=[20000,30000,25000,25000]), 0)
    expected = (30000 - 20000) / 30000.0
    assert abs(tf[49, 0] - expected) < 1e-6
    assert tf[48, 0] == 0.0  # self diff = 0
