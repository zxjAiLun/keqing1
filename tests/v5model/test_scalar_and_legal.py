"""Tests for scalar features and enumerate_legal_actions."""
import numpy as np
import pytest
from v5model.features import encode, N_SCALAR
from mahjong_env.legal_actions import enumerate_legal_actions


def _snap(**kw):
    s = {
        "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": ["1z"],
        "hand": ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
        "last_discard": None,
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
        "actor_to_move": 0,
    }
    s.update(kw)
    return s


# ---------------------------------------------------------------------------
# scalar features
# ---------------------------------------------------------------------------

def test_scalar_shape():
    _, sc = encode(_snap(), 0)
    assert sc.shape == (N_SCALAR,)
    assert sc.dtype == np.float32


def test_scalar_bakaze_E():
    _, sc = encode(_snap(bakaze="E"), 0)
    assert sc[0] == 1.0
    assert sc[1] == 0.0


def test_scalar_bakaze_S():
    _, sc = encode(_snap(bakaze="S"), 0)
    assert sc[0] == 0.0
    assert sc[1] == 1.0


def test_scalar_kyoku_honba_kyotaku():
    _, sc = encode(_snap(kyoku=3, honba=4, kyotaku=2), 0)
    assert abs(sc[2] - 3/4.0) < 1e-6
    assert abs(sc[3] - 4/8.0) < 1e-6
    assert abs(sc[4] - 2/8.0) < 1e-6


def test_scalar_score_and_rank():
    # actor=0 最高分 → rank=0
    _, sc = encode(_snap(scores=[40000, 20000, 20000, 20000]), 0)
    assert abs(sc[5] - 40000/50000.0) < 1e-6
    assert sc[6] == 0.0  # rank 0 / 3


def test_scalar_rank_last():
    # actor=0 最低分 → 3人比他高 → rank=3/3=1.0
    _, sc = encode(_snap(scores=[10000, 30000, 30000, 30000]), 0)
    assert abs(sc[6] - 1.0) < 1e-6


def test_scalar_riichi_flag():
    # actor=0 立直
    _, sc = encode(_snap(reached=[True, False, False, False]), 0)
    assert sc[9] == 1.0


def test_scalar_meld_count():
    melds = [[{"type":"pon","pai":"1m","consumed":["1m","1m"],"target":1},
              {"type":"chi","pai":"3p","consumed":["1p","2p"],"target":3}],
             [], [], []]
    _, sc = encode(_snap(melds=melds), 0)
    assert abs(sc[10] - 2/4.0) < 1e-6


def test_scalar_discard_count():
    discs = [["1m","2m","3m","4m","5m","6m"], [], [], []]
    _, sc = encode(_snap(discards=discs), 0)
    assert abs(sc[11] - 6/24.0) < 1e-6


def test_scalar_other_riichi_flags():
    # pid=1,3 立直
    _, sc = encode(_snap(reached=[False, True, False, True]), 0)
    assert sc[12] == 1.0  # pid=1
    assert sc[13] == 0.0  # pid=2
    assert sc[14] == 1.0  # pid=3


def test_scalar_style_zero_default():
    _, sc = encode(_snap(), 0)
    for i in range(16, 20):
        assert sc[i] == 0.0


def test_scalar_style_always_zero():
    """style 字段(sc[16-19])未实现，训练时恒为0，snap 注入无效。"""
    snap = _snap(style=[0.5, -0.5, 1.0, -1.0])
    _, sc = encode(snap, 0)
    assert sc[16] == 0.0
    assert sc[17] == 0.0
    assert sc[18] == 0.0
    assert sc[19] == 0.0


# ---------------------------------------------------------------------------
# enumerate_legal_actions
# ---------------------------------------------------------------------------

def test_legal_dahai_basic():
    """actor_to_move=0, 有手牌 → 应生成 dahai 动作。"""
    snap = _snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        actor_to_move=0,
        last_discard=None,
    )
    actions = enumerate_legal_actions(snap, 0)
    types = {a.type for a in actions}
    assert "dahai" in types


def test_legal_chi_available():
    """下家(pid=1)可以吃上家(pid=0)打出的 4m，手有 2m/3m。"""
    snap = _snap(
        hand=["2m","3m","5m","6m","7m","8m","9m","1p","2p","3p"],
        actor_to_move=1,
        last_discard={"actor": 0, "pai": "4m", "pai_raw": "4m"},
    )
    actions = enumerate_legal_actions(snap, 1)
    types = [a.type for a in actions]
    assert "chi" in types


def test_legal_chi_prefer_aka():
    """手牌有 5mr(内部赤5m格式) 时，chi consumed 优先返回 5mr 而非 5m。"""
    # 上家打 6m，下家手有 5mr/7m 可以 5-6-7 吃
    snap = _snap(
        hand=["5mr","7m","2p","3p","4p","5p","6p","7p","8p","9p"],
        actor_to_move=1,
        last_discard={"actor": 0, "pai": "6m", "pai_raw": "6m"},
    )
    actions = enumerate_legal_actions(snap, 1)
    chi_actions = [a for a in actions if a.type == "chi"]
    assert any("5mr" in a.consumed for a in chi_actions)


def test_legal_pon_available():
    """手有两张 1m，上家打 1m → 可碰。"""
    snap = _snap(
        hand=["1m","1m","5m","6m","7m","8m","9m","1p","2p","3p"],
        actor_to_move=2,
        last_discard={"actor": 1, "pai": "1m", "pai_raw": "1m"},
    )
    actions = enumerate_legal_actions(snap, 2)
    types = [a.type for a in actions]
    assert "pon" in types


def test_legal_none_when_not_actor_to_move():
    """不是自己的回合且无 last_discard → 只有 none。"""
    snap = _snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        actor_to_move=1,
        last_discard=None,
    )
    actions = enumerate_legal_actions(snap, 0)
    assert all(a.type == "none" for a in actions)


def test_legal_tsumogiri_uses_raw():
    """摸到赤5m(0m)，tsumogiri 打出时 pai 应为 0m 而非 5m。"""
    snap = _snap(
        hand=["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","5m"],
        actor_to_move=0,
        last_discard=None,
        last_tsumo=["5m", None, None, None],
        last_tsumo_raw=["0m", None, None, None],
    )
    actions = enumerate_legal_actions(snap, 0)
    dahai = [a for a in actions if a.type == "dahai" and a.pai == "0m"]
    assert len(dahai) >= 1
