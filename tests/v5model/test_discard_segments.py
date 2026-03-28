"""Tests for ch59-90 discard-turn segment encoding and aka flags."""
import numpy as np
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


# ---------------------------------------------------------------------------
# ch 59-90: 弃牌巡目分段
# ch = 59 + pid*8 + seg,  seg = min(turn//3, 7)
# ---------------------------------------------------------------------------

def test_segment_first_discard_pid0_seg0():
    """pid=0 第0张弃牌(turn=0) → seg=0 → ch=59+0*8+0=59。"""
    tf, _ = encode(_snap(discards=[["1m"], [], [], []]), 0)
    assert tf[59, 0] == 1.0  # 1m index 0
    # seg1 以上应为 0
    assert tf[60, 0] == 0.0


def test_segment_turn3_is_seg1():
    """pid=0 turn=3(第4张) → seg=1 → ch=59+0+1=60。"""
    discs = [["9m","9m","9m","1p"], [], [], []]
    tf, _ = encode(_snap(discards=discs), 0)
    # 1p = index 9
    assert tf[60, 9] == 1.0
    assert tf[59, 9] == 0.0


def test_segment_turn21_is_seg7():
    """turn=21 → seg=7 (max)。"""
    discs = [["9s"] * 22, [], [], []]
    tf, _ = encode(_snap(discards=discs), 0)
    # 9s = index 26
    assert tf[59 + 7, 26] == 1.0


def test_segment_pid1_ch67():
    """pid=1 第0张 → ch=59+1*8+0=67。"""
    tf, _ = encode(_snap(discards=[[], ["3s"], [], []]), 0)
    # 3s = index 20
    assert tf[67, 20] == 1.0
    assert tf[59, 20] == 0.0  # pid=0 无弃牌


def test_segment_pid3_ch83():
    """pid=3 第0张 → ch=59+3*8+0=83。"""
    tf, _ = encode(_snap(discards=[[], [], [], ["E"]]), 0)
    # E = index 27
    assert tf[83, 27] == 1.0


def test_segment_aka_normalized():
    """弃牌 '5mr' 去赤后编码到 index 4（5m），ch59 第4列=1。"""
    tf, _ = encode(_snap(discards=[["5mr"], [], [], []]), 0)
    # pid=0 turn=0 seg=0 ch=59, tile34=4
    assert tf[59, 4] == 1.0


def test_segment_overwrite_same_tile_same_seg():
    """同一牌在同一 seg 出现两次（不同 turn），ch 值仍为 1.0（赋值幂等）。"""
    discs = [["1m", "1m", "1m"], [], [], []]  # turn 0,1,2 都在 seg0
    tf, _ = encode(_snap(discards=discs), 0)
    assert tf[59, 0] == 1.0


def test_segment_channels_95_plus_zero():
    """ch95-127 应全为 0（保留）。"""
    tf, _ = encode(_snap(), 0)
    assert tf[95:, :].max() == 0.0


# ---------------------------------------------------------------------------
# ch 56-58 赤宝牌 flag 扩展测试
# ---------------------------------------------------------------------------

def test_aka_all_three():
    """手牌含 5mr/5pr/5sr → ch56/57/58 全置 1.0。"""
    hand = ["5mr","5pr","5sr","1m","2m","3m","4m","6m","7m","8m","9m","1p","2p"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[56, :].min() == 1.0
    assert tf[57, :].min() == 1.0
    assert tf[58, :].min() == 1.0


def test_aka_only_0p():
    """只有 5pr → ch57=1, ch56/ch58=0。"""
    hand = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","5pr","2p","3p","4p"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[56, :].max() == 0.0
    assert tf[57, :].min() == 1.0
    assert tf[58, :].max() == 0.0


def test_aka_hand_count_counts_as_5():
    """'5mr' 去赤后为 '5m'(tile34=4)，与普通 '5m' 共2张 → ch0,ch1=1, ch2=0。"""
    hand = ["5mr","5m","1m","2m","3m","4m","6m","7m","8m","9m","1p","2p","3p"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[0, 4] == 1.0
    assert tf[1, 4] == 1.0
    assert tf[2, 4] == 0.0
