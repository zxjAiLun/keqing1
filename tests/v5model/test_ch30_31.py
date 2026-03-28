"""Tests for ch30 (tenpai flag) and ch31 (waits positions)."""
import numpy as np
from v5model.features import encode


def _snap(**kw):
    s = {
        "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": ["1z"],
        "hand": [],
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
    }
    s.update(kw)
    return s


def test_ch30_tenpai_menzen():
    """门清听牌：ch30 全列=1。"""
    # 123m456m789m1234p 听1p + 4p
    hand = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[30, :].min() == 1.0


def test_ch30_not_tenpai():
    """非听牌：ch30 全列=0。"""
    hand = ["1m","3m","5m","9m","1p","3p","5p","9p","1s","3s","5s","9s","1z"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[30, :].max() == 0.0


def test_ch31_wait_position_injected():
    """注入 waits_tiles 时 ch31 精确填位置：等4p(index=12)。"""
    waits = [False] * 34
    waits[12] = True  # 4p = index 12
    hand = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]
    tf, _ = encode(_snap(hand=hand, waits_tiles=waits), 0)
    assert tf[31, 12] == 1.0
    assert tf[31, :].sum() == 1.0


def test_ch31_shanpon_two_waits_injected():
    """注入双碰进张(1p/2p = index 9/10)：ch31 有两个位置=1。"""
    waits = [False] * 34
    waits[9] = True   # 1p
    waits[10] = True  # 2p
    hand = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1p","2p","2p"]
    tf, _ = encode(_snap(hand=hand, waits_tiles=waits), 0)
    assert tf[31, 9] == 1.0
    assert tf[31, 10] == 1.0
    assert tf[31, :].sum() == 2.0


def test_ch31_fallback_no_inject_leaves_zero():
    """无注入时 ch31 fallback 为0（get_waits 语义不适用）。"""
    hand = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]
    tf, _ = encode(_snap(hand=hand), 0)
    assert tf[31, :].max() == 0.0


def test_ch30_31_with_pon_meld():
    """碰牌后手牌听牌：ch30=1。ch31 fallback 下为0（需注入 waits_tiles 才有位置信息）。"""
    hand = ["1m","2m","3m","4p","5p","6p","7p","8p","9p","1s"]
    melds = [[{"type": "pon", "pai": "1m", "consumed": ["1m", "1m"], "target": 3}], [], [], []]
    tf, _ = encode(_snap(hand=hand, melds=melds), 0)
    assert tf[30, :].min() == 1.0
    assert tf[31, :].max() == 0.0  # fallback 不填 ch31


def test_ch30_31_not_tenpai_with_meld():
    """碰牌后手牌非听牌：ch30/ch31 全0。"""
    hand = ["2m","4m","6m","8m","1p","3p","5p","7p","9p","1z"]
    melds = [[{"type": "pon", "pai": "1m", "consumed": ["1m", "1m"], "target": 3}], [], [], []]
    tf, _ = encode(_snap(hand=hand, melds=melds), 0)
    assert tf[30, :].max() == 0.0
    assert tf[31, :].max() == 0.0
