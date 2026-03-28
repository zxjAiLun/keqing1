#!/usr/bin/env python3
"""Convert Tenhou XML log (mjloggm) to Tenhou6 JSON format for convlog."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote
from xml.etree import ElementTree as ET


# tile136（天凤XML）→ TenhouTile 编码（天凤6 JSON）
# tile136: 每种牌4张，0-35=1-9m, 36-71=1-9p, 72-107=1-9s, 108-135=字牌(E/S/W/N/P/F/C)
# 赤宝牌: tile136=16(5mr), 52(5pr), 88(5sr)
def _tile136_to_t6(t: int) -> int:
    if t == 16:  return 51  # AkaMan5
    if t == 52:  return 52  # AkaPin5
    if t == 88:  return 53  # AkaSou5
    if t < 36:   return 11 + (t // 4)        # 1-9m → 11-19
    if t < 72:   return 21 + ((t - 36) // 4) # 1-9p → 21-29
    if t < 108:  return 31 + ((t - 72) // 4) # 1-9s → 31-39
    # 字牌: 108-111=E, 112-115=S, 116-119=W, 120-123=N, 124-127=P, 128-131=F, 132-135=C
    return 41 + ((t - 108) // 4)             # E/S/W/N/P/F/C → 41-47


def _decode_naki_m(m: int) -> str:
    """Decode Tenhou XML 'm' attribute to Tenhou6 naki string for convlog.
    Based on tenhou-paifu-to-json reference implementation.
    """
    def t6(tile136: int) -> str:
        return f"{_tile136_to_t6(tile136):02d}"

    if m & 0x4:  # Chi
        tile_detail = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
        block1 = m >> 10
        called = block1 % 3
        base = (block1 // 21) * 8 + (block1 // 3) * 4
        called_t136 = tile_detail[called] + 4 * called + base
        consumed_t136 = [tile_detail[i] + 4 * i + base for i in range(3) if i != called]
        called_s = t6(called_t136)
        consumed_s = [t6(x) for x in consumed_t136]
        return f"c{called_s}{consumed_s[0]}{consumed_s[1]}"

    elif m & 0x18:  # Pon or Kakan
        tile4th = (m >> 5) & 3
        target_r = m & 3
        block1 = m >> 9
        called = block1 % 3
        base = 4 * (block1 // 3)
        is_pon = ((m >> 3) & 1) != 0
        pon_tiles = [i + base for i in range(4) if i != tile4th]
        if is_pon:
            called_t136 = pon_tiles[called]
            consumed_t136 = [pon_tiles[i] for i in range(3) if i != called]
            called_s = t6(called_t136)
            consumed_s = [t6(x) for x in consumed_t136]
            if target_r == 3:
                return f"p{called_s}{consumed_s[0]}{consumed_s[1]}"
            elif target_r == 2:
                return f"{consumed_s[0]}p{called_s}{consumed_s[1]}"
            else:
                return f"{consumed_s[0]}{consumed_s[1]}p{called_s}"
        else:  # kakan
            kakan_t136 = tile4th + base
            consumed_t136 = pon_tiles
            kakan_s = t6(kakan_t136)
            consumed_s = [t6(x) for x in consumed_t136]
            if target_r == 3:
                return f"k{kakan_s}{consumed_s[0]}{consumed_s[1]}{consumed_s[2]}"
            elif target_r == 2:
                return f"{consumed_s[0]}k{kakan_s}{consumed_s[1]}{consumed_s[2]}"
            else:
                return f"{consumed_s[0]}{consumed_s[1]}k{kakan_s}{consumed_s[2]}"

    else:  # Kan (daiminkan or ankan)
        target_r = m & 3
        block1 = m >> 8
        called = block1 % 4
        base = 4 * (block1 // 4)
        all_s = [t6(i + base) for i in range(4)]
        if target_r == 0:  # ankan
            return f"{all_s[0]}{all_s[1]}{all_s[2]}a{all_s[3]}"
        elif target_r == 3:  # kamicha daiminkan
            return f"m{all_s[0]}{all_s[1]}{all_s[2]}{all_s[3]}"
        elif target_r == 2:  # toimen daiminkan
            return f"{all_s[0]}m{all_s[1]}{all_s[2]}{all_s[3]}"
        else:               # shimocha daiminkan
            return f"{all_s[0]}{all_s[1]}{all_s[2]}m{all_s[3]}"

# 摸牌标签前缀 → 玩家 index
_DRAW_PREFIX = {"T": 0, "U": 1, "V": 2, "W": 3}
# 打牌标签前缀 → 玩家 index
_DISCARD_PREFIX = {"D": 0, "E": 1, "F": 2, "G": 3}
# tsumogiri 在天凤6里用 60 表示
_TSUMOGIRI = 60


def _tag_prefix(tag: str) -> str:
    """从标签名提取前缀字母，如 'T52' -> 'T'。"""
    m = re.match(r'^([A-Z]+)', tag)
    return m.group(1) if m else ""


def _tag_number(tag: str) -> Optional[int]:
    """从标签名提取数字，如 'T52' -> 52。"""
    m = re.search(r'(\d+)$', tag)
    return int(m.group(1)) if m else None


class KyokuState:
    def __init__(self):
        self.meta = [0, 0, 0]
        self.scoreboard = [25000, 25000, 25000, 25000]
        self.dora_indicators: List[int] = []
        self.ura_indicators: List[int] = []
        self.haipai = [[], [], [], []]
        self.takes = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.results = []
        self._last_draw = [None, None, None, None]
        self._reach_pending = [False, False, False, False]


def _kyoku_to_list(k: KyokuState) -> list:
    entry = [k.meta, k.scoreboard, k.dora_indicators, k.ura_indicators]
    for i in range(4):
        entry.append(k.haipai[i])
        entry.append(k.takes[i])
        entry.append(k.discards[i])
    entry.append(k.results)
    return entry


def _parse_agari(elem: ET.Element) -> list:
    sc = elem.get("sc", "")
    sc_parts = [int(x) for x in sc.split(",")] if sc else []
    deltas = [0, 0, 0, 0]
    if len(sc_parts) >= 8:
        for i in range(4):
            deltas[i] = sc_parts[i * 2 + 1]
    return ["和了", deltas, []]


def _parse_ryuukyoku(elem: ET.Element) -> list:
    sc = elem.get("sc", "")
    sc_parts = [int(x) for x in sc.split(",")] if sc else []
    deltas = [0, 0, 0, 0]
    if len(sc_parts) >= 8:
        for i in range(4):
            deltas[i] = sc_parts[i * 2 + 1]
    return ["流局", deltas, []]


def convert_xml_to_tenhou6(xml_str: str) -> dict:
    xml_str = xml_str.lstrip('\ufeff')
    root = ET.fromstring(xml_str)

    logs = []
    names = ["NoName", "NoName", "NoName", "NoName"]
    rule = {"disp": "", "aka": 1}
    kyoku: Optional[KyokuState] = None

    for elem in root:
        tag = elem.tag
        prefix = _tag_prefix(tag)
        num = _tag_number(tag)

        if tag == "UN":
            for i, attr in enumerate(["n0", "n1", "n2", "n3"]):
                val = elem.get(attr, "")
                if val:
                    names[i] = unquote(val)

        elif tag == "GO":
            rule["disp"] = elem.get("lobby", "")

        elif tag == "INIT":
            if kyoku is not None:
                logs.append(_kyoku_to_list(kyoku))
            kyoku = KyokuState()

            seed_str = elem.get("seed", "0,0,0,0,0,0")
            seed_parts = [int(x) for x in seed_str.split(",")]
            round_num = seed_parts[0] if len(seed_parts) > 0 else 0
            honba     = seed_parts[1] if len(seed_parts) > 1 else 0
            kyotaku   = seed_parts[2] if len(seed_parts) > 2 else 0
            first_dora = seed_parts[5] if len(seed_parts) > 5 else 0

            kyoku.meta = [round_num, honba, kyotaku]
            ten_str = elem.get("ten", "250,250,250,250")
            kyoku.scoreboard = [int(x) * 100 for x in ten_str.split(",")]
            kyoku.dora_indicators = [_tile136_to_t6(first_dora)]

            for i in range(4):
                hai_str = elem.get(f"hai{i}", "")
                if hai_str:
                    kyoku.haipai[i] = [_tile136_to_t6(int(x)) for x in hai_str.split(",")]

        elif prefix in _DRAW_PREFIX and num is not None and kyoku is not None:
            player = _DRAW_PREFIX[prefix]
            t6 = _tile136_to_t6(num)
            kyoku.takes[player].append(t6)
            kyoku._last_draw[player] = (num, t6)

        elif prefix in _DISCARD_PREFIX and num is not None and kyoku is not None:
            player = _DISCARD_PREFIX[prefix]
            t6 = _tile136_to_t6(num)
            last = kyoku._last_draw[player]
            if kyoku._reach_pending[player]:
                kyoku._reach_pending[player] = False
                if last is not None and last[0] == num:
                    kyoku.discards[player].append(f"r{_TSUMOGIRI:02d}")
                else:
                    kyoku.discards[player].append(f"r{t6:02d}")
            elif last is not None and last[0] == num:
                kyoku.discards[player].append(_TSUMOGIRI)
            else:
                kyoku.discards[player].append(t6)

        elif tag == "N" and kyoku is not None:
            who = int(elem.get("who", "0"))
            m = int(elem.get("m", "0"))
            naki_str = _decode_naki_m(m)
            # ankan and kakan go in discards; chi/pon/daiminkan go in takes
            if 'a' in naki_str or 'k' in naki_str:
                kyoku.discards[who].append(naki_str)
            else:
                kyoku.takes[who].append(naki_str)

        elif tag == "REACH" and kyoku is not None:
            who = int(elem.get("who", "0"))
            step = int(elem.get("step", "1"))
            if step == 1:
                kyoku._reach_pending[who] = True

        elif tag == "DORA" and kyoku is not None:
            hai = int(elem.get("hai", "0"))
            kyoku.dora_indicators.append(_tile136_to_t6(hai))

        elif tag == "AGARI" and kyoku is not None:
            kyoku.results.append(_parse_agari(elem))
            if elem.get("owari"):
                logs.append(_kyoku_to_list(kyoku))
                kyoku = None

        elif tag == "RYUUKYOKU" and kyoku is not None:
            kyoku.results.append(_parse_ryuukyoku(elem))
            if elem.get("owari"):
                logs.append(_kyoku_to_list(kyoku))
                kyoku = None

    if kyoku is not None:
        logs.append(_kyoku_to_list(kyoku))

    return {"log": logs, "name": names, "rule": rule}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Tenhou XML to Tenhou6 JSON")
    parser.add_argument("input", help="Input XML file (.json from tenhou.net/0/log/)")
    parser.add_argument("output", help="Output Tenhou6 JSON file")
    args = parser.parse_args()

    xml_str = Path(args.input).read_text(encoding="utf-8", errors="replace")
    result = convert_xml_to_tenhou6(xml_str)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    kyoku_count = len(result["log"])
    print(f"Converted {args.input} -> {args.output} ({kyoku_count} kyoku)")


if __name__ == "__main__":
    main()
