from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HONOR_TO_CODE = {
    "E": 41,
    "S": 42,
    "W": 43,
    "N": 44,
    "P": 45,
    "F": 46,
    "C": 47,
}

SUIT_BASE = {
    "m": 10,
    "p": 20,
    "s": 30,
}


def mjai_tile_to_tenhou_code(tile: str) -> int:
    """
    Convert mjai tile string (e.g. "5p", "5pr", "E") into tenhou.net/6 tile code.
    Red 5 tiles:
      5mr -> 51, 5pr -> 52, 5sr -> 53
    """
    if tile in HONOR_TO_CODE:
        return HONOR_TO_CODE[tile]

    if len(tile) == 2 and tile[0].isdigit() and tile[1] in SUIT_BASE:
        num = int(tile[0])
        return SUIT_BASE[tile[1]] + num

    if len(tile) == 3 and tile[0].isdigit() and tile[2] == "r" and tile[1] in SUIT_BASE:
        num = int(tile[0])
        if num != 5:
            return SUIT_BASE[tile[1]] + num
        return 50 + {"m": 1, "p": 2, "s": 3}[tile[1]]

    raise ValueError(f"unsupported mjai tile: {tile!r}")


def dora_indicator_code(tile: str) -> int:
    """
    Convert dora indicator tile to tenhou code.
    Red 5 tiles should be preserved as red 5 codes (51/52/53).
    e.g. "5pr" -> 52, "5mr" -> 51, "5sr" -> 53
    """
    if len(tile) == 3 and tile[0] == "5" and tile[2] == "r" and tile[1] in SUIT_BASE:
        return 50 + {"m": 1, "p": 2, "s": 3}[tile[1]]
    return mjai_tile_to_tenhou_code(tile)


def mjai_tile_to_two_digits(tile: str) -> str:
    return f"{mjai_tile_to_tenhou_code(tile):02d}"


def encode_chi(actor: int, target: int, pai: str, consumed: List[str]) -> str:
    # tenhou format: "c" + called tile + two consumed tiles
    c = [mjai_tile_to_two_digits(t) for t in consumed]
    return "c" + mjai_tile_to_two_digits(pai) + c[0] + c[1]


def encode_pon(actor: int, target: int, pai: str, consumed: List[str]) -> str:
    # tenhou format (see mjai-reviewer/convlog/src/conv.rs):
    # - from kamicha: "p" + pai + c0 + c1
    # - from toimen : c0 + "p" + pai + c1
    # - from shimocha: c0 + c1 + "p" + pai
    c0 = mjai_tile_to_two_digits(consumed[0])
    c1 = mjai_tile_to_two_digits(consumed[1])
    pp = mjai_tile_to_two_digits(pai)

    # kamicha = (actor + 3) % 4, toimen = (actor + 2) % 4, shimocha = (actor + 1) % 4
    if target == (actor + 3) % 4:
        return "p" + pp + c0 + c1
    if target == (actor + 2) % 4:
        return c0 + "p" + pp + c1
    return c0 + c1 + "p" + pp


def encode_daiminkan(actor: int, target: int, pai: str, consumed: List[str]) -> str:
    # consumed must be 3 tiles (excluding called pai).
    c = [mjai_tile_to_two_digits(t) for t in consumed]
    pp = mjai_tile_to_two_digits(pai)

    # See conv.rs mapping for where the 'm' appears.
    if target == (actor + 3) % 4:
        # m + pai + c0 + c1 + c2
        return "m" + pp + c[0] + c[1] + c[2]
    if target == (actor + 2) % 4:
        # c0 + m + pai + c1 + c2
        return c[0] + "m" + pp + c[1] + c[2]
    # c0 + c1 + c2 + m + pai
    return c[0] + c[1] + c[2] + "m" + pp


def encode_reach_discard(pai: str, tsumogiri: bool) -> str:
    # Reach discard item in tenhou6: "r" + two digits, or "r60" for tsumogiri.
    if tsumogiri:
        return "r60"
    return "r" + mjai_tile_to_two_digits(pai)


def encode_ankan(event: Dict[str, Any]) -> str:
    # tenhou6 ankan in discards: "<c0><c1><c2>a<pai>"
    pai = event["pai"]
    consumed = event.get("consumed", [])
    if len(consumed) != 4:
        raise ValueError(f"ankan expected 4 consumed tiles, got {len(consumed)}")
    c = [mjai_tile_to_two_digits(t) for t in consumed[:3]]
    return c[0] + c[1] + c[2] + "a" + mjai_tile_to_two_digits(pai)


def encode_kakan(event: Dict[str, Any]) -> str:
    # tenhou6 kakan in discards: "k" + <pai> + <c0><c1><c2>
    pai = event["pai"]
    consumed = event.get("consumed", [])
    if len(consumed) != 3:
        # Some logs may store 3 consumed tiles; keep strict for now.
        raise ValueError(f"kakan expected 3 consumed tiles, got {len(consumed)}")
    c = [mjai_tile_to_two_digits(t) for t in consumed]
    return "k" + mjai_tile_to_two_digits(pai) + c[0] + c[1] + c[2]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def convert_mjai_jsonl_to_tenhou6(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    names: Optional[List[str]] = None
    aka51 = aka52 = aka53 = False

    # First pass: detect red tiles for rule flags (best-effort).
    for ev in events:
        if ev.get("type") == "start_kyoku" and isinstance(ev.get("tehais"), list):
            for hand in ev["tehais"]:
                for t in hand:
                    if isinstance(t, str) and t.endswith("r") and t.startswith("5") and len(t) == 3:
                        suit = t[1]
                        if suit == "m":
                            aka51 = True
                        elif suit == "p":
                            aka52 = True
                        elif suit == "s":
                            aka53 = True

        if ev.get("type") == "start_game" and isinstance(ev.get("names"), list):
            names = [str(x) for x in ev["names"]]

    if names is None:
        names = ["A", "B", "C", "D"]

    aka = int(aka51 or aka52 or aka53)

    log: List[List[Any]] = []
    kyoku_active = False
    kyoku_obj: List[Any] = []

    # per-kyoku buffers - use dict to allow modification in nested scope
    state: Dict[str, Any] = {
        "kyoku_meta": [0, 0, 0],
        "scores": [25000, 25000, 25000, 25000],
        "dora_indicators": [],
        "ura_indicators": [],
        "haipai": [[] for _ in range(4)],
        "takes": [[] for _ in range(4)],
        "discards": [[] for _ in range(4)],
        "reach_pending": [False, False, False, False],
        "results": None,
    }

    def reset_kyoku() -> None:
        nonlocal kyoku_active, kyoku_obj
        kyoku_obj = []
        kyoku_active = True
        state["kyoku_meta"] = [0, 0, 0]
        state["scores"] = [25000, 25000, 25000, 25000]
        state["dora_indicators"] = []
        state["ura_indicators"] = []
        state["haipai"] = [[] for _ in range(4)]
        state["takes"] = [[] for _ in range(4)]
        state["discards"] = [[] for _ in range(4)]
        state["reach_pending"] = [False, False, False, False]
        state["results"] = None

    def finalize_kyoku() -> None:
        nonlocal kyoku_active, kyoku_obj
        if not kyoku_active:
            return
        if state["results"] is None:
            state["results"] = ["流局", [0, 0, 0, 0]]
        kyoku_obj = [
            state["kyoku_meta"],
            state["scores"],
            state["dora_indicators"],
            state["ura_indicators"],
            state["haipai"][0],
            state["takes"][0],
            state["discards"][0],
            state["haipai"][1],
            state["takes"][1],
            state["discards"][1],
            state["haipai"][2],
            state["takes"][2],
            state["discards"][2],
            state["haipai"][3],
            state["takes"][3],
            state["discards"][3],
            state["results"],
        ]
        log.append(kyoku_obj)
        kyoku_active = False

    for ev in events:
        t = ev.get("type")
        if t == "start_kyoku":
            if kyoku_active:
                finalize_kyoku()
            reset_kyoku()
            bakaze = ev.get("bakaze", "E")
            bakaze_offset = {"E": 0, "S": 4, "W": 8, "N": 12}.get(bakaze, 0)
            kyoku_num = bakaze_offset + int(ev["kyoku"]) - 1
            state["kyoku_meta"] = [kyoku_num, int(ev["honba"]), int(ev["kyotaku"])]
            state["scores"] = [int(x) for x in ev.get("scores", state["scores"])]

            dora_marker = ev.get("dora_marker")
            if isinstance(dora_marker, str) and dora_marker:
                state["dora_indicators"] = [dora_indicator_code(dora_marker)]
            else:
                state["dora_indicators"] = []
            state["ura_indicators"] = []

            tehais = ev.get("tehais", [])
            if isinstance(tehais, list) and len(tehais) == 4:
                for pid in range(4):
                    for tile in tehais[pid]:
                        if tile == "?":
                            continue
                        if not isinstance(tile, str):
                            raise ValueError(f"unexpected tile type in tehais: {tile!r}")
                        state["haipai"][pid].append(mjai_tile_to_tenhou_code(tile))

        elif t in {"tsumo", "chi", "pon", "daiminkan"} and kyoku_active:
            actor = int(ev["actor"])
            if t == "tsumo":
                state["takes"][actor].append(mjai_tile_to_tenhou_code(ev["pai"]))
            elif t == "chi":
                state["takes"][actor].append(encode_chi(actor, int(ev["target"]), ev["pai"], ev["consumed"]))
            elif t == "pon":
                state["takes"][actor].append(encode_pon(actor, int(ev["target"]), ev["pai"], ev["consumed"]))
            elif t == "daiminkan":
                state["takes"][actor].append(
                    encode_daiminkan(actor, int(ev["target"]), ev["pai"], ev["consumed"])
                )

        elif t == "reach" and kyoku_active:
            state["reach_pending"][int(ev["actor"])] = True

        elif t == "dora" and kyoku_active:
            dora_marker = ev.get("dora_marker")
            if isinstance(dora_marker, str) and dora_marker:
                state["dora_indicators"].append(dora_indicator_code(dora_marker))

        elif t == "dahai" and kyoku_active:
            actor = int(ev["actor"])
            pai = ev.get("pai")
            tsumogiri = bool(ev.get("tsumogiri", False))
            if pai is None:
                pai = "?"

            if state["reach_pending"][actor]:
                state["discards"][actor].append(encode_reach_discard(str(pai), tsumogiri))
                state["reach_pending"][actor] = False
            else:
                if tsumogiri:
                    state["discards"][actor].append(60)
                else:
                    state["discards"][actor].append(mjai_tile_to_tenhou_code(str(pai)))

        elif t in {"ankan", "kakan"} and kyoku_active:
            actor = int(ev["actor"])
            if t == "ankan":
                state["discards"][actor].append(encode_ankan(ev))
            else:
                state["discards"][actor].append(encode_kakan(ev))

        elif t == "hora" and kyoku_active:
            sd = ev.get("score_delta") or ev.get("deltas")
            if not isinstance(sd, list) or len(sd) != 4:
                sd = [0, 0, 0, 0]
            actor = int(ev.get("actor", 0))
            target = int(ev.get("target", 0))
            from_who = int(ev.get("from_who", target))
            ura_markers = ev.get("ura_markers")
            if isinstance(ura_markers, list) and ura_markers:
                state["ura_indicators"] = [mjai_tile_to_tenhou_code(m) if not isinstance(m, int) else m for m in ura_markers]
            state["results"] = ["和了", [int(x) for x in sd], [actor, target, from_who, ""]]

        elif t == "ryukyoku" and kyoku_active:
            sd = ev.get("deltas") or ev.get("score_delta")
            if not isinstance(sd, list) or len(sd) != 4:
                sd = [0, 0, 0, 0]
            state["results"] = ["流局", [int(x) for x in sd]]

        elif t == "end_kyoku" and kyoku_active:
            finalize_kyoku()

    if kyoku_active:
        finalize_kyoku()

    return {
        "ver": 2.3,
        "ref": "mjai_export",
        "rule": {
            "disp": "般南喰赤",
            "aka": aka,
            "aka51": int(aka51),
            "aka52": int(aka52),
            "aka53": int(aka53),
        },
        "name": names,
        # tenhou viewer usually tolerates missing sc; keep a placeholder.
        "sc": [0, 0, 0, 0, 0, 0, 0, 0],
        "log": log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    events = load_jsonl(args.input_jsonl)
    out = convert_mjai_jsonl_to_tenhou6(events)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

