"""tenhou6 JSON → mjai JSONL 转换工具。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

CONVLOG_BIN = (
    Path(__file__).parent.parent.parent
    / "third_party"
    / "mjai-reviewer"
    / "target"
    / "release"
    / "convlog"
)

_TSUMOGIRI = 60
_ROUND_BY_OFFSET = {0: "E", 4: "S", 8: "W", 12: "N"}
_HONORS_BY_CODE = {41: "E", 42: "S", 43: "W", 44: "N", 45: "P", 46: "F", 47: "C"}


def _tile_from_tenhou6(value: int | str) -> str:
    code = int(value)
    if code in _HONORS_BY_CODE:
        return _HONORS_BY_CODE[code]
    if code == 51:
        return "5mr"
    if code == 52:
        return "5pr"
    if code == 53:
        return "5sr"
    number = code % 10
    suit = code // 10
    if suit == 1:
        return f"{number}m"
    if suit == 2:
        return f"{number}p"
    if suit == 3:
        return f"{number}s"
    raise ValueError(f"invalid tenhou6 tile code: {value!r}")


def _meld_codes(raw: str) -> list[str]:
    compact = raw.replace("c", "").replace("p", "").replace("m", "").replace("a", "").replace("k", "")
    if len(compact) % 2 != 0:
        raise ValueError(f"invalid tenhou6 meld string: {raw!r}")
    return [compact[i : i + 2] for i in range(0, len(compact), 2)]


def _called_and_consumed(raw: str, marker: str) -> tuple[str, list[str]]:
    marker_pos = raw.index(marker)
    called_code = raw[marker_pos + 1 : marker_pos + 3]
    codes = _meld_codes(raw)
    consumed_codes = list(codes)
    try:
        consumed_codes.remove(called_code)
    except ValueError:
        pass
    return _tile_from_tenhou6(called_code), [_tile_from_tenhou6(code) for code in consumed_codes]


def _decode_discard(raw: int | str, last_draw: str | None) -> tuple[bool, str, bool]:
    reach = False
    value: int | str = raw
    if isinstance(raw, str) and raw.startswith("r"):
        reach = True
        value = raw[1:]
    code = int(value)
    if code == _TSUMOGIRI:
        if not last_draw:
            raise ValueError("tsumogiri discard without a previous draw")
        return reach, last_draw, True
    return reach, _tile_from_tenhou6(code), False


def _rule_has_aka(rule: dict[str, Any]) -> bool:
    return bool(rule.get("aka") or rule.get("aka51") or rule.get("aka52") or rule.get("aka53"))


def _start_kyoku_event(kyoku: list[Any], names: list[str], rule: dict[str, Any]) -> dict[str, Any]:
    del names, rule
    meta = kyoku[0]
    round_index = int(meta[0])
    round_base = (round_index // 4) * 4
    return {
        "type": "start_kyoku",
        "bakaze": _ROUND_BY_OFFSET.get(round_base, "E"),
        "dora_marker": _tile_from_tenhou6(kyoku[2][0]) if len(kyoku) > 2 and kyoku[2] else None,
        "kyoku": round_index % 4 + 1,
        "honba": int(meta[1]) if len(meta) > 1 else 0,
        "kyotaku": int(meta[2]) if len(meta) > 2 else 0,
        "oya": round_index % 4,
        "scores": [int(score) for score in kyoku[1]],
        "tehais": [[_tile_from_tenhou6(tile) for tile in kyoku[4 + seat * 3]] for seat in range(4)],
    }


def _result_events(result: list[Any]) -> list[dict[str, Any]]:
    if not result:
        return []
    kind = result[0]
    deltas = [int(delta) for delta in (result[1] if len(result) > 1 and isinstance(result[1], list) else [0, 0, 0, 0])]
    if kind == "流局":
        return [{"type": "ryukyoku", "reason": "ryukyoku", "deltas": deltas}]
    if kind != "和了":
        return []
    events: list[dict[str, Any]] = []
    for detail in result[2:]:
        if not isinstance(detail, list) or len(detail) < 2:
            continue
        actor = int(detail[0])
        target = int(detail[1])
        events.append(
            {
                "type": "hora",
                "actor": actor,
                "target": target,
                "deltas": deltas,
            }
        )
    return events


def _convert_kyoku_to_events(kyoku: list[Any], names: list[str], rule: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [_start_kyoku_event(kyoku, names, rule)]
    oya = int(events[0]["oya"])
    takes = [list(kyoku[5 + seat * 3]) for seat in range(4)]
    discards = [list(kyoku[6 + seat * 3]) for seat in range(4)]
    take_idx = [0, 0, 0, 0]
    discard_idx = [0, 0, 0, 0]
    last_draw: list[str | None] = [None, None, None, None]
    last_discard_actor: int | None = None
    actor = oya

    def has_pending() -> bool:
        return any(take_idx[seat] < len(takes[seat]) or discard_idx[seat] < len(discards[seat]) for seat in range(4))

    def next_call_after(discarder: int) -> int | None:
        for offset in (1, 2, 3):
            candidate = (discarder + offset) % 4
            if take_idx[candidate] < len(takes[candidate]) and isinstance(takes[candidate][take_idx[candidate]], str):
                return candidate
        return None

    guard = 0
    while has_pending():
        guard += 1
        if guard > 1000:
            raise RuntimeError("tenhou6 conversion exceeded action guard")
        progressed = False

        if take_idx[actor] < len(takes[actor]):
            take = takes[actor][take_idx[actor]]
            if isinstance(take, str):
                target = last_discard_actor if last_discard_actor is not None else (actor + 3) % 4
                if take.startswith("c"):
                    pai, consumed = _called_and_consumed(take, "c")
                    events.append({"type": "chi", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
                elif "p" in take:
                    pai, consumed = _called_and_consumed(take, "p")
                    events.append({"type": "pon", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
                elif "m" in take:
                    pai, consumed = _called_and_consumed(take, "m")
                    events.append({"type": "daiminkan", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
                else:
                    raise ValueError(f"unsupported tenhou6 take meld: {take!r}")
                take_idx[actor] += 1
                progressed = True
            else:
                pai = _tile_from_tenhou6(take)
                events.append({"type": "tsumo", "actor": actor, "pai": pai})
                last_draw[actor] = pai
                take_idx[actor] += 1
                progressed = True

        if discard_idx[actor] < len(discards[actor]):
            discard = discards[actor][discard_idx[actor]]
            if isinstance(discard, str) and "a" in discard:
                consumed = [_tile_from_tenhou6(code) for code in _meld_codes(discard)]
                events.append({"type": "ankan", "actor": actor, "consumed": consumed})
                discard_idx[actor] += 1
                progressed = True
                continue
            if isinstance(discard, str) and discard.startswith("k"):
                codes = _meld_codes(discard)
                events.append(
                    {
                        "type": "kakan",
                        "actor": actor,
                        "pai": _tile_from_tenhou6(codes[0]),
                        "consumed": [_tile_from_tenhou6(code) for code in codes[1:]],
                    }
                )
                discard_idx[actor] += 1
                progressed = True
                continue

            reach, pai, tsumogiri = _decode_discard(discard, last_draw[actor])
            if reach:
                events.append({"type": "reach", "actor": actor})
            events.append({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri})
            last_draw[actor] = None
            last_discard_actor = actor
            discard_idx[actor] += 1
            progressed = True

            caller = next_call_after(actor)
            actor = caller if caller is not None else (actor + 1) % 4
            continue

        if not progressed:
            actor = (actor + 1) % 4

    events.extend(_result_events(kyoku[-1] if kyoku else []))
    events.append({"type": "end_kyoku"})
    return events


def tenhou6_to_mjai_events(t6_json: dict[str, Any]) -> list[dict[str, Any]]:
    names = [str(name) for name in t6_json.get("name", ["A", "B", "C", "D"])[:4]]
    while len(names) < 4:
        names.append(f"P{len(names)}")
    rule = t6_json.get("rule", {}) if isinstance(t6_json.get("rule"), dict) else {}
    events: list[dict[str, Any]] = [
        {
            "type": "start_game",
            "names": names,
            "kyoku_first": 0,
            "aka_flag": _rule_has_aka(rule),
        }
    ]
    for kyoku in t6_json.get("log", []):
        if isinstance(kyoku, list) and len(kyoku) >= 17:
            events.extend(_convert_kyoku_to_events(kyoku, names, rule))
    events.append({"type": "end_game"})
    return events


def _convlog_candidates() -> list[Path]:
    if os.name == "nt":
        return [CONVLOG_BIN.with_suffix(".exe"), CONVLOG_BIN]
    return [CONVLOG_BIN]


def tenhou6_to_mjson(t6_json: dict, output_path: Path) -> bool:
    """Write tenhou6 JSON to an mjai JSONL file. Returns success."""
    try:
        events = tenhou6_to_mjai_events(t6_json)
        output_path.write_text(
            "\n".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in events) + "\n",
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        print(f"  WARN python tenhou6 converter failed, trying convlog: {exc}")

    tmp = output_path.with_suffix(".tmp.json")
    try:
        tmp.write_text(json.dumps(t6_json, ensure_ascii=False), encoding="utf-8")
        convlog_bin = next((candidate for candidate in _convlog_candidates() if candidate.exists()), CONVLOG_BIN)
        result = subprocess.run(
            [str(convlog_bin), str(tmp), str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR convlog: {result.stderr.strip()}")
            return False
        return True
    finally:
        if tmp.exists():
            tmp.unlink()
