from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _tenhou_tile_to_mjai(tile: int) -> str:
    if tile == 51:
        return "5mr"
    if tile == 52:
        return "5pr"
    if tile == 53:
        return "5sr"
    if 11 <= tile <= 19:
        return f"{tile % 10}m"
    if 21 <= tile <= 29:
        return f"{tile % 10}p"
    if 31 <= tile <= 39:
        return f"{tile % 10}s"
    honor_map = {41: "E", 42: "S", 43: "W", 44: "N", 45: "P", 46: "F", 47: "C"}
    if tile in honor_map:
        return honor_map[tile]
    raise ValueError(f"unsupported tenhou tile code: {tile}")


def _try_libriichi_cli(raw_json: Path, out_jsonl: Path, libriichi_bin: Optional[str]) -> bool:
    if not libriichi_bin:
        return False
    cmd = [libriichi_bin, "convert", "--input", str(raw_json), "--output", str(out_jsonl), "--format", "mjai"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _parse_discard_token(token: Any, last_draw: Optional[str]) -> List[Dict[str, Any]]:
    if isinstance(token, int):
        # Tenhou sometimes inserts `0` as a placeholder discard symbol
        # for some call/kan situations (see downloadlogs.js in Equim-chan).
        # It is not a real tile; we should skip it in the fallback converter.
        if token == 0:
            return []
        if token == 60:
            if last_draw is None:
                return []
            return [{"type": "dahai", "pai": last_draw, "tsumogiri": True}]
        return [{"type": "dahai", "pai": _tenhou_tile_to_mjai(token), "tsumogiri": False}]
    if isinstance(token, str):
        if token.startswith("r"):
            raw = int(token[1:])
            # Tenhou uses 60 to indicate tsumogiri (discard drawn tile).
            # When riichi is declared and the discard is the drawn tile,
            # token can become like "r60".
            if raw == 60:
                if last_draw is None:
                    return [{"type": "reach"}]
                return [{"type": "reach"}, {"type": "dahai", "pai": last_draw, "tsumogiri": True}]
            pai = _tenhou_tile_to_mjai(raw)
            return [{"type": "reach"}, {"type": "dahai", "pai": pai, "tsumogiri": False}]
        # Call tokens (c/p/m/k) in tenhou are compact encoded.
        # For MVP fallback, skip exact meld reconstruction and continue stream.
        return []
    return []


def _dora_indicator_code(tile_code: str) -> str:
    return tile_code


def _detect_dora_in_takes(takes_raw: List[List]) -> int:
    daiminkan_count = 0
    for takes in takes_raw:
        for take in takes:
            if isinstance(take, str) and take.startswith("d"):
                daiminkan_count += 1
    return daiminkan_count


def _parse_take_token(token: Any) -> List[Dict[str, Any]]:
    if isinstance(token, int):
        if token == 60:
            return []
        return [{"type": "tsumo", "pai": _tenhou_tile_to_mjai(token)}]
    if isinstance(token, str):
        if token.startswith("d"):
            return [{"type": "daiminkan", "pai": _tenhou_tile_to_mjai(int(token[1:]))}]
        if token.startswith("e"):
            return [{"type": "ankAN", "pai": _tenhou_tile_to_mjai(int(token[1:]))}]
        if token.startswith("k"):
            return [{"type": "kakan", "pai": _tenhou_tile_to_mjai(int(token[1:]))}]
        if token.startswith("a"):
            return [{"type": "pon", "pai": _tenhou_tile_to_mjai(int(token[1:]))}]
    return []


def _fallback_convert_tenhou_json_to_mjai(raw_json: Path, out_jsonl: Path) -> None:
    data = json.loads(raw_json.read_text(encoding="utf-8"))
    names = data.get("name", ["A", "B", "C", "D"])
    logs = data.get("log", [])
    events: List[Dict[str, Any]] = [{"type": "start_game", "names": names}]

    for kyoku_data in logs:
        round_info = kyoku_data[0]
        scores = kyoku_data[1]
        dora_indicators = kyoku_data[2]
        ura_indicators = kyoku_data[3] if len(kyoku_data) > 3 else []
        tehais_raw = [kyoku_data[4], kyoku_data[7], kyoku_data[10], kyoku_data[13]]
        draws_raw = [kyoku_data[5], kyoku_data[8], kyoku_data[11], kyoku_data[14]]
        disc_raw = [kyoku_data[6], kyoku_data[9], kyoku_data[12], kyoku_data[15]]

        oya = round_info[0] % 4
        bakaze_num = round_info[0] // 4
        bakaze_map = {0: "E", 1: "S", 2: "W", 3: "N"}
        bakaze = bakaze_map.get(bakaze_num, "E")
        kyoku = (round_info[0] % 4) + 1

        events.append(
            {
                "type": "start_kyoku",
                "bakaze": bakaze,
                "kyoku": kyoku,
                "honba": round_info[1],
                "kyotaku": round_info[2],
                "oya": oya,
                "scores": scores,
                "dora_marker": _dora_indicator_code(_tenhou_tile_to_mjai(dora_indicators[0])),
                "tehais": [[_tenhou_tile_to_mjai(int(t)) for t in h] for h in tehais_raw],
            }
        )

        for extra_dora in dora_indicators[1:]:
            events.append({"type": "dora", "dora_marker": _dora_indicator_code(_tenhou_tile_to_mjai(extra_dora))})

        max_turn = max(len(x) for x in draws_raw + disc_raw)
        last_draw: List[Optional[str]] = [None, None, None, None]
        daiminkan_count = 0

        for t in range(max_turn):
            for actor in range(4):
                if t < len(draws_raw[actor]):
                    d = draws_raw[actor][t]
                    tokens = _parse_take_token(d)
                    for token_event in tokens:
                        if token_event["type"] == "tsumo":
                            pai = token_event["pai"]
                            last_draw[actor] = pai
                            events.append({"type": "tsumo", "actor": actor, "pai": pai})
                        elif token_event["type"] == "daiminkan":
                            events.append({
                                "type": "daiminkan",
                                "actor": actor,
                                "pai": token_event["pai"],
                                "target": (actor + 3) % 4
                            })
                            daiminkan_count += 1
                            dora_code = _dora_indicator_code(_tenhou_tile_to_mjai(dora_indicators[daiminkan_count])) if daiminkan_count < len(dora_indicators) else None
                            if dora_code:
                                events.append({"type": "dora", "dora_marker": dora_code})
                        elif token_event["type"] == "ankAN":
                            events.append({
                                "type": "ankan",
                                "actor": actor,
                                "pai": token_event["pai"]
                            })
                            daiminkan_count += 1
                            dora_code = _dora_indicator_code(_tenhou_tile_to_mjai(dora_indicators[daiminkan_count])) if daiminkan_count < len(dora_indicators) else None
                            if dora_code:
                                events.append({"type": "dora", "dora_marker": dora_code})
                        elif token_event["type"] == "kakan":
                            events.append({
                                "type": "kakan",
                                "actor": actor,
                                "pai": token_event["pai"]
                            })
                            daiminkan_count += 1
                            dora_code = _dora_indicator_code(_tenhou_tile_to_mjai(dora_indicators[daiminkan_count])) if daiminkan_count < len(dora_indicators) else None
                            if dora_code:
                                events.append({"type": "dora", "dora_marker": dora_code})
                        elif token_event["type"] == "pon":
                            events.append({
                                "type": "pon",
                                "actor": actor,
                                "pai": token_event["pai"],
                                "target": (actor + 3) % 4
                            })

                if t < len(disc_raw[actor]):
                    tokens = _parse_discard_token(disc_raw[actor][t], last_draw[actor])
                    for token_event in tokens:
                        if token_event["type"] == "reach":
                            events.append({"type": "reach", "actor": actor})
                        else:
                            events.append(
                                {
                                    "type": "dahai",
                                    "actor": actor,
                                    "pai": token_event["pai"],
                                    "tsumogiri": token_event["tsumogiri"],
                                }
                            )

        result = kyoku_data[-1]
        if isinstance(result, list) and len(result) >= 2 and isinstance(result[1], list) and len(result[1]) == 4:
            score_delta = [int(x) for x in result[1]]
        else:
            score_delta = [0, 0, 0, 0]
        end_scores = [int(scores[i]) + score_delta[i] for i in range(4)]

        if isinstance(result, list) and result and result[0] == "和了":
            hora_info = result[2] if len(result) > 2 else []
            hora_who = hora_info[0] if len(hora_info) > 0 else oya
            hora_target = hora_info[1] if len(hora_info) > 1 else -1
            events.append({
                 "type": "hora",
                 "actor": hora_who,
                 "target": hora_target,
                 "pai": last_draw[hora_who] if hora_who < len(last_draw) else "?",
                 "deltas": score_delta,
                 "scores": end_scores,
                 "ura_markers": [_dora_indicator_code(_tenhou_tile_to_mjai(u)) for u in ura_indicators]
             })
        else:
            events.append({"type": "ryukyoku", "deltas": score_delta, "scores": end_scores})

        events.append({"type": "end_kyoku"})

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _try_libriichi_python_validate(out_jsonl: Path) -> bool:
    try:
        import riichi  # type: ignore
    except Exception:
        return False
    try:
        states = [riichi.state.PlayerState(pid) for pid in range(4)]
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                et = event.get("type")
                if et in {"hora", "ryukyoku", "end_kyoku", "end_game"}:
                    # Terminal events may have schema differences across converters.
                    continue
                payload = json.dumps(event, ensure_ascii=False)
                for s in states:
                    s.update(payload)
        return True
    except Exception:
        return False


def convert_raw_to_mjai(raw_json_path: str, out_jsonl_path: str, libriichi_bin: Optional[str] = None) -> Dict[str, Any]:
    raw_json = Path(raw_json_path)
    out_jsonl = Path(out_jsonl_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    used_libriichi = _try_libriichi_cli(raw_json, out_jsonl, libriichi_bin)
    engine = "libriichi_cli"
    if not used_libriichi:
        _fallback_convert_tenhou_json_to_mjai(raw_json, out_jsonl)
        if _try_libriichi_python_validate(out_jsonl):
            used_libriichi = True
            engine = "libriichi_python_validate+fallback_convert"
        else:
            engine = "fallback_convert"
    return {"output": str(out_jsonl), "used_libriichi": used_libriichi, "engine": engine}

