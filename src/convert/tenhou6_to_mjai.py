"""Convert tenhou6 JSON to mjai jsonl format.

Reference: parse_mjlog.py (mjlog XML -> mjai) for the reverse direction logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Tenhou tile code to mjai tile string
# Reference: parse参考.txt (Rust tid_to_mjai logic)
HONORS = ["E", "S", "W", "N", "P", "F", "C"]


def tenhou_tile_to_mjai(tile: int) -> str:
    """Convert tenhou tile code to mjai tile string.

    Logic from Rust:
    - tid == 16 -> "5mr" (red 5 man)
    - tid == 52 -> "5pr" (red 5 pin)
    - tid == 88 -> "5sr" (red 5 sou)
    - kind = tid / 36
      - kind < 3: suit (0=m, 1=p, 2=s), num = (tid % 36) / 4 + 1
      - kind >= 3: honors, offset = tid - 108, num = offset / 4 + 1
    """
    # Check for red 5s
    if tile == 16:
        return "5mr"
    if tile == 52:
        return "5pr"
    if tile == 88:
        return "5sr"

    kind = tile // 36
    if kind < 3:
        suit_char = ["m", "p", "s"][kind]
        offset = tile % 36
        num = offset // 4 + 1
        return f"{num}{suit_char}"
    else:
        offset = tile - 108
        num = offset // 4 + 1
        if 1 <= num <= 7:
            return HONORS[num - 1]
        else:
            return f"{num}z"  # Should not happen in normal tenhou6 data


def mjai_tile_to_tenhou(tile: str) -> int:
    """Convert mjai tile string to tenhou tile code."""
    # Handle aka tiles
    if tile == "5mr":
        return 16
    if tile == "5pr":
        return 52
    if tile == "5sr":
        return 88

    # Parse tile string
    if tile.endswith("m"):
        kind = 0
        num = int(tile[:-1])
    elif tile.endswith("p"):
        kind = 1
        num = int(tile[:-1])
    elif tile.endswith("s"):
        kind = 2
        num = int(tile[:-1])
    elif tile in HONORS:
        kind = 3
        num = HONORS.index(tile) + 1
    else:
        raise ValueError(f"unsupported mjai tile: {tile}")

    return kind * 36 + (num - 1) * 4


def _parse_call_token(token: str) -> Optional[Dict[str, Any]]:
    """Parse tenhou call token like 'c010203' (chi), 'p414141' (pon), 'k3333' (kan), etc."""
    if not isinstance(token, str):
        return None

    # Tokens are encoded as 2-char pairs per tile code
    def parse_tiles(s: str) -> List[int]:
        return [int(s[i:i+2]) for i in range(0, len(s), 2)]

    if token.startswith("c"):  # Chi: c + consumed tiles (2-char each) + called tile at end
        tiles = parse_tiles(token[1:])
        if len(tiles) < 2:
            return None
        called = tiles[-1]
        consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
        return {"type": "chi", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}

    if token.startswith("p"):  # Pon: p + 2 consumed tiles + called tile at end
        tiles = parse_tiles(token[1:])
        if len(tiles) < 2:
            return None
        called = tiles[-1]
        consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
        return {"type": "pon", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}

    if token.startswith("m"):  # MinKan (daiminkan): m + 3 consumed + called at end
        tiles = parse_tiles(token[1:])
        if len(tiles) < 3:
            return None
        called = tiles[-1]
        consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
        return {"type": "daiminkan", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}

    if token.startswith("k"):  # KaKan: k + 3 same tiles
        tiles = parse_tiles(token[1:])
        pai = tenhou_tile_to_mjai(tiles[0])
        consumed = [tenhou_tile_to_mjai(t) for t in tiles]
        return {"type": "kakan", "pai": pai, "consumed": consumed}

    if token.startswith("e"):  # AnKan: e + 4 same tiles
        tiles = parse_tiles(token[1:])
        pai = tenhou_tile_to_mjai(tiles[0])
        consumed = [tenhou_tile_to_mjai(t) for t in tiles]
        return {"type": "ankan", "pai": pai, "consumed": consumed}

    return None


def convert_tenhou6_to_mjai(tenhou6_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert tenhou6 JSON dict to list of mjai events.

    Args:
        tenhou6_data: Parsed tenhou6 JSON with fields: ver, ref, log, ratingc, rule, lobby, dan, rate, sx, sc, name

    Returns:
        List of mjai event dicts
    """
    events: List[Dict[str, Any]] = []

    names = tenhou6_data.get("name", ["P0", "P1", "P2", "P3"])
    logs = tenhou6_data.get("log", [])

    # start_game event
    events.append({
        "type": "start_game",
        "names": names,
        "kyoku_first": 0,  # tenhou starts from East
        "aka_flag": True,  # default, will be overridden if rule info available
    })

    for kyoku_data in logs:
        if len(kyoku_data) < 6:
            continue

        round_info = kyoku_data[0]  # [oya_with_round, honba, kyotaku, ...]
        scores = kyoku_data[1]
        dora_indicators = kyoku_data[2] if len(kyoku_data) > 2 else []
        ura_indicators = kyoku_data[3] if len(kyoku_data) > 3 else []

        tehais_raw = [
            kyoku_data[4],  # player 0
            kyoku_data[7],  # player 1
            kyoku_data[10], # player 2
            kyoku_data[13], # player 3
        ]
        draws_raw = [
            kyoku_data[5],
            kyoku_data[8],
            kyoku_data[11],
            kyoku_data[14],
        ]
        disc_raw = [
            kyoku_data[6],
            kyoku_data[9],
            kyoku_data[12],
            kyoku_data[15],
        ]

        oya = round_info[0] % 4
        round_num = round_info[0] // 4
        bakaze_map = {0: "E", 1: "S", 2: "W", 3: "N"}
        bakaze = bakaze_map.get(round_num % 4, "E")
        kyoku = (round_info[0] % 4) + 1
        honba = round_info[1] if len(round_info) > 1 else 0
        kyotaku = round_info[2] if len(round_info) > 2 else 0

        # Convert tehais - sort each hand using same logic as parse_mjlog.py (xor 3)
        tehais = []
        for hand in tehais_raw:
            converted = []
            for t in hand:
                if isinstance(t, int) and t > 0 and t != 60:
                    converted.append(tenhou_tile_to_mjai(t))
                elif isinstance(t, str) and not t.startswith("r") and not t.startswith(("c", "p", "m", "k", "e", "d", "a")):
                    # Skip non-tile strings like "r24" (riichi) or other markers
                    pass
            tehais.append(converted)

        # start_kyoku event
        events.append({
            "type": "start_kyoku",
            "bakaze": bakaze,
            "dora_marker": tenhou_tile_to_mjai(dora_indicators[0]) if dora_indicators else "?" ,
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        })

        # Process game flow
        last_draw: List[Optional[str]] = [None, None, None, None]
        need_dora: List[int] = []
        pon_history: Dict[int, List[str]] = {}  # tile_base -> [called, consumed1, consumed2]
        reach_accepted_count = 0
        ura_markers: List[str] = []

        max_turn = max(len(draws_raw[i]) + len(disc_raw[i]) for i in range(4))

        # Parse all discards to track tsumogiri
        all_discards: List[List[Any]] = [[] for _ in range(4)]

        # Process events turn by turn
        turn_idx = 0
        actor = 0
        while turn_idx < max_turn * 4 * 2:  # Simplified loop
            # Actually we need to process player by player, turn by turn
            break

        # Simplified approach: interleave draws and discards per turn for each player
        for t in range(max_turn):
            for actor in range(4):
                # Handle draw/take event
                if t < len(draws_raw[actor]):
                    draw = draws_raw[actor][t]
                    tokens = _parse_draw_token(draw, last_draw[actor])
                    for tok in tokens:
                        if tok["type"] == "tsumo":
                            last_draw[actor] = tok["pai"]
                            events.append({"type": "tsumo", "actor": actor, "pai": tok["pai"]})
                        elif tok["type"] == "pon":
                            events.append({
                                "type": "pon",
                                "actor": actor,
                                "target": (actor + 3) % 4,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            base = mjai_tile_to_tenhou(tok["pai"]) // 4
                            pon_history[base] = [tok["pai"]] + tok["consumed"]
                        elif tok["type"] == "chi":
                            events.append({
                                "type": "chi",
                                "actor": actor,
                                "target": (actor + 3) % 4,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                        elif tok["type"] == "ankan":
                            events.append({
                                "type": "ankan",
                                "actor": actor,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            need_dora.append(len(events) - 1)
                        elif tok["type"] == "daiminkan":
                            events.append({
                                "type": "daiminkan",
                                "actor": actor,
                                "target": (actor + 3) % 4,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            need_dora.append(len(events) - 1)
                        elif tok["type"] == "kakan":
                            events.append({
                                "type": "kakan",
                                "actor": actor,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            base = mjai_tile_to_tenhou(tok["pai"]) // 4
                            if base in pon_history:
                                pon_history[base] = []
                            need_dora.append(len(events) - 1)

                # Handle discard event
                if t < len(disc_raw[actor]):
                    disc = disc_raw[actor][t]
                    tokens = _parse_discard_token(disc, last_draw[actor])
                    for tok in tokens:
                        if tok["type"] == "reach":
                            events.append({"type": "reach", "actor": actor})
                        elif tok["type"] == "reach_accepted":
                            reach_accepted_count += 1
                            if reach_accepted_count < 4:
                                events.append({"type": "reach_accepted", "actor": actor})
                        elif tok["type"] == "dahai":
                            events.append({
                                "type": "dahai",
                                "actor": actor,
                                "pai": tok["pai"],
                                "tsumogiri": tok["tsumogiri"],
                            })
                        elif tok["type"] == "kakan":
                            # 加杠：自己摸牌后进行，actor是自己
                            events.append({
                                "type": "kakan",
                                "actor": actor,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            need_dora.append(len(events) - 1)
                            # 清空pon_history中对应的entry
                            base = mjai_tile_to_tenhou(tok["pai"]) // 4
                            if base in pon_history:
                                pon_history[base] = []
                        elif tok["type"] == "ankan":
                            # 暗杠：自己摸牌后进行，actor是自己
                            events.append({
                                "type": "ankan",
                                "actor": actor,
                                "pai": tok["pai"],
                                "consumed": tok["consumed"],
                            })
                            need_dora.append(len(events) - 1)

        # Handle result (hora or ryukyoku)
        result = kyoku_data[-1] if len(kyoku_data) > 0 else []

        if isinstance(result, list) and len(result) >= 2:
            if result[0] == "和了":
                # 和了: [type, score_delta, hora_info, yaku_info]
                score_delta = result[1] if len(result) > 1 and isinstance(result[1], list) else [0, 0, 0, 0]
                hora_info = result[2] if len(result) > 2 else []
                if isinstance(hora_info, list) and len(hora_info) >= 2:
                    hora_actor = hora_info[0]
                    hora_target = hora_info[1]
                else:
                    hora_actor = oya
                    hora_target = oya

                # Collect ura markers if available
                ura_markers = []
                if ura_indicators:
                    ura_markers = [tenhou_tile_to_mjai(u) for u in ura_indicators]

                events.append({
                    "type": "hora",
                    "actor": hora_actor,
                    "target": hora_target,
                    "deltas": score_delta,
                    "ura_markers": ura_markers,
                })
            elif result[0] == "流局":
                # 流局: [type, score_delta]
                score_delta = result[1] if len(result) > 1 and isinstance(result[1], list) else [0, 0, 0, 0]
                events.append({
                    "type": "ryukyoku",
                    "deltas": score_delta,
                })

        events.append({"type": "end_kyoku"})

    events.append({"type": "end_game"})
    return events


def _parse_draw_token(token: Any, last_draw: Optional[str]) -> List[Dict[str, Any]]:
    """Parse a tenhou draw/take token into mjai events."""
    if isinstance(token, int):
        if token == 0 or token == 60:
            return []
        return [{"type": "tsumo", "pai": tenhou_tile_to_mjai(token)}]

    if isinstance(token, str):
        if token.startswith("r"):
            # Riichi declaration followed by drawn tile
            return []
        # Tokens are encoded as 2-char pairs per tile code
        def parse_tiles(s: str) -> List[int]:
            return [int(s[i:i+2]) for i in range(0, len(s), 2)]
        if token.startswith("a"):  # Pon (called from left)
            tiles = parse_tiles(token[1:])
            pai = tenhou_tile_to_mjai(tiles[-1])
            consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
            return [{"type": "pon", "pai": pai, "consumed": consumed}]
        if token.startswith("c"):  # Chi
            tiles = parse_tiles(token[1:])
            if len(tiles) < 2:
                return []
            called = tiles[-1]
            consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
            return [{"type": "chi", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}]
        if token.startswith("p"):  # Pon (called from opposite or same)
            tiles = parse_tiles(token[1:])
            if len(tiles) < 2:
                return []
            called = tiles[-1]
            consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
            return [{"type": "pon", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}]
        if token.startswith("m"):  # Daiminkan
            tiles = parse_tiles(token[1:])
            if len(tiles) < 3:
                return []
            called = tiles[-1]
            consumed = [tenhou_tile_to_mjai(t) for t in tiles[:-1]]
            return [{"type": "daiminkan", "pai": tenhou_tile_to_mjai(called), "consumed": consumed}]
        if token.startswith("k"):  # KaKan (加杠)
            tiles = parse_tiles(token[1:])
            # kakan: 3 consumed + 1 called = 4 tiles
            # consumed = tiles[:-1], called = tiles[-1]
            called = tiles[-1]
            consumed = tiles[:-1]
            pai = tenhou_tile_to_mjai(called)
            consumed_mjai = [tenhou_tile_to_mjai(t) for t in consumed]
            return [{"type": "kakan", "pai": pai, "consumed": consumed_mjai}]
        if token.startswith("e"):  # AnKan
            tiles = parse_tiles(token[1:])
            pai = tenhou_tile_to_mjai(tiles[0])
            consumed = [tenhou_tile_to_mjai(t) for t in tiles]
            return [{"type": "ankan", "pai": pai, "consumed": consumed}]

    return []


def _parse_discard_token(token: Any, last_draw: Optional[str]) -> List[Dict[str, Any]]:
    """Parse a tenhou discard token into mjai events."""
    if isinstance(token, int):
        if token == 0:
            return []
        if token == 60:
            if last_draw is None:
                return []
            return [{"type": "dahai", "pai": last_draw, "tsumogiri": True}]
        return [{"type": "dahai", "pai": tenhou_tile_to_mjai(token), "tsumogiri": False}]

    if isinstance(token, str):
        # Tokens are encoded as 2-char pairs per tile code
        def parse_tiles(s: str) -> List[int]:
            return [int(s[i:i+2]) for i in range(0, len(s), 2)]

        if token.startswith("r"):
            raw = int(token[1:])
            if raw == 60:
                if last_draw is None:
                    return [{"type": "reach"}]
                return [{"type": "reach"}, {"type": "dahai", "pai": last_draw, "tsumogiri": True}]
            return [{"type": "reach"}, {"type": "dahai", "pai": tenhou_tile_to_mjai(raw), "tsumogiri": False}]

        if token.startswith("k"):  # KaKan (加杠)
            tiles = parse_tiles(token[1:])
            # kakan: 3 consumed + 1 called = 4 tiles
            called = tiles[-1]
            consumed = tiles[:-1]
            pai = tenhou_tile_to_mjai(called)
            consumed_mjai = [tenhou_tile_to_mjai(t) for t in consumed]
            return [{"type": "kakan", "pai": pai, "consumed": consumed_mjai}]

        if token.startswith("e"):  # AnKan (暗杠) - sometimes appears in discards
            tiles = parse_tiles(token[1:])
            # ankan: 4 tiles all consumed (暗杠4张)
            called = tiles[-1]
            consumed = tiles[:-1]
            pai = tenhou_tile_to_mjai(called)
            consumed_mjai = [tenhou_tile_to_mjai(t) for t in consumed]
            return [{"type": "ankan", "pai": pai, "consumed": consumed_mjai}]

    return []


def load_tenhou6(path: str | Path) -> Dict[str, Any]:
    """Load tenhou6 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_tenhou6_file(input_path: str | Path, output_path: str | Path) -> None:
    """Convert a tenhou6 JSON file to mjai jsonl format.

    Args:
        input_path: Path to tenhou6 JSON file
        output_path: Path to output mjai jsonl file
    """
    data = load_tenhou6(input_path)
    events = convert_tenhou6_to_mjai(data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def batch_convert_tenhou6_directory(
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool = False,
) -> tuple[int, int]:
    """Batch convert all tenhou6 JSON files in a directory.

    Args:
        input_dir: Directory containing tenhou6 JSON files
        output_dir: Directory to output mjai jsonl files
        skip_existing: If True, skip files that already exist

    Returns:
        (success_count, failed_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        return 0, 0

    success = 0
    failed = 0

    for json_file in json_files:
        out_file = output_dir / f"{json_file.stem}.jsonl"
        if skip_existing and out_file.exists():
            continue

        try:
            convert_tenhou6_file(json_file, out_file)
            success += 1
        except Exception as e:
            print(f"[ERROR] {json_file.name}: {e}")
            failed += 1

    return success, failed


def batch_convert_all_tenhou6(
    base_input_dir: Path,
    base_output_dir: Path,
    ds_list: list[str] | None = None,
    skip_existing: bool = False,
) -> tuple[int, int]:
    """Batch convert all tenhou6 directories.

    Args:
        base_input_dir: Base directory containing dsX subdirectories (e.g., dataset/tenhou6)
        base_output_dir: Base output directory (e.g., artifacts/converted)
        ds_list: List of ds names to process (e.g., ["ds1", "ds2"]). If None, process all.
        skip_existing: If True, skip files that already exist

    Returns:
        (total_success, total_failed)
    """
    base_input_dir = Path(base_input_dir)
    base_output_dir = Path(base_output_dir)

    if ds_list is None:
        # Find all ds directories
        ds_list = [d.name for d in base_input_dir.iterdir() if d.is_dir() and d.name.startswith("ds")]

    total_success = 0
    total_failed = 0

    for ds_name in sorted(ds_list):
        input_dir = base_input_dir / ds_name
        output_dir = base_output_dir / ds_name

        if not input_dir.exists():
            print(f"[WARN] Directory not found: {input_dir}")
            continue

        print(f"Processing {ds_name}...")
        s, f = batch_convert_tenhou6_directory(input_dir, output_dir, skip_existing)
        total_success += s
        total_failed += f
        print(f"  {ds_name}: {s} succeeded, {f} failed")

    return total_success, total_failed
