#!/usr/bin/env python3
"""
Verify that tenhou6 raw operations match the converted mjai operations.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Tile conversion based on parse参考.txt
def tenhou_tile_to_mjai(tile: int) -> str:
    if tile == 16: return "5mr"
    if tile == 52: return "5pr"
    if tile == 88: return "5sr"
    kind = tile // 36
    if kind < 3:
        suit_char = ["m", "p", "s"][kind]
        num = (tile % 36) // 4 + 1
        return f"{num}{suit_char}"
    else:
        honors = ["E", "S", "W", "N", "P", "F", "C"]
        num = (tile - 108) // 4 + 1
        return honors[num - 1]


@dataclass
class TenhouOp:
    """Represents an operation extracted from tenhou6 raw data"""
    kyoku: int
    honba: int
    actor: int
    optype: str  # chi, pon, daiminkan, kakan, ankan, reach
    pai: Optional[str] = None
    consumed: list = field(default_factory=list)
    target: Optional[int] = None


def parse_call_token(token: str, actor: int) -> Optional[TenhouOp]:
    """Parse a call token from tenhou6 (c, p, m, k, e prefixes)"""
    if not token or len(token) < 2:
        return None

    prefix = token[0]
    content = token[1:]

    # Parse tiles as 2-character pairs
    def parse_tiles(s):
        return [int(s[i:i+2]) for i in range(0, len(s), 2)]

    try:
        tiles = parse_tiles(content)
    except:
        return None

    if prefix == "c":  # Chi
        called = tiles[-1]
        consumed = tiles[:-1]
        # Chi: figure out target from the sequence
        return TenhouOp(
            kyoku=0, honba=0, actor=actor, optype="chi",
            pai=tenhou_tile_to_mjai(called),
            consumed=[tenhou_tile_to_mjai(t) for t in consumed]
        )
    elif prefix == "p":  # Pon
        called = tiles[-1]
        consumed = tiles[:-1]
        return TenhouOp(
            kyoku=0, honba=0, actor=actor, optype="pon",
            pai=tenhou_tile_to_mjai(called),
            consumed=[tenhou_tile_to_mjai(t) for t in consumed]
        )
    elif prefix == "m":  # Daiminkan
        called = tiles[-1]
        consumed = tiles[:-1]
        return TenhouOp(
            kyoku=0, honba=0, actor=actor, optype="daiminkan",
            pai=tenhou_tile_to_mjai(called),
            consumed=[tenhou_tile_to_mjai(t) for t in consumed]
        )
    elif prefix == "k":  # Kakan
        called = tiles[-1]
        consumed = tiles[:-1]
        return TenhouOp(
            kyoku=0, honba=0, actor=actor, optype="kakan",
            pai=tenhou_tile_to_mjai(called),
            consumed=[tenhou_tile_to_mjai(t) for t in consumed]
        )
    elif prefix == "e":  # Ankan
        called = tiles[-1]
        consumed = tiles[:-1]
        return TenhouOp(
            kyoku=0, honba=0, actor=actor, optype="ankan",
            pai=tenhou_tile_to_mjai(called),
            consumed=[tenhou_tile_to_mjai(t) for t in consumed]
        )

    return None


def extract_tenhou6_ops(json_path: Path) -> list[TenhouOp]:
    """Extract all operations from a tenhou6 JSON file"""
    with open(json_path) as f:
        data = json.load(f)

    ops = []
    log = data.get("log", [])

    for kyoku_idx, kyoku in enumerate(log):
        if not kyoku or len(kyoku) < 6:
            continue

        # Extract seed for honba
        seed = kyoku[0] if kyoku[0] else []
        honba = seed[1] if len(seed) > 1 else 0

        # draws_raw is index 3, disc_raw is index 4
        draws_raw = kyoku[3] if len(kyoku) > 3 else []
        disc_raw = kyoku[4] if len(kyoku) > 4 else []

        # Process draws_raw for chi, pon, daiminkan, kakan, ankan
        for item in draws_raw:
            if isinstance(item, str) and len(item) >= 2:
                # Try to determine actor from position in the draws sequence
                # This is tricky because we don't have direct actor info in the raw
                pass

        # Process disc_raw for reach and kakan/ankan that appear in discards
        for item in disc_raw:
            if isinstance(item, str):
                if item == "r":
                    # Reach - need actor info
                    pass
                elif item[0] in "cpmek":
                    # These are call tokens in discards
                    pass

        # More detailed parsing: in tenhou6, the draws/disc are per-player interleaved
        # Format: [player0_draws, player1_draws, player2_draws, player3_draws,
        #          player0_disc, player1_disc, player2_disc, player3_disc, ...]

        # Actually looking at the data, it seems to be:
        # [seed, scores, haipai, draws_raw, disc_raw, ...]
        # where draws_raw and disc_raw are arrays that CONTAIN all players' draws/discs mixed
        # and the string tokens include actor info implicitly

        # Let me re-examine by looking at specific examples
        for item in draws_raw:
            if isinstance(item, str) and len(item) >= 2:
                prefix = item[0]
                if prefix in "cpmek":
                    # These are call tokens - actor is encoded in the draw order
                    # We need to track whose turn it is
                    pass

    return ops


def extract_mjai_ops(mjai_path: Path) -> list:
    """Extract operations from mjai JSONL file"""
    ops = []
    with open(mjai_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            etype = event.get("type", "")
            if etype in ("chi", "pon", "daiminkan", "kakan", "ankan", "reach", "reach_accepted"):
                ops.append(event)
    return ops


def analyze_tenhou6_file(json_path: Path):
    """Analyze a tenhou6 file and print operations"""
    with open(json_path) as f:
        data = json.load(f)

    log = data.get("log", [])
    print(f"File: {json_path.name}")
    print(f"Number of kyokus: {len(log)}")

    all_ops = []

    for kyoku_idx, kyoku in enumerate(log):
        if not kyoku or len(kyoku) < 6:
            continue

        seed = kyoku[0] if kyoku[0] else []
        honba = seed[1] if len(seed) > 1 else 0
        draws_raw = kyoku[3] if len(kyoku) > 3 else []
        disc_raw = kyoku[4] if len(kyoku) > 4 else []

        # Count string tokens
        chi_count = 0
        pon_count = 0
        daiminkan_count = 0
        kakan_count = 0
        ankan_count = 0
        reach_count = 0

        all_tokens = []
        for item in draws_raw:
            if isinstance(item, str):
                all_tokens.append(("draw", item))
        for item in disc_raw:
            if isinstance(item, str):
                all_tokens.append(("disc", item))

        for source, item in all_tokens:
            if not isinstance(item, str):
                continue
            if item == "r":
                reach_count += 1
            elif item.startswith("c"):
                chi_count += 1
            elif item.startswith("p"):
                pon_count += 1
            elif item.startswith("m"):
                daiminkan_count += 1
            elif item.startswith("k"):
                kakan_count += 1
            elif item.startswith("e"):
                ankan_count += 1

        if any([chi_count, pon_count, daiminkan_count, kakan_count, ankan_count, reach_count]):
            print(f"  Kyoku {kyoku_idx + 1}: chi={chi_count}, pon={pon_count}, daiminkan={daiminkan_count}, kakan={kakan_count}, ankan={ankan_count}, reach={reach_count}")

    return all_ops


def count_ops_in_mjai(mjai_path: Path):
    """Count operations in mjai file"""
    counts = {"chi": 0, "pon": 0, "daiminkan": 0, "kakan": 0, "ankan": 0, "reach": 0, "reach_accepted": 0}

    with open(mjai_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            etype = event.get("type", "")
            if etype in counts:
                counts[etype] += 1

    return counts


def count_tenhou6_ops(data):
    """Count operations by scanning ALL elements in the kyoku for string tokens"""
    counts = {"chi": 0, "pon": 0, "daiminkan": 0, "kakan": 0, "ankan": 0, "reach": 0}
    log = data.get("log", [])

    for kyoku in log:
        if not kyoku:
            continue
        # Search ALL elements for string tokens
        for elem in kyoku:
            if isinstance(elem, list):
                for item in elem:
                    if isinstance(item, str) and len(item) >= 2:
                        if item[0] == "c": counts["chi"] += 1
                        elif item[0] == "p": counts["pon"] += 1
                        elif item[0] == "m": counts["daiminkan"] += 1
                        elif item[0] == "k": counts["kakan"] += 1
                        elif item[0] == "e": counts["ankan"] += 1
                        elif item[0] == "r" and len(item) >= 2: counts["reach"] += 1
            elif isinstance(elem, str) and len(elem) >= 2:
                if elem[0] == "c": counts["chi"] += 1
                elif elem[0] == "p": counts["pon"] += 1
                elif elem[0] == "m": counts["daiminkan"] += 1
                elif elem[0] == "k": counts["kakan"] += 1
                elif elem[0] == "e": counts["ankan"] += 1
                elif elem[0] == "r": counts["reach"] += 1

    return counts


def main():
    import sys

    if len(sys.argv) < 2:
        # Default: process all datasets
        tenhou6_base = Path("dataset/tenhou6")
        mjai_base = Path("artifacts/converted_mjai")
        mode = "all"
    elif len(sys.argv) == 2:
        arg = Path(sys.argv[1])
        if arg.is_file():
            mode = "single"
            tenhou6_file = arg
            mjai_file = Path("artifacts/converted_mjai") / arg.with_suffix(".mjson").name
        else:
            mode = "dir"
            tenhou6_base = arg
            mjai_base = Path("artifacts/converted_mjai")
    else:
        tenhou6_base = Path(sys.argv[1])
        mjai_base = Path(sys.argv[2])
        mode = "all"

    if mode == "single":
        print("=== Tenhou6 Analysis ===")
        with open(tenhou6_file) as f:
            data = json.load(f)
        tenhou_counts = count_tenhou6_ops(data)
        print(f"  chi={tenhou_counts['chi']}, pon={tenhou_counts['pon']}, daiminkan={tenhou_counts['daiminkan']}, kakan={tenhou_counts['kakan']}, ankan={tenhou_counts['ankan']}, reach={tenhou_counts['reach']}")
        print("\n=== MJAI Analysis ===")
        if mjai_file.exists():
            mjai_counts = count_ops_in_mjai(mjai_file)
            print(f"  chi={mjai_counts['chi']}, pon={mjai_counts['pon']}, daiminkan={mjai_counts['daiminkan']}, kakan={mjai_counts['kakan']}, ankan={mjai_counts['ankan']}, reach={mjai_counts['reach']}")
        else:
            print(f"  MJAI file not found: {mjai_file}")
        return

    if mode == "dir":
        ds_dirs = [tenhou6_base]
    else:
        ds_dirs = sorted(tenhou6_base.glob("ds*"))

    print("Comparing tenhou6 operations with mjai operations...\n")

    total_tenhou = defaultdict(int)
    total_mjai = defaultdict(int)

    print(f"Found {len(ds_dirs)} dataset directories")

    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        mjai_ds_dir = mjai_base / ds_name

        if not mjai_ds_dir.exists():
            print(f"Skipping {ds_name} - no mjai output dir")
            continue

        tenhou_files = list(ds_dir.glob("*.json"))
        tenhou_ops = defaultdict(int)
        mjai_ops = defaultdict(int)

        for tf in tenhou_files:
            if tf.name == "browser_export_report.json":
                continue

            mjai_file = mjai_ds_dir / (tf.stem + ".mjson")

            # Count tenhou ops
            with open(tf) as f:
                data = json.load(f)
            t_ops = count_tenhou6_ops(data)
            for k, v in t_ops.items():
                tenhou_ops[k] += v

            # Count mjai ops
            if mjai_file.exists():
                m_ops = count_ops_in_mjai(mjai_file)
                for k, v in m_ops.items():
                    mjai_ops[k] += v

        print(f"{ds_name}:")
        print(f"  Tenhou6: chi={tenhou_ops['chi']}, pon={tenhou_ops['pon']}, daiminkan={tenhou_ops['daiminkan']}, kakan={tenhou_ops['kakan']}, ankan={tenhou_ops['ankan']}, reach={tenhou_ops['reach']}")
        print(f"  MJAI:    chi={mjai_ops['chi']}, pon={mjai_ops['pon']}, daiminkan={mjai_ops['daiminkan']}, kakan={mjai_ops['kakan']}, ankan={mjai_ops['ankan']}, reach={mjai_ops['reach']}")

        for k in tenhou_ops:
            total_tenhou[k] += tenhou_ops[k]
        for k in mjai_ops:
            total_mjai[k] += mjai_ops[k]

    print("\n=== TOTAL ===")
    print(f"  Tenhou6: chi={total_tenhou['chi']}, pon={total_tenhou['pon']}, daiminkan={total_tenhou['daiminkan']}, kakan={total_tenhou['kakan']}, ankan={total_tenhou['ankan']}, reach={total_tenhou['reach']}")
    print(f"  MJAI:    chi={total_mjai['chi']}, pon={total_mjai['pon']}, daiminkan={total_mjai['daiminkan']}, kakan={total_mjai['kakan']}, ankan={total_mjai['ankan']}, reach={total_mjai['reach']}")


if __name__ == "__main__":
    main()
