#!/usr/bin/env python3
"""Export Tenhou6 review cases for outcome-risk slices.

The selector is intentionally outcome-driven. It exports cases where the target
model's own game result was poor in broad risk contexts; it does not use a
reference model's chosen action as a label.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6, load_mjai_jsonl


DEFAULT_TITLE_INFO = ["玉の間四人南", "2026/6/13 01:44:27"]
DEFAULT_RATING_INFO = "[125,60,-5,-240],[125,60,-5,-195],[125,60,-5,-195],[125,60,-5,-180],1"


@dataclass
class TargetRound:
    source_log: Path
    kyoku_index: int
    names: list[str]
    seat: int
    start_scores: list[int]
    dealer: bool
    turns: int = 0
    fuuro: bool = False
    riichi: bool = False
    saw_discard_vs_riichi: bool = False
    saw_late_discard_vs_riichi: bool = False
    saw_fuuro_discard_vs_riichi: bool = False
    first_riichi_turn: int | None = None
    first_fuuro_turn: int | None = None
    agari: bool = False
    houjuu: bool = False
    delta_score: float = 0.0
    tags: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--target-model", default="T1_71000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-bucket", type=int, default=12)
    parser.add_argument("--title-disp", default=DEFAULT_TITLE_INFO[0])
    parser.add_argument("--title-date", default=DEFAULT_TITLE_INFO[1])
    parser.add_argument("--rating-info", default=DEFAULT_RATING_INFO)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_key(path: Path) -> tuple[int, str]:
    try:
        seed = int(path.name.split("_", 1)[0])
    except ValueError:
        seed = 0
    return seed, path.name


def parse_deltas(event: dict[str, Any]) -> list[float]:
    raw = event.get("deltas", [0, 0, 0, 0])
    if not isinstance(raw, list) or len(raw) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(raw[seat] or 0.0) for seat in range(4)]


def iter_target_rounds(log_path: Path, *, target_model: str) -> list[TargetRound]:
    events = load_mjai_jsonl(log_path)
    names = [str(seat) for seat in range(4)]
    current: TargetRound | None = None
    rounds: list[TargetRound] = []
    kyoku_index = -1
    riichi = [False, False, False, False]
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            raw_names = event.get("names")
            if isinstance(raw_names, list) and len(raw_names) >= 4:
                names = [str(name) for name in raw_names[:4]]
        elif event_type == "start_kyoku":
            if current is not None:
                rounds.append(current)
            kyoku_index += 1
            riichi = [False, False, False, False]
            if target_model not in names:
                current = None
                continue
            seat = names.index(target_model)
            scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
            current = TargetRound(
                source_log=log_path,
                kyoku_index=kyoku_index,
                names=list(names),
                seat=seat,
                start_scores=scores,
                dealer=seat == int(event.get("oya", -1)),
            )
        elif current is None:
            continue
        elif event_type == "tsumo":
            actor = event.get("actor")
            if actor == current.seat:
                current.turns += 1
        elif event_type == "reach":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                riichi[actor] = True
                if actor == current.seat and not current.riichi:
                    current.riichi = True
                    current.first_riichi_turn = max(1, current.turns)
        elif event_type in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            actor = event.get("actor")
            if actor == current.seat and not current.fuuro:
                current.fuuro = True
                current.first_fuuro_turn = max(1, current.turns)
        elif event_type == "dahai":
            actor = event.get("actor")
            if actor == current.seat:
                turn = max(1, current.turns)
                opponents_riichi = any(value for idx, value in enumerate(riichi) if idx != current.seat)
                if opponents_riichi:
                    current.saw_discard_vs_riichi = True
                    if turn >= 13:
                        current.saw_late_discard_vs_riichi = True
                    if current.fuuro:
                        current.saw_fuuro_discard_vs_riichi = True
        elif event_type == "hora":
            deltas = parse_deltas(event)
            current.delta_score += float(deltas[current.seat])
            actor = event.get("actor")
            target = event.get("target")
            if actor == current.seat:
                current.agari = True
            if target == current.seat and target != actor:
                current.houjuu = True
        elif event_type == "ryukyoku":
            deltas = parse_deltas(event)
            current.delta_score += float(deltas[current.seat])
        elif event_type == "end_kyoku":
            rounds.append(current)
            current = None
    if current is not None:
        rounds.append(current)
    return rounds


def classify_round(round_: TargetRound) -> list[str]:
    tags: list[str] = []
    if round_.houjuu:
        tags.append("houjuu_any")
    if round_.houjuu and round_.saw_discard_vs_riichi:
        tags.append("houjuu_vs_riichi")
    if round_.houjuu and round_.saw_late_discard_vs_riichi:
        tags.append("houjuu_vs_riichi_late_13_plus")
    if round_.houjuu and round_.saw_fuuro_discard_vs_riichi:
        tags.append("houjuu_after_fuuro_vs_riichi")
    if round_.riichi and not round_.agari and round_.delta_score < 0:
        tags.append("riichi_negative_no_agari")
    if round_.riichi and round_.first_riichi_turn is not None and round_.first_riichi_turn <= 6 and round_.delta_score < 0:
        tags.append("early_riichi_negative")
    if round_.fuuro and round_.houjuu:
        tags.append("after_fuuro_houjuu")
    if round_.dealer and round_.houjuu:
        tags.append("dealer_houjuu")
    return tags


def round_priority(round_: TargetRound) -> tuple[int, float]:
    score = 0
    if round_.saw_fuuro_discard_vs_riichi:
        score += 6
    if round_.saw_late_discard_vs_riichi:
        score += 5
    if round_.saw_discard_vs_riichi:
        score += 4
    if round_.houjuu:
        score += 4
    if round_.dealer:
        score += 1
    if round_.riichi and not round_.agari:
        score += 1
    return score, -round_.delta_score


def select_cases(log_dir: Path, *, target_model: str, limit_per_bucket: int) -> list[TargetRound]:
    buckets: dict[str, list[TargetRound]] = defaultdict(list)
    for source in sorted(log_dir.glob("*.json.gz"), key=source_key):
        for round_ in iter_target_rounds(source, target_model=target_model):
            tags = classify_round(round_)
            round_.tags.update(tags)
            for tag in tags:
                buckets[tag].append(round_)
    selected: dict[tuple[str, int], TargetRound] = {}
    preferred = [
        "houjuu_after_fuuro_vs_riichi",
        "houjuu_vs_riichi_late_13_plus",
        "riichi_negative_no_agari",
        "early_riichi_negative",
        "after_fuuro_houjuu",
        "dealer_houjuu",
    ]
    for bucket in preferred:
        cases = sorted(buckets.get(bucket, []), key=round_priority, reverse=True)[:limit_per_bucket]
        for case in cases:
            selected[(str(case.source_log), case.kyoku_index)] = case
    return sorted(selected.values(), key=lambda case: (source_key(case.source_log), case.kyoku_index))

def naga_payload(source: dict[str, Any], kyoku_log: list[Any], *, title_disp: str, title_date: str, rating_info: str) -> dict[str, Any]:
    return {
        "title": [[title_disp, title_date], rating_info],
        "name": list(source.get("name", ["A", "B", "C", "D"]))[:4],
        "rule": {"disp": title_disp, "aka53": 1, "aka52": 1, "aka51": 1},
        "log": [kyoku_log],
    }


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_zip(zip_path: Path, files: list[Path], *, root: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))


def export_cases(args: argparse.Namespace, cases: list[TargetRound]) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output dir already exists, use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hanchan_dir = output_dir / "hanchan_tenhou6"
    naga_dir = output_dir / "naga_kyoku_tenhou6"
    hanchan_dir.mkdir(parents=True, exist_ok=True)
    naga_dir.mkdir(parents=True, exist_ok=True)

    tenhou_cache: dict[Path, tuple[Path, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    blocks: list[str] = []
    url_lines: list[str] = []
    for index, case in enumerate(cases):
        if case.source_log not in tenhou_cache:
            events = load_mjai_jsonl(case.source_log)
            tenhou6 = convert_mjai_jsonl_to_tenhou6(events)
            hanchan_path = hanchan_dir / f"{case.source_log.stem.removesuffix('.json')}.tenhou6.json"
            hanchan_path.write_text(compact_json(tenhou6) + "\n", encoding="utf-8")
            tenhou_cache[case.source_log] = (hanchan_path, tenhou6)
        hanchan_path, tenhou6 = tenhou_cache[case.source_log]
        logs = list(tenhou6.get("log") or [])
        if not (0 <= case.kyoku_index < len(logs)):
            continue
        payload = naga_payload(
            tenhou6,
            logs[case.kyoku_index],
            title_disp=str(args.title_disp),
            title_date=str(args.title_date),
            rating_info=str(args.rating_info),
        )
        case_stem = f"case_{index:03d}_{case.source_log.stem.removesuffix('.json')}_kyoku_{case.kyoku_index:02d}"
        naga_path = naga_dir / f"{case_stem}.naga.tenhou6.json"
        naga_json = compact_json(payload)
        naga_path.write_text(naga_json + "\n", encoding="utf-8")
        url = "https://tenhou.net/6/#json=" + naga_json
        url_lines.append(url)
        blocks.append(
            "\n".join(
                [
                    f"===== CASE {index:03d} | kyoku={case.kyoku_index} | seat={case.seat} | tags={','.join(sorted(case.tags))} =====",
                    naga_json,
                    "",
                ]
            )
        )
        rows.append(
            {
                "case_index": index,
                "source_log": str(case.source_log),
                "hanchan_tenhou6_path": str(hanchan_path),
                "naga_kyoku_tenhou6_path": str(naga_path),
                "kyoku_index": case.kyoku_index,
                "target_model": str(args.target_model),
                "target_seat": case.seat,
                "player_names": case.names,
                "tags": sorted(case.tags),
                "start_scores": case.start_scores,
                "dealer": case.dealer,
                "turns": case.turns,
                "first_riichi_turn": case.first_riichi_turn,
                "first_fuuro_turn": case.first_fuuro_turn,
                "agari": case.agari,
                "houjuu": case.houjuu,
                "delta_score": case.delta_score,
                "url": url,
            }
        )

    manifest = output_dir / "case_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_dir / "naga_kyoku_blocks.txt").write_text("\n".join(blocks), encoding="utf-8")
    (output_dir / "naga_kyoku_urls.txt").write_text("\n".join(url_lines) + "\n", encoding="utf-8")
    write_zip(output_dir / "hanchan_tenhou6_cases.zip", sorted(hanchan_dir.glob("*.json")), root=output_dir)
    write_zip(output_dir / "naga_kyoku_tenhou6_cases.zip", sorted(naga_dir.glob("*.json")), root=output_dir)


def main() -> None:
    args = parse_args()
    cases = select_cases(args.log_dir, target_model=str(args.target_model), limit_per_bucket=int(args.limit_per_bucket))
    if not cases:
        raise SystemExit("no cases selected")
    export_cases(args, cases)
    print(f"selected cases: {len(cases)}", flush=True)
    print(f"saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
