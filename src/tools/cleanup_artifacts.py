from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set


def _collect_keep_dirs(root: Path) -> Set[Path]:
    """
    Aggressive-but-safe cleanup:
    - Keep directories that contain `best.*`, `last.*`, `metrics.json`
    - Always keep `converted/` and `mjai_submission/`
    """
    keep: Set[Path] = set()

    # Always keep conversion inputs/outputs.
    for p in (root / "converted", root / "mjai_submission"):
        if p.exists():
            keep.add(p)

    patterns = ["best.*", "last.*", "metrics.json"]
    matched_files: List[Path] = []
    for pat in patterns:
        matched_files.extend(list(root.glob(f"**/{pat}")))

    # Keep the directory containing each key file.
    for f in matched_files:
        keep.add(f.parent)

    # Keep any directory that is an ancestor of a key directory.
    # (Avoid accidentally deleting a parent that is needed by kept children.)
    all_keep: Set[Path] = set()
    for kd in keep:
        cur = kd
        while cur != root and cur not in all_keep:
            all_keep.add(cur)
            cur = cur.parent
    all_keep.add(root)

    return {d for d in keep if d.is_dir()} | all_keep


def _plan_deletions(root: Path, keep_dirs: Set[Path]) -> List[Path]:
    delete_dirs: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # Keep if any keep_dir is inside this directory (or equal).
        if any(kd == child or kd.is_relative_to(child) for kd in keep_dirs):
            continue
        delete_dirs.append(child)
    # Sort by depth desc (remove deeper subtrees first if we ever add per-dir deletion later)
    delete_dirs.sort(key=lambda p: len(p.parts), reverse=True)
    return delete_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts", help="Artifacts root directory")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Only print what to delete")
    parser.add_argument("--apply", action="store_true", help="Actually delete (still requires confirmation)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--print-limit", type=int, default=500, help="Max printed deletion entries")
    parser.add_argument("--output-plan", default=None, help="Optional path to write JSON plan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise RuntimeError(f"root not found: {root}")

    keep_dirs = _collect_keep_dirs(root)
    delete_dirs = _plan_deletions(root, keep_dirs)

    plan: Dict[str, object] = {
        "root": str(root),
        "dry_run": bool(args.dry_run and not args.apply),
        "apply": bool(args.apply),
        "keep_dir_count": len(keep_dirs),
        "delete_dir_count": len(delete_dirs),
        "delete_dirs": [str(p) for p in delete_dirs[: args.print_limit]],
        "delete_dirs_total": len(delete_dirs),
    }

    if args.output_plan:
        Path(args.output_plan).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_plan).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print preview.
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.apply:
        return

    if args.yes:
        ok = True
    else:
        confirm = input("Type 'YES' to delete the above directories: ").strip()
        ok = confirm == "YES"

    if not ok:
        print("Aborted.")
        return

    # Delete.
    for p in delete_dirs:
        print(f"Deleting: {p}")
        shutil.rmtree(p, ignore_errors=False)

    print("Cleanup finished.")


if __name__ == "__main__":
    main()

