"""tenhou6 JSON → mjai JSONL 转换工具（调用 convlog 二进制）。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CONVLOG_BIN = (
    Path(__file__).parent.parent.parent
    / "third_party"
    / "mjai-reviewer"
    / "target"
    / "release"
    / "convlog"
)


def tenhou6_to_mjson(t6_json: dict, output_path: Path) -> bool:
    """Write tenhou6 JSON to a temp file then run convlog. Returns success."""
    tmp = output_path.with_suffix(".tmp.json")
    try:
        tmp.write_text(json.dumps(t6_json, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [str(CONVLOG_BIN), str(tmp), str(output_path)],
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
