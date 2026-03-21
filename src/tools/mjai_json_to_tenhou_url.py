#!/usr/bin/env python3
"""
将 mjai 自战导出 JSON 转换为天凤牌谱链接

用法:
    python -m tools.mjai_json_to_tenhou_url \
        --input dataset/bot_export/mjai_selfplay_v2_ds3_latest_001.json
"""

import argparse
import json
import urllib.parse
from pathlib import Path


def convert_mjai_to_tenhou_url(mjai_data: dict) -> list[str]:
    result = {
        "name": mjai_data.get("name", []),
        "rule": {
            "disp": mjai_data.get("rule", {}).get("disp", ""),
            "aka": mjai_data.get("rule", {}).get("aka", 1),
        },
        "log": mjai_data.get("log", []),
    }

    json_str = json.dumps(result, separators=(",", ":"))
    encoded = urllib.parse.quote(json_str, safe="")
    url = f"https://tenhou.net/5/#json={encoded}"

    return [url]


def main() -> None:
    parser = argparse.ArgumentParser(description="将 mjai JSON 转换为天凤链接")
    parser.add_argument(
        "--input",
        required=True,
        help="mjai 自战导出 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        help="输出文件路径（可选，默认输出到 stdout）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mjai_data = json.loads(input_path.read_text(encoding="utf-8"))

    urls = convert_mjai_to_tenhou_url(mjai_data)

    output = "\n".join(urls)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"已保存 {len(urls)} 个链接到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
