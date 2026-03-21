#!/usr/bin/env python3
"""
批量转换 dataset/tenhou6/ds* 目录下的json文件到 artifacts/converted/{ds*}

用法:
    python scripts/batch_convert_ds.py
    python scripts/batch_convert_ds.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 添加src目录到path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl


def convert_dir(input_dir: Path, libriichi_bin: str | None, skip_existing: bool) -> dict:
    """转换单个ds目录，保持目录结构"""
    output_dir = Path("artifacts/converted") / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = []
    errors = []
    skipped = []
    
    json_files = sorted(input_dir.glob("*.json"))
    
    for src in json_files:
        out = output_dir / f"{src.stem}.jsonl"
        
        if skip_existing and out.exists():
            skipped.append(src.name)
            continue
        
        try:
            convert_raw_to_mjai(str(src), str(out), libriichi_bin)
            errs = validate_mjai_jsonl(out)
            if errs:
                errors.append({"file": src.name, "errors": errs})
                print(f"  错误: {src.name} - {errs}")
            else:
                converted.append(src.name)
                print(f"  转换: {src.name}")
        except Exception as e:
            errors.append({"file": src.name, "error": str(e)})
            print(f"  异常: {src.name} - {e}")
    
    return {"converted": converted, "errors": errors, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量转换ds目录到mjai jsonl格式")
    parser.add_argument("--input-base", default="dataset/tenhou6", help="输入目录根路径")
    parser.add_argument("--libriichi-bin", default=None, help="libriichi可执行文件路径")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的文件")
    parser.add_argument("--ds-list", type=str, default=None, help="指定要处理的ds目录，如 ds1,ds2,ds3")
    args = parser.parse_args()
    
    input_base = Path(args.input_base)
    
    # 获取要处理的ds目录
    if args.ds_list:
        ds_names = args.ds_list.split(",")
        ds_dirs = [input_base / name for name in ds_names]
    else:
        ds_dirs = sorted(input_base.glob("ds*"), key=lambda p: int(p.name[2:]) if p.name[2:].isdigit() else 0)
    
    print(f"输入目录: {input_base}")
    print(f"输出目录: artifacts/converted/{{ds_name}}")
    print(f"处理目录: {[d.name for d in ds_dirs]}")
    print("-" * 50)
    
    total_converted = 0
    total_errors = 0
    total_skipped = 0
    
    for ds_dir in ds_dirs:
        if not ds_dir.exists():
            print(f"\n目录不存在: {ds_dir}")
            continue
        
        file_count = len(list(ds_dir.glob("*.json")))
        print(f"\n处理 {ds_dir.name} ({file_count} 个json文件)...")
        
        result = convert_dir(ds_dir, args.libriichi_bin, args.skip_existing)
        total_converted += len(result["converted"])
        total_errors += len(result["errors"])
        total_skipped += len(result["skipped"])
    
    print("\n" + "=" * 50)
    print(f"转换完成!")
    print(f"成功: {total_converted}")
    print(f"跳过: {total_skipped}")
    print(f"错误: {total_errors}")


if __name__ == "__main__":
    main()
