#!/usr/bin/env python3
"""预处理脚本：将所有 .mjson 文件解析为 .npz 缓存（keqingv1 特征格式）。

用法：
  uv run python scripts/preprocess_cached.py --data_dirs artifacts/converted_mjai/ds1 ... --output_dir processed
  uv run python scripts/preprocess_cached.py --config configs/keqing_default.yaml --output_dir processed
  # ds7 仅 NAGA 视角：
  uv run python scripts/preprocess_cached.py --data_dirs artifacts/converted_mjai/ds7 --output_dir processed_naga --actor_name_filter 'ⓝNAGA25'

输出结构：
  processed/ds1/0000_xxx.npz
  processed/ds2/0001_xxx.npz
  ...

每个 .npz 包含：
  tile_feat : float16, shape (N, 54, 34)
  scalar    : float16, shape (N, 48)
  mask      : uint8,   shape (N, 45)
  action_idx: int16,   shape (N,)
  value     : float16, shape (N,)
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

# 确保 src/ 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from keqingv1.action_space import action_to_idx, build_legal_mask
from keqingv1.features import encode

_SUIT_PERMS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ('m', 'p', 's')
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')


def _permute_mjson_text(text: str, perm: tuple) -> str:
    src_to_dst = {_SUITS[src]: _SUITS[dst] for dst, src in enumerate(perm)}

    def replace_pai(m: re.Match) -> str:
        num, suit = m.group(1), m.group(2)
        new_suit = src_to_dst[suit]
        full = m.group(0)
        if full.endswith('r"'):
            return f'"{num}{new_suit}r"'
        return f'"{num}{new_suit}"'

    return _PAI_RE.sub(replace_pai, text)


def _parse_events_to_arrays(events, actor_name_filter=None) -> Tuple[np.ndarray, ...]:
    """解析事件列表，返回 (tile_feat, scalar, mask, action_idx, value) 的 numpy 数组。"""
    rows_tile, rows_scalar, rows_mask, rows_action, rows_value = [], [], [], [], []
    try:
        samples = build_supervised_samples(events, actor_name_filter=actor_name_filter)
    except Exception:
        return None
    for s in samples:
        try:
            tile_feat, scalar = encode(s.state, s.actor)
            mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
            action_idx = action_to_idx(s.label_action)
            if mask[action_idx] == 0:
                continue
            value = float(np.clip(s.value_target, -1.0, 1.0))
            rows_tile.append(tile_feat)
            rows_scalar.append(scalar)
            rows_mask.append(mask)
            rows_action.append(action_idx)
            rows_value.append(value)
        except Exception:
            continue
    if not rows_tile:
        return None
    return (
        np.stack(rows_tile).astype(np.float16),
        np.stack(rows_scalar).astype(np.float16),
        np.stack(rows_mask).astype(np.uint8),
        np.array(rows_action, dtype=np.int16),
        np.array(rows_value, dtype=np.float16),
    )


def process_file(args: Tuple) -> Tuple[str, int]:
    """处理单个文件，输出 .npz。返回 (状态, 样本数)。"""
    src_path, out_path, augment, actor_name_filter = args
    if out_path.exists():
        return ('skip', 0)
    try:
        text = src_path.read_text(encoding='utf-8')
        events = read_mjai_jsonl(src_path)
    except Exception as e:
        return ('error', 0)

    all_tile, all_scalar, all_mask, all_action, all_value = [], [], [], [], []

    # 原始
    result = _parse_events_to_arrays(events, actor_name_filter=actor_name_filter)
    if result is not None:
        all_tile.append(result[0])
        all_scalar.append(result[1])
        all_mask.append(result[2])
        all_action.append(result[3])
        all_value.append(result[4])

    # 花色增强
    if augment:
        for perm in _SUIT_PERMS[1:]:
            permuted_text = _permute_mjson_text(text, perm)
            tmp = Path(tempfile.mktemp(suffix='.mjson'))
            try:
                tmp.write_text(permuted_text, encoding='utf-8')
                perm_events = read_mjai_jsonl(tmp)
                result = _parse_events_to_arrays(perm_events, actor_name_filter=actor_name_filter)
                if result is not None:
                    all_tile.append(result[0])
                    all_scalar.append(result[1])
                    all_mask.append(result[2])
                    all_action.append(result[3])
                    all_value.append(result[4])
            except Exception:
                pass
            finally:
                if tmp.exists():
                    tmp.unlink()

    if not all_tile:
        return ('empty', 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_feat = np.concatenate(all_tile, axis=0)
    scalar = np.concatenate(all_scalar, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    action_idx = np.concatenate(all_action, axis=0)
    value = np.concatenate(all_value, axis=0)

    np.savez_compressed(
        out_path,
        tile_feat=tile_feat,
        scalar=scalar,
        mask=mask,
        action_idx=action_idx,
        value=value,
    )
    return ('ok', len(tile_feat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_dirs', nargs='+', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='processed')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='花色增强 ×6（预处理阶段，推荐不开启，改用训练时随机增强）')
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--actor_name_filter', nargs='+', type=str, default=None,
                        help='只提取指定玩家名的样本，如 --actor_name_filter "ⓝNAGA25"')
    args = parser.parse_args()

    data_dirs: List[str] = []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        data_dirs = cfg.get('data_dirs', [])
    if args.data_dirs:
        data_dirs = args.data_dirs

    if not data_dirs:
        print('错误：请通过 --data_dirs 或 --config 指定数据目录')
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_name_filter = set(args.actor_name_filter) if args.actor_name_filter else None

    # 收集所有任务
    tasks = []
    for data_dir in data_dirs:
        src_dir = Path(data_dir)
        ds_name = src_dir.name  # e.g. 'ds1'
        out_ds_dir = output_dir / ds_name
        for mjson_file in sorted(src_dir.glob('*.mjson')):
            out_file = out_ds_dir / (mjson_file.stem + '.npz')
            tasks.append((mjson_file, out_file, args.augment, actor_name_filter))

    total = len(tasks)
    filter_info = f'，actor_filter={args.actor_name_filter}' if actor_name_filter else ''
    print(f'共 {total} 个文件，使用 {args.workers} 个进程，augment={args.augment}{filter_info}')

    done, skipped, errors, total_samples = 0, 0, 0, 0
    with Pool(processes=args.workers) as pool:
        for status, n_samples in pool.imap_unordered(process_file, tasks, chunksize=4):
            done += 1
            if status == 'ok':
                total_samples += n_samples
            elif status == 'skip':
                skipped += 1
            elif status in ('error', 'empty'):
                errors += 1
            if done % 200 == 0 or done == total:
                print(f'  [{done}/{total}] 跳过={skipped} 错误={errors} 样本={total_samples:,}')

    print(f'\n完成！总样本数: {total_samples:,}，跳过: {skipped}，错误: {errors}')


if __name__ == '__main__':
    main()
