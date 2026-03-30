#!/usr/bin/env python3
"""预处理脚本 v2：在 v1 基础上额外保存 snap_json（仅 meld/none 样本）。

用法：
  uv run python scripts/preprocess_v2.py --data_dirs artifacts/converted_mjai/ds1 ... --output_dir processed_v2
  uv run python scripts/preprocess_v2.py --data_dirs artifacts/converted_mjai/ds7 --output_dir processed_v2_naga --actor_name_filter 'ⓝNAGA25'

每个 .npz 包含（比 v1 多一个字段）：
  tile_feat : float16, shape (N, 54, 34)
  scalar    : float16, shape (N, 48)
  mask      : uint8,   shape (N, 45)
  action_idx: int16,   shape (N,)
  value     : float16, shape (N,)
  snap_json : object,  shape (N,)  ← 新增：meld/none 样本存 JSON 字符串，其余存空字符串

snap_json 格式（meld/none 样本）：
  {
    "hand": [...],                  # 副露前手牌
    "melds": [[...], ...],           # 副露前各家副露
    "discards": [[...], ...],
    "dora_markers": [...],
    "reached": [...],
    "scores": [...],
    "bakaze": "E",
    "kyoku": 1,
    "honba": 0,
    "kyotaku": 0,
    "jikaze": 0,
    "actor": 0,
    "label_action": {...},           # GT 动作（含 type/pai/consumed）
    "meld_candidate": {...} | null   # none 样本：法律副露动作中的第一个候选（用于 ranking loss）
  }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from keqingv1.action_space import action_to_idx, build_legal_mask
from keqingv1.action_space import (
    CHI_LOW_IDX, CHI_MID_IDX, CHI_HIGH_IDX,
    PON_IDX, DAIMINKAN_IDX, ANKAN_IDX, KAKAN_IDX, NONE_IDX,
)
from keqingv1.features import encode

_MELD_ACTION_IDXS = {CHI_LOW_IDX, CHI_MID_IDX, CHI_HIGH_IDX, PON_IDX, DAIMINKAN_IDX, ANKAN_IDX, KAKAN_IDX}
_MELD_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}

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


def _permute_snap_json(snap_json_str: str, perm: tuple) -> str:
    """对 snap_json 字符串里的牌名做花色置换（与 tile_feat 增强保持一致）。"""
    if not snap_json_str:
        return snap_json_str
    return _permute_mjson_text(snap_json_str, perm)


def _snap_min(state: dict, actor: int, label_action: dict, legal_dicts: list) -> str:
    """构造精简 snap dict 并序列化为 JSON 字符串。"""
    # none 样本：从 legal_actions 里找第一个副露候选
    meld_candidate: Optional[dict] = None
    action_idx_val = action_to_idx(label_action)
    if action_idx_val == NONE_IDX:
        # 按优先级选最强副露候选：daiminkan > pon > chi（收益递减）
        _MELD_PRIORITY = ["daiminkan", "ankan", "kakan", "pon", "chi"]
        candidates = {a.get("type"): a for a in legal_dicts if a.get("type") in _MELD_TYPES}
        for mtype in _MELD_PRIORITY:
            if mtype in candidates:
                meld_candidate = candidates[mtype]
                break

    snap = {
        "hand": state.get("hand", []),
        "melds": state.get("melds", [[], [], [], []]),
        "discards": state.get("discards", [[], [], [], []]),
        "dora_markers": state.get("dora_markers", []),
        "reached": state.get("reached", [False, False, False, False]),
        "scores": state.get("scores", [25000, 25000, 25000, 25000]),
        "bakaze": state.get("bakaze", "E"),
        "kyoku": state.get("kyoku", 1),
        "honba": state.get("honba", 0),
        "kyotaku": state.get("kyotaku", 0),
        "jikaze": state.get("jikaze", 0),
        "oya": state.get("oya", 0),
        "actor": actor,
        "label_action": label_action,
        "meld_candidate": meld_candidate,
    }
    return json.dumps(snap, ensure_ascii=False)


def _parse_events_to_arrays(events, actor_name_filter=None) -> Optional[Tuple]:
    """解析事件列表，返回 (tile_feat, scalar, mask, action_idx, value, snap_json) 的 numpy 数组。"""
    rows_tile, rows_scalar, rows_mask, rows_action, rows_value, rows_snap = [], [], [], [], [], []
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

            # snap_json：仅 meld/none 样本存完整 snap，其余存空字符串
            if action_idx in _MELD_ACTION_IDXS or action_idx == NONE_IDX:
                snap_json_str = _snap_min(s.state, s.actor, s.label_action, s.legal_actions)
            else:
                snap_json_str = ""

            rows_tile.append(tile_feat)
            rows_scalar.append(scalar)
            rows_mask.append(mask)
            rows_action.append(action_idx)
            rows_value.append(value)
            rows_snap.append(snap_json_str)
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
        np.array(rows_snap, dtype=object),
    )


def process_file(args: Tuple) -> Tuple[str, int]:
    """处理单个文件，输出 .npz。返回 (状态, 样本数)。"""
    src_path, out_path, augment, actor_name_filter = args
    if out_path.exists():
        return ('skip', 0)
    try:
        text = src_path.read_text(encoding='utf-8')
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception:
        return ('error', 0)

    all_tile, all_scalar, all_mask, all_action, all_value, all_snap = [], [], [], [], [], []

    result = _parse_events_to_arrays(events, actor_name_filter=actor_name_filter)
    if result is not None:
        all_tile.append(result[0])
        all_scalar.append(result[1])
        all_mask.append(result[2])
        all_action.append(result[3])
        all_value.append(result[4])
        all_snap.append(result[5])

    if augment:
        for perm in _SUIT_PERMS[1:]:
            permuted_text = _permute_mjson_text(text, perm)
            try:
                perm_events = [json.loads(line) for line in permuted_text.splitlines() if line.strip()]
                result = _parse_events_to_arrays(perm_events, actor_name_filter=actor_name_filter)
                if result is not None:
                    all_tile.append(result[0])
                    all_scalar.append(result[1])
                    all_mask.append(result[2])
                    all_action.append(result[3])
                    all_value.append(result[4])
                    # snap_json 里的牌名也做同样的花色置换
                    perm_snap = np.array(
                        [_permute_snap_json(s, perm) for s in result[5]],
                        dtype=object,
                    )
                    all_snap.append(perm_snap)
            except Exception:
                pass

    if not all_tile:
        return ('empty', 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_feat = np.concatenate(all_tile, axis=0)
    scalar = np.concatenate(all_scalar, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    action_idx = np.concatenate(all_action, axis=0)
    value = np.concatenate(all_value, axis=0)
    snap_json = np.concatenate(all_snap, axis=0)

    np.savez_compressed(
        out_path,
        tile_feat=tile_feat,
        scalar=scalar,
        mask=mask,
        action_idx=action_idx,
        value=value,
        snap_json=snap_json,
    )
    return ('ok', len(tile_feat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_dirs', nargs='+', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='processed_v2')
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--actor_name_filter', nargs='+', type=str, default=None)
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

    tasks = []
    for data_dir in data_dirs:
        src_dir = Path(data_dir)
        ds_name = src_dir.name
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
