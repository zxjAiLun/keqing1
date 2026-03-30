"""流式数据集：从 .mjson 文件逐局解析，value target 使用 build_supervised_samples 提供的 local EV proxy。

设计：
- IterableDataset，不预加载全量数据
- 维护 shuffle_buffer（大小 2000），随机 pop 样本减少顺序偏差
- value target = ReplaySample.value_target（向听数/进张数变化的即时奖励代理），clip 到 [-1, 1]
- 多 epoch 自动重 shuffle 文件列表
"""

from __future__ import annotations

import json
import random
import re
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from keqingv1.action_space import action_to_idx, build_legal_mask, ACTION_SPACE
from keqingv1.features import encode

# value target 归一化分母（日麻点数范围约 ±30000）
_VALUE_NORM = 30000.0

# shuffle buffer 大小
_BUFFER_SIZE = 2000

# m/p/s 三色全排列（slot 0=万, 1=饼, 2=索）
_SUIT_PERMS = [
    (0, 1, 2),  # 原始
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ('m', 'p', 's')

# 匹配 JSON 字符串值中的牌名："Nm" / "Np" / "Ns" / "5mr" / "5pr" / "5sr"
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')


def _permute_mjson_text(text: str, perm: tuple) -> str:
    """对 mjson 文本中所有牌名做花色置换，不影响 JSON 键名和非牌字段。"""
    dst = _SUITS
    src_to_dst = {_SUITS[src_slot]: dst[dst_slot] for dst_slot, src_slot in enumerate(perm)}

    def replace_pai(m: re.Match) -> str:
        num, suit = m.group(1), m.group(2)
        new_suit = src_to_dst[suit]
        full = m.group(0)  # 包含引号，如 "5mr"
        # 赤宝牌：原字符串含 r
        if full.endswith('r"'):
            return f'"{num}{new_suit}r"'
        return f'"{num}{new_suit}"'

    return _PAI_RE.sub(replace_pai, text)


def _parse_events(events: List[dict], actor_name: Optional[str] = None) -> Iterator[Tuple]:
    """从已解析事件列表生成训练样本。"""
    actor_name_filter = {actor_name} if actor_name else None
    try:
        samples = build_supervised_samples(events, actor_name_filter=actor_name_filter)
    except Exception:
        return
    for s in samples:
        actor = s.actor
        value_target = float(np.clip(s.value_target, -1.0, 1.0))
        tile_feat, scalar = encode(s.state, actor)
        mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
        action_idx = action_to_idx(s.label_action)
        if mask[action_idx] == 0:
            continue
        yield tile_feat, scalar, mask, action_idx, np.float32(value_target)


def _extract_final_deltas(events: List[dict]) -> List[int]:
    """累加一局所有 hora/ryukyoku 的 deltas，返回 4 个玩家的总分变化。"""
    totals = [0, 0, 0, 0]
    for e in events:
        if e.get("type") in ("hora", "ryukyoku"):
            deltas = e.get("deltas", [])
            for pid in range(min(4, len(deltas))):
                totals[pid] += deltas[pid]
    return totals


def _parse_file(
    path: Path,
    augment: bool = False,
    actor_name: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """解析单个 .mjson 文件，yield (tile_feat, scalar, legal_mask, action_idx, value)。
    augment=True 时额外生成 5 种 m/p/s 花色置换版本。
    actor_name 非 None 时只收集该玩家名字的样本。
    """
    try:
        text = path.read_text(encoding='utf-8')
        events = read_mjai_jsonl(path)
    except Exception:
        return

    yield from _parse_events(events, actor_name=actor_name)

    if augment:
        for perm in _SUIT_PERMS[1:]:
            permuted_text = _permute_mjson_text(text, perm)
            tmp = Path(tempfile.mktemp(suffix='.mjson'))
            try:
                tmp.write_text(permuted_text, encoding='utf-8')
                perm_events = read_mjai_jsonl(tmp)
            except Exception:
                continue
            finally:
                if tmp.exists():
                    tmp.unlink()
            yield from _parse_events(perm_events, actor_name=actor_name)


class MjaiIterableDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = _BUFFER_SIZE,
        seed: Optional[int] = None,
        augment: bool = False,
        actor_name: Optional[str] = None,
    ):
        self.file_paths = list(file_paths)
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        self.augment = augment
        self.actor_name = actor_name

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.file_paths)
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(paths)

        buffer: List[Tuple] = []

        for path in paths:
            for sample in _parse_file(path, augment=self.augment, actor_name=self.actor_name):
                buffer.append(sample)
                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

        # 清空剩余 buffer
        if self.shuffle:
            rng.shuffle(buffer)
        for sample in buffer:
            yield sample

    @staticmethod
    def collate(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        tile_feats, scalars, masks, action_idxs, values = zip(*batch)
        return (
            torch.from_numpy(np.stack(tile_feats)),
            torch.from_numpy(np.stack(scalars)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(action_idxs, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
        )


def split_files(
    root_dirs: List[Path],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """收集所有 .mjson 文件并按比例分为训练/验证集。"""
    all_files: List[Path] = []
    for d in root_dirs:
        all_files.extend(sorted(d.glob("*.mjson")))

    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    return all_files[n_val:], all_files[:n_val]
