"""流式数据集：从 .mjson 文件逐局解析，value target 使用最终相对得分。

设计：
- IterableDataset，不预加载全量数据
- 维护 shuffle_buffer（大小 2000），随机 pop 样本减少顺序偏差
- value target = 该局 actor 最终累计 delta / 30000（归一化到 [-1, 1]）
- 多 epoch 自动重 shuffle 文件列表
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from v5model.action_space import action_to_idx, build_legal_mask, ACTION_SPACE
from v5model.features import encode

# value target 归一化分母（日麻点数范围约 ±30000）
_VALUE_NORM = 30000.0

# shuffle buffer 大小
_BUFFER_SIZE = 2000


def _extract_final_deltas(events: List[dict]) -> List[int]:
    """累加一局所有 hora/ryukyoku 的 deltas，返回 4 个玩家的总分变化。"""
    totals = [0, 0, 0, 0]
    for e in events:
        if e.get("type") in ("hora", "ryukyoku"):
            deltas = e.get("deltas", [])
            for pid in range(min(4, len(deltas))):
                totals[pid] += deltas[pid]
    return totals


def _parse_file(path: Path) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """解析单个 .mjson 文件，yield (tile_feat, scalar, legal_mask, action_idx, value)。"""
    try:
        events = read_mjai_jsonl(path)
    except Exception:
        return

    final_deltas = _extract_final_deltas(events)

    try:
        samples = build_supervised_samples(events)
    except Exception:
        return

    for s in samples:
        actor = s.actor
        value_target = final_deltas[actor] / _VALUE_NORM
        value_target = float(np.clip(value_target, -1.0, 1.0))

        tile_feat, scalar = encode(s.state, actor)
        mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
        action_idx = action_to_idx(s.label_action)

        yield tile_feat, scalar, mask, action_idx, np.float32(value_target)


class MjaiIterableDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = _BUFFER_SIZE,
        seed: Optional[int] = None,
    ):
        self.file_paths = list(file_paths)
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator:
        rng = random.Random(self.seed)
        paths = list(self.file_paths)
        if self.shuffle:
            rng.shuffle(paths)

        buffer: List[Tuple] = []

        for path in paths:
            for sample in _parse_file(path):
                buffer.append(sample)
                if len(buffer) >= self.buffer_size:
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
