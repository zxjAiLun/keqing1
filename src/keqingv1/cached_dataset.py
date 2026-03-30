"""缓存数据集：直接加载预处理好的 .npz 文件，跳过实时解析。

设计：
- IterableDataset，流式加载 .npz 文件
- 维护 shuffle_buffer（大小 2000），随机 pop 样本
- 多 epoch 自动重 shuffle 文件列表
- .npz 格式：tile_feat(float16,(N,54,34)), scalar(float16,(N,48)), mask(uint8,(N,45)), action_idx(int16,(N,)), value(float16,(N,))
- aug_perms: 每个文件随机选 N 种花色置换实时做（特征层面 swap，不重新解析）
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

_BUFFER_SIZE = 2000

# tile34 花色区间：万(0-8), 饼(9-17), 索(18-26), 字(27-33)
_SUIT_SLICES = [slice(0, 9), slice(9, 18), slice(18, 27)]

# 全部 5 种非原始置换
_SUIT_PERMS = [
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def _apply_suit_perm(tile_feat: np.ndarray, mask: np.ndarray, action_idx: int, perm: tuple) -> Tuple:
    """在特征层面做花色置换。tile_feat: (54,34), mask: (45,), action_idx: int。"""
    src = [_SUIT_SLICES[i] for i in perm]
    new_tile = tile_feat.copy()
    new_tile[:, _SUIT_SLICES[0]] = tile_feat[:, src[0]]
    new_tile[:, _SUIT_SLICES[1]] = tile_feat[:, src[1]]
    new_tile[:, _SUIT_SLICES[2]] = tile_feat[:, src[2]]

    # action_idx: dahai(0-26) 数牌需置换，其他不变
    new_action = action_idx
    if action_idx < 27:
        suit_i = action_idx // 9
        rank = action_idx % 9
        new_suit_i = perm.index(suit_i)
        new_action = new_suit_i * 9 + rank

    # mask: dahai 数牌部分(0-26) 同样置换
    new_mask = mask.copy()
    new_mask[_SUIT_SLICES[0]] = mask[src[0]]
    new_mask[_SUIT_SLICES[1]] = mask[src[1]]
    new_mask[_SUIT_SLICES[2]] = mask[src[2]]

    return new_tile, new_mask, new_action


class CachedMjaiDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = _BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
        """aug_perms: 每个文件额外随机选几种花色置换（0=不增强，1-5=随机选N种）。"""
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.aug_perms = aug_perms

    def __iter__(self) -> Iterator[Tuple]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.file_paths
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(paths)

        buffer: List[Tuple] = []

        for path in paths:
            try:
                data = np.load(path)
                tile_feats = data['tile_feat'].astype(np.float32)
                scalars = data['scalar'].astype(np.float32)
                masks = data['mask'].astype(np.float32)
                action_idxs = data['action_idx'].astype(np.int64)
                values = data['value'].astype(np.float32)
            except Exception:
                continue

            perms_to_use = []
            if self.aug_perms > 0:
                perms_to_use = rng.sample(_SUIT_PERMS, min(self.aug_perms, len(_SUIT_PERMS)))

            n = len(tile_feats)
            for i in range(n):
                sample = (tile_feats[i], scalars[i], masks[i], int(action_idxs[i]), values[i])
                buffer.append(sample)
                for perm in perms_to_use:
                    tf, mk, ai = _apply_suit_perm(tile_feats[i], masks[i], int(action_idxs[i]), perm)
                    buffer.append((tf, scalars[i], mk, ai, values[i]))
                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

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


def split_cached_files(
    root_dirs: List[Path],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """收集所有 .npz 文件并按比例分为训练/验证集。"""
    all_files: List[Path] = []
    for d in root_dirs:
        all_files.extend(sorted(d.glob('*.npz')))

    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    return all_files[n_val:], all_files[:n_val]
