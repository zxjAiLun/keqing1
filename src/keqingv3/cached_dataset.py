"""keqingv3 缓存数据集：基础样本 + 多头辅助标签。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from training.cached_dataset import (
    DEFAULT_BUFFER_SIZE,
    GenericCachedMjaiDataset,
    V3AuxAdapter,
    split_cached_files,
)


class CachedMjaiDatasetV3(GenericCachedMjaiDataset):
    collate = staticmethod(V3AuxAdapter.collate)

    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
        super().__init__(
            file_paths,
            adapter=V3AuxAdapter(),
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )


__all__ = ["CachedMjaiDatasetV3", "split_cached_files"]
