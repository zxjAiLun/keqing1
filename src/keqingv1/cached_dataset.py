"""keqingv1 缓存数据集兼容层：复用共享 cached dataset。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from training.cached_dataset import (
    DEFAULT_BUFFER_SIZE,
    BaseCacheAdapter,
    GenericCachedMjaiDataset,
    split_cached_files,
)


class CachedMjaiDataset(GenericCachedMjaiDataset):
    collate = staticmethod(BaseCacheAdapter.collate)

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
            adapter=BaseCacheAdapter(),
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )


__all__ = ["CachedMjaiDataset", "split_cached_files"]
