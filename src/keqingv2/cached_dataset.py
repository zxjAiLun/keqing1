"""keqingv2 缓存数据集兼容层：复用共享 cached dataset + meld rank adapter。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from training.cached_dataset import (
    BaseCacheAdapter,
    GenericCachedMjaiDataset,
    MeldRankAdapter,
)

_BUFFER_SIZE = 1000


class CachedMjaiDatasetV2(GenericCachedMjaiDataset):
    collate = staticmethod(MeldRankAdapter.collate)

    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = _BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
        super().__init__(
            file_paths,
            adapter=MeldRankAdapter(),
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )


__all__ = ["CachedMjaiDatasetV2", "BaseCacheAdapter"]
