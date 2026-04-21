"""xmodel2 cached dataset on top of the shared base cache schema."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from training.cache_schema import XMODEL2_EXTRA_FIELDS
from training.cached_dataset import (
    DEFAULT_BUFFER_SIZE,
    BaseCacheAdapter,
    GenericCachedMjaiDataset,
    split_cached_files,
)


class Xmodel2CacheAdapter(BaseCacheAdapter):
    extra_fields = XMODEL2_EXTRA_FIELDS
    fail_fast = True

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        missing = [field for field in self.extra_fields if field not in data]
        if missing:
            raise ValueError(f"xmodel2 cache missing required fields {missing} in {path}")
        return {
            "score_delta_target": np.asarray(data["score_delta_target"], dtype=np.float32).reshape(sample_count),
            "win_target": np.asarray(data["win_target"], dtype=np.float32).reshape(sample_count),
            "dealin_target": np.asarray(data["dealin_target"], dtype=np.float32).reshape(sample_count),
            "pts_given_win_target": np.asarray(data["pts_given_win_target"], dtype=np.float32).reshape(sample_count),
            "pts_given_dealin_target": np.asarray(data["pts_given_dealin_target"], dtype=np.float32).reshape(sample_count),
            "opp_tenpai_target": np.asarray(data["opp_tenpai_target"], dtype=np.float32).reshape(sample_count, 3),
            "final_rank_target": np.asarray(data["final_rank_target"], dtype=np.int8).reshape(sample_count),
            "final_score_delta_points_target": np.asarray(
                data["final_score_delta_points_target"],
                dtype=np.int32,
            ).reshape(sample_count),
        }

    def build_sample(
        self,
        tile_feat: np.ndarray,
        scalar: np.ndarray,
        mask: np.ndarray,
        action_idx: int,
        value: float,
        row_extra: Dict[str, object],
    ) -> Tuple:
        return (
            tile_feat,
            scalar,
            mask,
            action_idx,
            value,
            np.float32(row_extra.get("score_delta_target", 0.0)),
            np.float32(row_extra.get("win_target", 0.0)),
            np.float32(row_extra.get("dealin_target", 0.0)),
            np.float32(row_extra.get("pts_given_win_target", 0.0)),
            np.float32(row_extra.get("pts_given_dealin_target", 0.0)),
            np.asarray(
                row_extra.get("opp_tenpai_target", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3),
            np.int8(row_extra.get("final_rank_target", 0)),
            np.int32(row_extra.get("final_score_delta_points_target", 0)),
        )

    def permute_row_extra(self, row_extra: Dict[str, object], perm: tuple, action_idx: int) -> Dict[str, object]:
        del perm, action_idx
        return {
            "score_delta_target": np.float32(row_extra.get("score_delta_target", 0.0)),
            "win_target": np.float32(row_extra.get("win_target", 0.0)),
            "dealin_target": np.float32(row_extra.get("dealin_target", 0.0)),
            "pts_given_win_target": np.float32(row_extra.get("pts_given_win_target", 0.0)),
            "pts_given_dealin_target": np.float32(row_extra.get("pts_given_dealin_target", 0.0)),
            "opp_tenpai_target": np.asarray(
                row_extra.get("opp_tenpai_target", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3),
            "final_rank_target": np.int8(row_extra.get("final_rank_target", 0)),
            "final_score_delta_points_target": np.int32(row_extra.get("final_score_delta_points_target", 0)),
        }

    @staticmethod
    def collate(batch: List[Tuple]) -> Tuple:
        (
            tile_feats,
            scalars,
            masks,
            action_idxs,
            values,
            score_targets,
            win_targets,
            dealin_targets,
            pts_given_win_targets,
            pts_given_dealin_targets,
            opp_tenpai_targets,
            final_rank_targets,
            final_score_delta_points_targets,
        ) = zip(*batch)
        return (
            torch.from_numpy(np.stack(tile_feats)),
            torch.from_numpy(np.stack(scalars)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(action_idxs, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(score_targets, dtype=torch.float32),
            torch.tensor(win_targets, dtype=torch.float32),
            torch.tensor(dealin_targets, dtype=torch.float32),
            torch.tensor(pts_given_win_targets, dtype=torch.float32),
            torch.tensor(pts_given_dealin_targets, dtype=torch.float32),
            torch.from_numpy(np.stack(opp_tenpai_targets)).float(),
            torch.tensor(final_rank_targets, dtype=torch.long),
            torch.tensor(final_score_delta_points_targets, dtype=torch.int32),
        )


class CachedMjaiDatasetXmodel2(GenericCachedMjaiDataset):
    collate = staticmethod(Xmodel2CacheAdapter.collate)

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
            adapter=Xmodel2CacheAdapter(),
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )


__all__ = ["CachedMjaiDatasetXmodel2", "Xmodel2CacheAdapter", "split_cached_files"]
