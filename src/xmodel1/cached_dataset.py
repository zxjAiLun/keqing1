"""Xmodel1 cached dataset scaffold.

Rust is expected to export candidate-centric `.npz` caches. Python consumes
those caches directly for training. This module intentionally keeps the first
implementation narrow: read validated arrays, expose an iterable dataset, and
provide a collate function for discard training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from xmodel1.schema import XMODEL1_DISCARD_REQUIRED_FIELDS

DEFAULT_BUFFER_SIZE = 512


def split_cached_files(
    root_dirs: List[Path],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    all_files: List[Path] = []
    for d in root_dirs:
        all_files.extend(sorted(d.glob("*.npz")))

    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    return all_files[n_val:], all_files[:n_val]


class Xmodel1DiscardDataset(IterableDataset):
    """Read discard-centric Xmodel1 caches exported by Rust."""

    def __init__(
        self,
        file_paths: List[Path],
        *,
        shuffle: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
    ):
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else random.randint(0, 2**31)

    def _iter_file_rows(self, path: Path) -> Iterator[Tuple]:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            return

        missing = [field for field in XMODEL1_DISCARD_REQUIRED_FIELDS if field not in data]
        if missing:
            raise ValueError(f"Xmodel1 cache missing required fields: {missing} in {path}")

        sample_count = int(data["state_tile_feat"].shape[0])
        for i in range(sample_count):
            yield (
                data["state_tile_feat"][i],
                data["state_scalar"][i],
                data["candidate_feat"][i],
                data["candidate_tile_id"][i],
                data["candidate_mask"][i],
                data["candidate_flags"][i],
                int(data["chosen_candidate_idx"][i]),
                data["candidate_quality_score"][i],
                data["candidate_rank_bucket"][i],
                data["candidate_hard_bad_flag"][i],
                np.float32(data["global_value_target"][i]),
                np.float32(data["score_delta_target"][i]),
                np.float32(data["win_target"][i]),
                np.float32(data["dealin_target"][i]),
                np.float32(data["offense_quality_target"][i]),
            )

    def __iter__(self) -> Iterator[Tuple]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.file_paths
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        rng = random.Random(self.seed)
        paths = list(paths)
        if self.shuffle:
            rng.shuffle(paths)

        buffer: List[Tuple] = []
        for path in paths:
            for row in self._iter_file_rows(path):
                buffer.append(row)
                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
        for row in buffer:
            yield row

    @staticmethod
    def collate(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        (
            state_tile_feat,
            state_scalar,
            candidate_feat,
            candidate_tile_id,
            candidate_mask,
            candidate_flags,
            chosen_candidate_idx,
            candidate_quality_score,
            candidate_rank_bucket,
            candidate_hard_bad_flag,
            global_value_target,
            score_delta_target,
            win_target,
            dealin_target,
            offense_quality_target,
        ) = zip(*batch)
        return {
            "state_tile_feat": torch.from_numpy(np.stack(state_tile_feat)),
            "state_scalar": torch.from_numpy(np.stack(state_scalar)),
            "candidate_feat": torch.from_numpy(np.stack(candidate_feat)),
            "candidate_tile_id": torch.from_numpy(np.stack(candidate_tile_id)).long(),
            "candidate_mask": torch.from_numpy(np.stack(candidate_mask)),
            "candidate_flags": torch.from_numpy(np.stack(candidate_flags)),
            "chosen_candidate_idx": torch.tensor(chosen_candidate_idx, dtype=torch.long),
            "candidate_quality_score": torch.from_numpy(np.stack(candidate_quality_score)).float(),
            "candidate_rank_bucket": torch.from_numpy(np.stack(candidate_rank_bucket)).long(),
            "candidate_hard_bad_flag": torch.from_numpy(np.stack(candidate_hard_bad_flag)).float(),
            "global_value_target": torch.tensor(global_value_target, dtype=torch.float32),
            "score_delta_target": torch.tensor(score_delta_target, dtype=torch.float32),
            "win_target": torch.tensor(win_target, dtype=torch.float32),
            "dealin_target": torch.tensor(dealin_target, dtype=torch.float32),
            "offense_quality_target": torch.tensor(offense_quality_target, dtype=torch.float32),
        }


__all__ = ["DEFAULT_BUFFER_SIZE", "Xmodel1DiscardDataset", "split_cached_files"]
