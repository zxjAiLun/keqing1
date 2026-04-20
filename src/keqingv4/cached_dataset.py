"""keqingv4 cached dataset with typed preprocess summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from training.cache_schema import (
    KEQINGV4_EXTRA_FIELDS,
    KEQINGV4_SUMMARY_DIM,
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_OPPORTUNITY_DIM,
)
from training.cached_dataset import (
    DEFAULT_BUFFER_SIZE,
    BaseCacheAdapter,
    GenericCachedMjaiDataset,
    split_cached_files,
)
from keqingv4.cache_contract import validate_keqingv4_npz


class KeqingV4CacheAdapter(BaseCacheAdapter):
    extra_fields = KEQINGV4_EXTRA_FIELDS
    fail_fast = True

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        problems = validate_keqingv4_npz(data, path=path, require_full_contract=True)
        if problems:
            raise ValueError("; ".join(problems))
        score_delta = data["score_delta_target"]
        win = data["win_target"]
        dealin = data["dealin_target"]
        pts_given_win = data["pts_given_win_target"]
        pts_given_dealin = data["pts_given_dealin_target"]
        opp_tenpai = data["opp_tenpai_target"]
        event_history = data["event_history"]
        v4_opportunity = data["v4_opportunity"]
        discard = data["v4_discard_summary"]
        call = data["v4_call_summary"]
        special = data["v4_special_summary"]
        return {
            "score_delta_target": np.asarray(score_delta, dtype=np.float32),
            "win_target": np.asarray(win, dtype=np.float32),
            "dealin_target": np.asarray(dealin, dtype=np.float32),
            "pts_given_win_target": np.asarray(pts_given_win, dtype=np.float32),
            "pts_given_dealin_target": np.asarray(pts_given_dealin, dtype=np.float32),
            "opp_tenpai_target": np.asarray(opp_tenpai, dtype=np.float32).reshape(sample_count, 3),
            "pts_given_win_available": np.ones((sample_count,), dtype=bool),
            "pts_given_dealin_available": np.ones((sample_count,), dtype=bool),
            "opp_tenpai_available": np.ones((sample_count,), dtype=bool),
            "event_history": np.asarray(event_history, dtype=np.int16).reshape(
                sample_count, KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM
            ),
            "event_history_available": np.ones((sample_count,), dtype=bool),
            "v4_opportunity": np.asarray(v4_opportunity, dtype=np.uint8).reshape(sample_count, KEQINGV4_OPPORTUNITY_DIM),
            "v4_discard_summary": np.asarray(discard, dtype=np.float16),
            "v4_call_summary": np.asarray(call, dtype=np.float16),
            "v4_special_summary": np.asarray(special, dtype=np.float16),
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
            np.asarray(row_extra.get("opp_tenpai_target", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3),
            np.bool_(row_extra.get("pts_given_win_available", False)),
            np.bool_(row_extra.get("pts_given_dealin_available", False)),
            np.bool_(row_extra.get("opp_tenpai_available", False)),
            np.asarray(row_extra["event_history"], dtype=np.int16).reshape(
                KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM
            ),
            np.bool_(row_extra.get("event_history_available", False)),
            np.asarray(row_extra["v4_opportunity"], dtype=np.uint8).reshape(KEQINGV4_OPPORTUNITY_DIM),
            np.asarray(row_extra.get("v4_discard_summary"), dtype=np.float16),
            np.asarray(row_extra.get("v4_call_summary"), dtype=np.float16),
            np.asarray(row_extra.get("v4_special_summary"), dtype=np.float16),
        )

    def permute_row_extra(self, row_extra: Dict[str, object], perm: tuple, action_idx: int) -> Dict[str, object]:
        del action_idx
        discard = np.asarray(row_extra.get("v4_discard_summary"), dtype=np.float16).copy()
        src_slices = [slice(i * 9, (i + 1) * 9) for i in perm]
        discard[0:9] = discard[src_slices[0]]
        discard[9:18] = discard[src_slices[1]]
        discard[18:27] = discard[src_slices[2]]
        return {
            "score_delta_target": np.float32(row_extra.get("score_delta_target", 0.0)),
            "win_target": np.float32(row_extra.get("win_target", 0.0)),
            "dealin_target": np.float32(row_extra.get("dealin_target", 0.0)),
            "pts_given_win_target": np.float32(row_extra.get("pts_given_win_target", 0.0)),
            "pts_given_dealin_target": np.float32(row_extra.get("pts_given_dealin_target", 0.0)),
            "opp_tenpai_target": np.asarray(row_extra.get("opp_tenpai_target", np.zeros((3,), dtype=np.float32)), dtype=np.float32),
            "pts_given_win_available": np.bool_(row_extra.get("pts_given_win_available", False)),
            "pts_given_dealin_available": np.bool_(row_extra.get("pts_given_dealin_available", False)),
            "opp_tenpai_available": np.bool_(row_extra.get("opp_tenpai_available", False)),
            "event_history": np.asarray(row_extra["event_history"], dtype=np.int16),
            "event_history_available": np.bool_(row_extra.get("event_history_available", False)),
            "v4_opportunity": np.asarray(row_extra["v4_opportunity"], dtype=np.uint8).reshape(KEQINGV4_OPPORTUNITY_DIM),
            "v4_discard_summary": discard,
            "v4_call_summary": np.asarray(row_extra.get("v4_call_summary"), dtype=np.float16),
            "v4_special_summary": np.asarray(row_extra.get("v4_special_summary"), dtype=np.float16),
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
            pts_given_win_available,
            pts_given_dealin_available,
            opp_tenpai_available,
            event_history,
            event_history_available,
            v4_opportunity,
            discard_summary,
            call_summary,
            special_summary,
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
            torch.tensor(pts_given_win_available, dtype=torch.bool),
            torch.tensor(pts_given_dealin_available, dtype=torch.bool),
            torch.tensor(opp_tenpai_available, dtype=torch.bool),
            torch.from_numpy(np.stack(event_history)).long(),
            torch.tensor(event_history_available, dtype=torch.bool),
            torch.from_numpy(np.stack(v4_opportunity)).bool(),
            torch.from_numpy(np.stack(discard_summary)),
            torch.from_numpy(np.stack(call_summary)),
            torch.from_numpy(np.stack(special_summary)),
        )


class CachedMjaiDatasetV4(GenericCachedMjaiDataset):
    collate = staticmethod(KeqingV4CacheAdapter.collate)

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
            adapter=KeqingV4CacheAdapter(),
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )


__all__ = ["CachedMjaiDatasetV4", "KeqingV4CacheAdapter", "split_cached_files"]
