"""共享缓存数据集：统一 npz 读取、shuffle buffer 和花色增强。"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from training.cache_schema import BASE_CACHE_FIELDS, MELD_RANK_EXTRA_FIELDS, V3_AUX_EXTRA_FIELDS

DEFAULT_BUFFER_SIZE = 2000

_SUIT_SLICES = [slice(0, 9), slice(9, 18), slice(18, 27)]
_SUIT_PERMS = [
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ("m", "p", "s")
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')

_MELD_ACTION_IDXS = frozenset([35, 36, 37, 38, 39, 40, 41])
_TF_SHAPE = (54, 34)
_SC_SHAPE = (48,)
_ZERO_TF = np.zeros(_TF_SHAPE, dtype=np.float32)
_ZERO_SC = np.zeros(_SC_SHAPE, dtype=np.float32)


def apply_suit_perm(tile_feat: np.ndarray, mask: np.ndarray, action_idx: int, perm: tuple) -> Tuple[np.ndarray, np.ndarray, int]:
    """在特征层面做花色置换。"""
    src = [_SUIT_SLICES[i] for i in perm]
    new_tile = tile_feat.copy()
    new_tile[:, _SUIT_SLICES[0]] = tile_feat[:, src[0]]
    new_tile[:, _SUIT_SLICES[1]] = tile_feat[:, src[1]]
    new_tile[:, _SUIT_SLICES[2]] = tile_feat[:, src[2]]

    new_action = action_idx
    if action_idx < 27:
        suit_i = action_idx // 9
        rank = action_idx % 9
        new_suit_i = perm.index(suit_i)
        new_action = new_suit_i * 9 + rank

    new_mask = mask.copy()
    new_mask[_SUIT_SLICES[0]] = mask[src[0]]
    new_mask[_SUIT_SLICES[1]] = mask[src[1]]
    new_mask[_SUIT_SLICES[2]] = mask[src[2]]

    return new_tile, new_mask, new_action


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


class BaseCacheAdapter:
    extra_fields: tuple[str, ...] = ()
    fail_fast = False

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        del path
        del data, sample_count
        return {}

    def permute_row_extra(self, row_extra: Dict[str, object], perm: tuple, action_idx: int) -> Dict[str, object]:
        del perm, action_idx
        return row_extra

    def permute_scalar(self, scalar: np.ndarray, perm: tuple) -> np.ndarray:
        del perm
        return scalar

    def build_sample(
        self,
        tile_feat: np.ndarray,
        scalar: np.ndarray,
        mask: np.ndarray,
        action_idx: int,
        value: float,
        row_extra: Dict[str, object],
    ) -> Tuple:
        del row_extra
        return tile_feat, scalar, mask, action_idx, value

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


class MeldRankAdapter(BaseCacheAdapter):
    extra_fields = MELD_RANK_EXTRA_FIELDS

    @staticmethod
    def _permute_snap_json(snap_json_str: str, perm: tuple) -> str:
        if not snap_json_str:
            return snap_json_str
        src_to_dst = {_SUITS[src]: _SUITS[dst] for dst, src in enumerate(perm)}

        def replace_pai(m: re.Match) -> str:
            num, suit = m.group(1), m.group(2)
            new_suit = src_to_dst[suit]
            full = m.group(0)
            if full.endswith('r"'):
                return f'"{num}{new_suit}r"'
            return f'"{num}{new_suit}"'

        return _PAI_RE.sub(replace_pai, snap_json_str)

    @staticmethod
    def _make_fake_meld_snap(snap: dict, actor: int, meld_action: dict) -> dict:
        from mahjong_env.tiles import normalize_tile

        meld_type = meld_action.get("type", "")
        consumed = meld_action.get("consumed", [])
        pai = meld_action.get("pai", "")

        fake = dict(snap)
        hand = list(snap.get("hand", []))
        new_hand = list(hand)
        for c in consumed:
            norm_c = normalize_tile(c)
            for i, t in enumerate(new_hand):
                if normalize_tile(t) == norm_c:
                    new_hand.pop(i)
                    break
        fake["hand"] = new_hand

        melds = [list(m) for m in snap.get("melds", [[], [], [], []])]
        melds[actor] = melds[actor] + [{
            "type": meld_type,
            "pai": normalize_tile(pai) if pai else "",
            "consumed": [normalize_tile(c) for c in consumed],
        }]
        fake["melds"] = melds
        fake.pop("shanten", None)
        fake.pop("waits_count", None)
        fake.pop("waits_tiles", None)
        return fake

    @staticmethod
    def _encode_rank_pair(snap_json_str: str, action_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32]:
        from keqingv1.features import encode

        if not snap_json_str:
            return _ZERO_TF, _ZERO_SC, _ZERO_TF, _ZERO_SC, np.float32(0.0)

        try:
            snap = json.loads(snap_json_str)
        except Exception:
            return _ZERO_TF, _ZERO_SC, _ZERO_TF, _ZERO_SC, np.float32(0.0)

        actor = snap.get("actor", 0)
        gt_is_meld = action_idx in _MELD_ACTION_IDXS
        meld_action = snap.get("label_action") if gt_is_meld else snap.get("meld_candidate")
        if meld_action is None:
            return _ZERO_TF, _ZERO_SC, _ZERO_TF, _ZERO_SC, np.float32(0.0)

        try:
            tf_none, sc_none = encode(snap, actor)
            fake_snap = MeldRankAdapter._make_fake_meld_snap(snap, actor, meld_action)
            tf_meld, sc_meld = encode(fake_snap, actor)
        except Exception:
            return _ZERO_TF, _ZERO_SC, _ZERO_TF, _ZERO_SC, np.float32(0.0)

        sign = np.float32(1.0) if gt_is_meld else np.float32(-1.0)
        return (
            tf_none.astype(np.float32),
            sc_none.astype(np.float32),
            tf_meld.astype(np.float32),
            sc_meld.astype(np.float32),
            sign,
        )

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        del path
        if "snap_json" in data:
            snap_json = data["snap_json"]
        else:
            snap_json = np.array([""] * sample_count, dtype=object)
        return {"snap_json": snap_json}

    def permute_row_extra(self, row_extra: Dict[str, object], perm: tuple, action_idx: int) -> Dict[str, object]:
        del action_idx
        snap_json = str(row_extra.get("snap_json") or "")
        return {"snap_json": self._permute_snap_json(snap_json, perm)}

    def build_sample(
        self,
        tile_feat: np.ndarray,
        scalar: np.ndarray,
        mask: np.ndarray,
        action_idx: int,
        value: float,
        row_extra: Dict[str, object],
    ) -> Tuple:
        snap_json = str(row_extra.get("snap_json") or "")
        tf_none, sc_none, tf_meld, sc_meld, rank_sign = self._encode_rank_pair(snap_json, action_idx)
        return (
            tile_feat,
            scalar,
            mask,
            action_idx,
            value,
            tf_none,
            sc_none,
            tf_meld,
            sc_meld,
            rank_sign,
        )

    @staticmethod
    def collate(batch: List[Tuple]) -> Tuple:
        tile_feats, scalars, masks, action_idxs, values, tf_nones, sc_nones, tf_melds, sc_melds, rank_signs = zip(*batch)
        return (
            torch.from_numpy(np.stack(tile_feats)),
            torch.from_numpy(np.stack(scalars)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(action_idxs, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
            torch.from_numpy(np.stack(tf_nones)),
            torch.from_numpy(np.stack(sc_nones)),
            torch.from_numpy(np.stack(tf_melds)),
            torch.from_numpy(np.stack(sc_melds)),
            torch.tensor(rank_signs, dtype=torch.float32),
        )


class V3AuxAdapter(BaseCacheAdapter):
    extra_fields = V3_AUX_EXTRA_FIELDS

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        del path
        score_delta = data["score_delta_target"] if "score_delta_target" in data else np.zeros(sample_count, dtype=np.float32)
        win = data["win_target"] if "win_target" in data else np.zeros(sample_count, dtype=np.float32)
        dealin = data["dealin_target"] if "dealin_target" in data else np.zeros(sample_count, dtype=np.float32)
        return {
            "score_delta_target": np.asarray(score_delta, dtype=np.float32),
            "win_target": np.asarray(win, dtype=np.float32),
            "dealin_target": np.asarray(dealin, dtype=np.float32),
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
        )

    def permute_scalar(self, scalar: np.ndarray, perm: tuple) -> np.ndarray:
        # Keep suit-dependent scalar semantics aligned with tile/action permutation:
        # [18:21] = aka_m / aka_p / aka_s flags
        # [30:33] = man / pin / sou tile ratios
        new_scalar = scalar.copy()
        for dst, src in enumerate(perm):
            new_scalar[18 + dst] = scalar[18 + src]
            new_scalar[30 + dst] = scalar[30 + src]
        return new_scalar

    @staticmethod
    def collate(batch: List[Tuple]) -> Tuple:
        tile_feats, scalars, masks, action_idxs, values, score_targets, win_targets, dealin_targets = zip(*batch)
        return (
            torch.from_numpy(np.stack(tile_feats)),
            torch.from_numpy(np.stack(scalars)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(action_idxs, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(score_targets, dtype=torch.float32),
            torch.tensor(win_targets, dtype=torch.float32),
            torch.tensor(dealin_targets, dtype=torch.float32),
        )


class GenericCachedMjaiDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        *,
        adapter: BaseCacheAdapter,
        shuffle: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
        self.file_paths = file_paths
        self.adapter = adapter
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
                data = np.load(path, allow_pickle=True)
                # Keep on-disk dtypes here to avoid whole-file host-side copies.
                # Cast only where required later in collate / device transfer.
                tile_feats = data[BASE_CACHE_FIELDS[0]]
                scalars = data[BASE_CACHE_FIELDS[1]]
                masks = data[BASE_CACHE_FIELDS[2]]
                action_idxs = data[BASE_CACHE_FIELDS[3]]
                values = data[BASE_CACHE_FIELDS[4]]
                extra_arrays = self.adapter.load_optional_arrays(data, len(tile_feats), path=path)
            except Exception:
                if self.adapter.fail_fast:
                    raise
                continue

            perms_to_use = []
            if self.aug_perms > 0:
                perms_to_use = rng.sample(_SUIT_PERMS, min(self.aug_perms, len(_SUIT_PERMS)))

            for i in range(len(tile_feats)):
                action_idx = int(action_idxs[i])
                row_extra = {key: extra_arrays[key][i] for key in extra_arrays}
                buffer.append(
                    self.adapter.build_sample(
                        tile_feats[i],
                        scalars[i],
                        masks[i],
                        action_idx,
                        values[i],
                        row_extra,
                    )
                )

                for perm in perms_to_use:
                    tf, mk, new_ai = apply_suit_perm(tile_feats[i], masks[i], action_idx, perm)
                    sc = self.adapter.permute_scalar(scalars[i], perm)
                    perm_extra = self.adapter.permute_row_extra(row_extra, perm, new_ai)
                    buffer.append(
                        self.adapter.build_sample(
                            tf,
                            sc,
                            mk,
                            new_ai,
                            values[i],
                            perm_extra,
                        )
                    )

                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
        for sample in buffer:
            yield sample


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "BaseCacheAdapter",
    "MeldRankAdapter",
    "V3AuxAdapter",
    "GenericCachedMjaiDataset",
    "apply_suit_perm",
    "split_cached_files",
]
