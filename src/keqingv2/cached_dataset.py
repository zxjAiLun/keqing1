"""keqingv2 缓存数据集：在 v1 基础上额外透传 snap_json 字段（供 meld ranking loss 使用）。

与 keqingv1.cached_dataset 的区别：
- sample 为 8-tuple：(tile_feat, scalar, mask, action_idx, value,
                       tf_none, sc_none, tf_meld, sc_meld, rank_sign)
  其中 tf_none/sc_none/tf_meld/sc_meld 为预编码 numpy arrays（float32），
  rank_sign 为 float32（+1=GT副露, -1=GT none, 0=非 meld/none 样本）
- collate 将所有字段 stack 为 tensor
- 花色增强时对 snap_json_str 做字符串级牌名置换后再 encode
- 旧版 npz（无 snap_json 字段）graceful fallback，rank 字段全为零向量
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

# 复用 v1 的花色置换常量
_BUFFER_SIZE = 2000
_SUIT_SLICES = [slice(0, 9), slice(9, 18), slice(18, 27)]
_SUIT_PERMS = [
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ('m', 'p', 's')
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')

_MELD_ACTION_IDXS = frozenset([35, 36, 37, 38, 39, 40, 41])

# encode 输出维度（与 keqingv1/features.py 保持一致）
_TF_SHAPE = (54, 34)
_SC_SHAPE = (48,)
_ZERO_TF = np.zeros(_TF_SHAPE, dtype=np.float32)
_ZERO_SC = np.zeros(_SC_SHAPE, dtype=np.float32)


def _apply_suit_perm_arrays(
    tile_feat: np.ndarray,
    mask: np.ndarray,
    action_idx: int,
    perm: tuple,
) -> Tuple:
    """tile_feat/mask/action_idx 的花色置换（与 keqingv1.cached_dataset 逻辑相同）。"""
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


def _permute_snap_json(snap_json_str: str, perm: tuple) -> str:
    """对 snap_json 字符串里的牌名做花色置换。"""
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


def _make_fake_meld_snap(snap: dict, actor: int, meld_action: dict) -> dict:
    """构造副露后的 fake_snap，供 encode() 重算向听/役/进张。"""
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


def _encode_rank_pair(snap_json_str: str, action_idx: int):
    """Worker 内调用：对一个样本做 none/meld 双向 encode。

    返回 (tf_none, sc_none, tf_meld, sc_meld, rank_sign)。
    若失败或不适用返回 (zero, zero, zero, zero, 0.0)。
    """
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
        fake_snap = _make_fake_meld_snap(snap, actor, meld_action)
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


class CachedMjaiDatasetV2(IterableDataset):
    """读取 preprocess_v2.py 生成的 npz，额外预编码 rank loss 特征。"""

    def __init__(
        self,
        file_paths: List[Path],
        shuffle: bool = True,
        buffer_size: int = _BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
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
                data = np.load(path, allow_pickle=True)
                tile_feats = data['tile_feat'].astype(np.float32)
                scalars = data['scalar'].astype(np.float32)
                masks = data['mask'].astype(np.float32)
                action_idxs = data['action_idx'].astype(np.int64)
                values = data['value'].astype(np.float32)
                if 'snap_json' in data:
                    snap_jsons = data['snap_json']
                else:
                    snap_jsons = np.array([''] * len(tile_feats), dtype=object)
            except Exception:
                continue

            perms_to_use = []
            if self.aug_perms > 0:
                perms_to_use = rng.sample(_SUIT_PERMS, min(self.aug_perms, len(_SUIT_PERMS)))

            n = len(tile_feats)
            for i in range(n):
                sj = str(snap_jsons[i]) if snap_jsons[i] else ''
                ai = int(action_idxs[i])

                # 原始样本：在 worker 内预编码 rank pair
                tf_none, sc_none, tf_meld, sc_meld, rank_sign = _encode_rank_pair(sj, ai)
                sample = (
                    tile_feats[i], scalars[i], masks[i], ai, values[i],
                    tf_none, sc_none, tf_meld, sc_meld, rank_sign,
                )
                buffer.append(sample)

                for perm in perms_to_use:
                    tf, mk, new_ai = _apply_suit_perm_arrays(tile_feats[i], masks[i], ai, perm)
                    perm_sj = _permute_snap_json(sj, perm)
                    p_tf_none, p_sc_none, p_tf_meld, p_sc_meld, p_sign = _encode_rank_pair(perm_sj, new_ai)
                    buffer.append((
                        tf, scalars[i], mk, new_ai, values[i],
                        p_tf_none, p_sc_none, p_tf_meld, p_sc_meld, p_sign,
                    ))

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
    def collate(batch: List[Tuple]) -> Tuple:
        tile_feats, scalars, masks, action_idxs, values, \
            tf_nones, sc_nones, tf_melds, sc_melds, rank_signs = zip(*batch)
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
