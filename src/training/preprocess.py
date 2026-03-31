from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from keqingv1.action_space import (
    ANKAN_IDX,
    CHI_HIGH_IDX,
    CHI_LOW_IDX,
    CHI_MID_IDX,
    DAIMINKAN_IDX,
    KAKAN_IDX,
    NONE_IDX,
    PON_IDX,
    action_to_idx,
    build_legal_mask,
)
from keqingv1.features import encode

_MELD_ACTION_IDXS = {
    CHI_LOW_IDX,
    CHI_MID_IDX,
    CHI_HIGH_IDX,
    PON_IDX,
    DAIMINKAN_IDX,
    ANKAN_IDX,
    KAKAN_IDX,
}
_MELD_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}

_SUIT_PERMS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ("m", "p", "s")
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')


def _permute_mjson_text(text: str, perm: tuple) -> str:
    src_to_dst = {_SUITS[src]: _SUITS[dst] for dst, src in enumerate(perm)}

    def replace_pai(m: re.Match) -> str:
        num, suit = m.group(1), m.group(2)
        new_suit = src_to_dst[suit]
        full = m.group(0)
        if full.endswith('r"'):
            return f'"{num}{new_suit}r"'
        return f'"{num}{new_suit}"'

    return _PAI_RE.sub(replace_pai, text)


class BasePreprocessAdapter:
    extra_field_names: tuple[str, ...] = ()

    def init_rows(self) -> Dict[str, list]:
        return {}

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        del sample, action_idx
        return {}

    def permute_result_extras(self, result: Dict[str, np.ndarray], perm: tuple) -> Dict[str, np.ndarray]:
        del perm
        return {
            name: result[name]
            for name in self.extra_field_names
            if name in result
        }


class MeldRankPreprocessAdapter(BasePreprocessAdapter):
    extra_field_names = ("snap_json",)

    @staticmethod
    def _permute_snap_json(snap_json_str: str, perm: tuple) -> str:
        if not snap_json_str:
            return snap_json_str
        return _permute_mjson_text(snap_json_str, perm)

    @staticmethod
    def _snap_min(state: dict, actor: int, label_action: dict, legal_dicts: list) -> str:
        meld_candidate: Optional[dict] = None
        action_idx_val = action_to_idx(label_action)
        if action_idx_val == NONE_IDX:
            meld_priority = ["daiminkan", "ankan", "kakan", "pon", "chi"]
            candidates = {a.get("type"): a for a in legal_dicts if a.get("type") in _MELD_TYPES}
            for mtype in meld_priority:
                if mtype in candidates:
                    meld_candidate = candidates[mtype]
                    break

        snap = {
            "hand": state.get("hand", []),
            "melds": state.get("melds", [[], [], [], []]),
            "discards": state.get("discards", [[], [], [], []]),
            "dora_markers": state.get("dora_markers", []),
            "reached": state.get("reached", [False, False, False, False]),
            "scores": state.get("scores", [25000, 25000, 25000, 25000]),
            "bakaze": state.get("bakaze", "E"),
            "kyoku": state.get("kyoku", 1),
            "honba": state.get("honba", 0),
            "kyotaku": state.get("kyotaku", 0),
            "jikaze": state.get("jikaze", 0),
            "oya": state.get("oya", 0),
            "actor": actor,
            "label_action": label_action,
            "meld_candidate": meld_candidate,
        }
        return json.dumps(snap, ensure_ascii=False)

    def init_rows(self) -> Dict[str, list]:
        return {"snap_json": []}

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        if action_idx in _MELD_ACTION_IDXS or action_idx == NONE_IDX:
            snap_json_str = self._snap_min(
                sample.state,
                sample.actor,
                sample.label_action,
                sample.legal_actions,
            )
        else:
            snap_json_str = ""
        return {"snap_json": snap_json_str}

    def permute_result_extras(self, result: Dict[str, np.ndarray], perm: tuple) -> Dict[str, np.ndarray]:
        if "snap_json" not in result:
            return {}
        return {
            "snap_json": np.array(
                [self._permute_snap_json(s, perm) for s in result["snap_json"]],
                dtype=object,
            )
        }


class V3PreprocessAdapter(BasePreprocessAdapter):
    extra_field_names = ("score_delta_target", "win_target", "dealin_target")

    def init_rows(self) -> Dict[str, list]:
        return {
            "score_delta_target": [],
            "win_target": [],
            "dealin_target": [],
        }

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        del action_idx
        return {
            "score_delta_target": float(np.clip(sample.score_delta_target, -1.0, 1.0)),
            "win_target": float(sample.win_target),
            "dealin_target": float(sample.dealin_target),
        }


def _parse_events_to_arrays(
    events,
    *,
    actor_name_filter=None,
    adapter: BasePreprocessAdapter,
    value_strategy: str = "heuristic",
    encode_fn=encode,
) -> Optional[Dict[str, np.ndarray]]:
    rows = {
        "tile_feat": [],
        "scalar": [],
        "mask": [],
        "action_idx": [],
        "value": [],
    }
    rows.update(adapter.init_rows())

    try:
        samples = build_supervised_samples(
            events,
            actor_name_filter=actor_name_filter,
            value_strategy=value_strategy,
        )
    except Exception:
        return None

    for s in samples:
        try:
            tile_feat, scalar = encode_fn(s.state, s.actor)
            mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
            action_idx = action_to_idx(s.label_action)
            if mask[action_idx] == 0:
                continue
            value = float(np.clip(s.value_target, -1.0, 1.0))
            rows["tile_feat"].append(tile_feat)
            rows["scalar"].append(scalar)
            rows["mask"].append(mask)
            rows["action_idx"].append(action_idx)
            rows["value"].append(value)
            for key, val in adapter.sample_extras(s, action_idx).items():
                rows[key].append(val)
        except Exception:
            continue

    if not rows["tile_feat"]:
        return None

    result: Dict[str, np.ndarray] = {
        "tile_feat": np.stack(rows["tile_feat"]).astype(np.float16),
        "scalar": np.stack(rows["scalar"]).astype(np.float16),
        "mask": np.stack(rows["mask"]).astype(np.uint8),
        "action_idx": np.array(rows["action_idx"], dtype=np.int16),
        "value": np.array(rows["value"], dtype=np.float16),
    }
    for key in adapter.extra_field_names:
        if key in {"score_delta_target", "win_target", "dealin_target"}:
            result[key] = np.array(rows[key], dtype=np.float16)
        else:
            result[key] = np.array(rows[key], dtype=object)
    return result


def _concat_results(results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(results[0].keys())
    return {key: np.concatenate([r[key] for r in results], axis=0) for key in keys}


def process_file(args: Tuple) -> Tuple[str, int]:
    src_path, out_path, augment, actor_name_filter, adapter, value_strategy, encode_module = args
    if out_path.exists():
        return ("skip", 0)
    try:
        text = src_path.read_text(encoding="utf-8")
        events = read_mjai_jsonl(src_path)
    except Exception:
        return ("error", 0)

    all_results: List[Dict[str, np.ndarray]] = []
    encode_fn = importlib.import_module(encode_module).encode

    result = _parse_events_to_arrays(
        events,
        actor_name_filter=actor_name_filter,
        adapter=adapter,
        value_strategy=value_strategy,
        encode_fn=encode_fn,
    )
    if result is not None:
        all_results.append(result)

    if augment:
        for perm in _SUIT_PERMS[1:]:
            permuted_text = _permute_mjson_text(text, perm)
            try:
                perm_events = [json.loads(line) for line in permuted_text.splitlines() if line.strip()]
                result = _parse_events_to_arrays(
                    perm_events,
                    actor_name_filter=actor_name_filter,
                    adapter=adapter,
                    value_strategy=value_strategy,
                    encode_fn=encode_fn,
                )
                if result is not None:
                    result.update(adapter.permute_result_extras(result, perm))
                    all_results.append(result)
            except Exception:
                pass

    if not all_results:
        return ("empty", 0)

    merged = _concat_results(all_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    return ("ok", len(merged["tile_feat"]))


def run_preprocess(
    *,
    default_output_dir: str,
    adapter: BasePreprocessAdapter,
    default_value_strategy: str = "heuristic",
    encode_module: str = "keqingv1.features",
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--actor_name_filter", nargs="+", type=str, default=None)
    parser.add_argument(
        "--value-strategy",
        type=str,
        default=None,
        choices=["heuristic", "mc_return"],
    )
    args = parser.parse_args()

    cfg: dict = {}
    data_dirs: List[str] = []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        data_dirs = cfg.get("data_dirs", [])
    if args.data_dirs:
        data_dirs = args.data_dirs

    if not data_dirs:
        print("错误：请通过 --data_dirs 或 --config 指定数据目录")
        sys.exit(1)

    output_dir_value = args.output_dir or cfg.get("output_dir", default_output_dir)
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else int(cfg.get("workers", max(1, cpu_count() - 2)))
    augment = args.augment if args.augment is not None else bool(cfg.get("augment", False))
    actor_name_filter_raw = args.actor_name_filter if args.actor_name_filter is not None else cfg.get("actor_name_filter")
    actor_name_filter = set(actor_name_filter_raw) if actor_name_filter_raw else None
    value_strategy = args.value_strategy or cfg.get("value_strategy", default_value_strategy)

    tasks = []
    for data_dir in data_dirs:
        src_dir = Path(data_dir)
        ds_name = src_dir.name
        out_ds_dir = output_dir / ds_name
        for mjson_file in sorted(src_dir.glob("*.mjson")):
            out_file = out_ds_dir / (mjson_file.stem + ".npz")
            tasks.append((
                mjson_file,
                out_file,
                augment,
                actor_name_filter,
                adapter,
                value_strategy,
                encode_module,
            ))

    total = len(tasks)
    filter_info = f"，actor_filter={list(actor_name_filter)}" if actor_name_filter else ""
    print(
        f"共 {total} 个文件，使用 {workers} 个进程，augment={augment} "
        f"value_strategy={value_strategy}{filter_info}"
    )

    done = skipped = errors = total_samples = 0
    with Pool(processes=workers) as pool:
        for status, n_samples in pool.imap_unordered(process_file, tasks, chunksize=4):
            done += 1
            if status == "ok":
                total_samples += n_samples
            elif status == "skip":
                skipped += 1
            elif status in ("error", "empty"):
                errors += 1
            if done % 200 == 0 or done == total:
                print(f"  [{done}/{total}] 跳过={skipped} 错误={errors} 样本={total_samples:,}")

    print(f"\n完成！总样本数: {total_samples:,}，跳过: {skipped}，错误: {errors}")


__all__ = [
    "BasePreprocessAdapter",
    "MeldRankPreprocessAdapter",
    "run_preprocess",
]
