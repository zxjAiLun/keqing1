from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import yaml

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl
from mahjong_env.replay import build_supervised_samples, extract_actor_names, read_mjai_jsonl, replay_validate_label_legal
from model.policy_value import PolicyValueModel
from model.vocab import build_action_vocab
from train.data_stats import summarize_samples
from train.dataset import OBS_DIM, SupervisedMjaiDataset


def _print_progress(prefix: str, step: int, total: int) -> None:
    if total <= 0:
        return
    width = 28
    ratio = min(max(step / total, 0.0), 1.0)
    fill = int(width * ratio)
    bar = "#" * fill + "-" * (width - fill)
    msg = f"\r{prefix} [{bar}] {step}/{total} ({ratio * 100:5.1f}%)"
    print(msg, end="", file=sys.stdout, flush=True)
    if step >= total:
        print()  # newline when complete


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - m)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)


def _weighted_policy_grad(
    probs: np.ndarray,
    labels: np.ndarray,
    legal_mask: np.ndarray,
    value_t: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, float]:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    adv = value_t - value_t.mean()
    w = np.exp(np.clip(beta * adv, -3.0, 3.0)).astype(np.float32)
    w = w / np.clip(w.mean(), 1e-6, None)
    grad_logits = ((probs - onehot) * w[:, None]) / max(len(labels), 1)
    grad_logits[legal_mask <= 0] = 0.0
    ce = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, None))
    return grad_logits, float((ce * w).mean())


def _batch(ds: SupervisedMjaiDataset, indices: np.ndarray, batch_size: int):
    for i in range(0, len(indices), batch_size):
        idxs = indices[i : i + batch_size]
        items = [ds[int(j)] for j in idxs]
        obs = np.stack([x[0] for x in items], axis=0)
        legal = np.stack([x[1] for x in items], axis=0)
        label = np.array([x[2] for x in items], dtype=np.int64)
        value = np.array([x[3] for x in items], dtype=np.float32)
        yield obs, legal, label, value


def _collect_log_paths(args: argparse.Namespace) -> List[str]:
    raw_json = getattr(args, "raw_json", None)
    raw_dir = getattr(args, "raw_dir", None)
    pre_converted_dir = getattr(args, "pre_converted_dir", None)
    log_path = getattr(args, "log_path", "log.jsonl")
    converted_out = getattr(args, "converted_out", "artifacts/converted/train_log.jsonl")
    converted_dir = Path(getattr(args, "converted_dir", "artifacts/converted/train"))
    libriichi_bin = getattr(args, "libriichi_bin", None)

    if pre_converted_dir:
        pre_path = Path(pre_converted_dir)
        if not pre_path.exists():
            raise RuntimeError(f"pre-converted-dir 不存在: {pre_converted_dir}")
        jsonl_files = sorted(pre_path.glob("*.jsonl"))
        if not jsonl_files:
            raise RuntimeError(f"pre-converted-dir 中没有找到 .jsonl 文件: {pre_converted_dir}")
        print(f"从已转换目录直接加载数据: {pre_converted_dir} ({len(jsonl_files)} 个文件)")
        return [str(f) for f in jsonl_files]

    if raw_json:
        convert_raw_to_mjai(raw_json, converted_out, libriichi_bin)
        errors = validate_mjai_jsonl(converted_out)
        if errors:
            raise RuntimeError(f"converted mjai validation failed: {errors}")
        return [converted_out]

    if raw_dir:
        converted_dir.mkdir(parents=True, exist_ok=True)
        out_paths: List[str] = []
        raw_dir_path = Path(raw_dir)
        if raw_dir_path.is_file() and raw_dir_path.suffix == ".json":
            src_files = [raw_dir_path]
        else:
            src_files = sorted(raw_dir_path.glob("*.json"))
        
        skip_existing = getattr(args, "skip_existing", False)
        total_files = len(src_files)
        converted_count = 0
        skipped_count = 0
        
        for i, src in enumerate(src_files, 1):
            out = converted_dir / f"{src.stem}.jsonl"
            
            if skip_existing and out.exists():
                out_paths.append(str(out))
                skipped_count += 1
                converted_count += 1
                _print_progress(f"Converting ({skipped_count} skipped)", converted_count, total_files)
                continue
            
            convert_raw_to_mjai(str(src), str(out), libriichi_bin)
            errors = validate_mjai_jsonl(out)
            if errors:
                raise RuntimeError(f"converted mjai validation failed for {src.name}: {errors}")
            out_paths.append(str(out))
            converted_count += 1
            _print_progress("Converting", converted_count, total_files)
        
        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个已存在的文件")
        if not out_paths:
            raise RuntimeError(f"no json files found in raw-dir: {raw_dir}")
        return out_paths

    if converted_dir.exists():
        jsonl_files = sorted(converted_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"从已转换目录加载数据: {converted_dir}")
            return [str(f) for f in jsonl_files]

    return [log_path]


def _view_name_filter(args: argparse.Namespace) -> Optional[Set[str]]:
    mode = getattr(args, "view_mode", "all")
    if mode == "all":
        return None
    if mode == "me":
        return {"私"}
    view_name = getattr(args, "view_name", "").strip()
    if not view_name:
        raise RuntimeError("view-mode=single requires --view-name")
    return {view_name}


def train(args: argparse.Namespace) -> Dict:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    log_paths = _collect_log_paths(args)
    name_filter = _view_name_filter(args)

    samples = []
    available_names: Set[str] = set()
    for p in log_paths:
        events = read_mjai_jsonl(p)
        available_names.update(extract_actor_names(events))
        samples.extend(build_supervised_samples(events, actor_name_filter=name_filter))
    if not samples:
        if name_filter is not None:
            raise RuntimeError(f"no samples for view-name={list(name_filter)[0]}, available={sorted(available_names)}")
        raise RuntimeError("no supervised samples collected")
    errors = replay_validate_label_legal(samples)
    if errors:
        raise RuntimeError(f"replay validation failed, first error: {errors[0]}")

    actions, stoi = build_action_vocab()
    ds = SupervisedMjaiDataset(samples, stoi)
    n_train = int(len(ds) * cfg["train_split"])
    n_val = len(ds) - n_train
    all_idx = np.arange(len(ds))
    np.random.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:] if n_val > 0 else all_idx[:]

    model = PolicyValueModel(
        obs_dim=OBS_DIM,
        hidden_dim=cfg["hidden_dim"],
        action_dim=len(actions),
    )
    best_val = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])
    policy_adv_beta = float(cfg.get("policy_adv_beta", 0.5))

    for epoch in range(cfg["num_epochs"]):
        train_loss = 0.0
        train_batch_count = 0
        train_total_batches = (len(train_idx) + cfg["batch_size"] - 1) // cfg["batch_size"]
        
        for obs, legal_mask, label, value_t in _batch(ds, train_idx, cfg["batch_size"]):
            logits_raw, value, _, h = model.forward_with_cache(obs)
            logits = model.masked_logits(logits_raw, legal_mask)
            probs = _softmax(logits)
            grad_logits, ce_loss = _weighted_policy_grad(probs, label, legal_mask, value_t, policy_adv_beta)
            grad_Wp = h.T @ grad_logits + wd * model.Wp
            grad_bp = grad_logits.sum(axis=0)
            grad_value = ((value.squeeze() - value_t.squeeze()) / max(len(label), 1)).reshape(-1, 1)
            grad_Wv = h.T @ grad_value + wd * model.Wv
            grad_bv = grad_value.sum()
            grad_h = grad_logits @ model.Wp.T + (grad_value @ model.Wv.T)
            grad_h[h <= 0] = 0.0
            grad_W1 = obs.T @ grad_h + wd * model.encoder.W1
            grad_b1 = grad_h.sum(axis=0)
            model.encoder.W1 -= lr * grad_W1
            model.encoder.b1 -= lr * grad_b1
            model.Wp -= lr * grad_Wp
            model.bp -= lr * grad_bp
            model.Wv -= lr * grad_Wv
            model.bv -= lr * grad_bv
            mse_loss = ((value - value_t) ** 2).mean()
            train_loss += float(ce_loss + 0.1 * mse_loss) * len(label)
            train_batch_count += 1
            _print_progress(f"Epoch {epoch+1} train", train_batch_count, train_total_batches)
        train_loss /= max(len(train_idx), 1)

        train_correct = 0
        train_total = 0
        for obs, legal_mask, label, value_t in _batch(ds, train_idx, cfg["batch_size"]):
            logits_raw, value, _, _ = model.forward_with_cache(obs)
            logits = model.masked_logits(logits_raw, legal_mask)
            preds = logits.argmax(axis=1)
            train_correct += int((preds == label).sum())
            train_total += len(label)
        train_acc = train_correct / max(train_total, 1)

        val_loss = 0.0
        val_batch_count = 0
        val_total_batches = (len(val_idx) + cfg["batch_size"] - 1) // cfg["batch_size"] if len(val_idx) > 0 else 0
        
        for obs, legal_mask, label, value_t in _batch(ds, val_idx, cfg["batch_size"]):
            logits, value, _ = model.forward(obs)
            logits = model.masked_logits(logits, legal_mask)
            probs = _softmax(logits)
            ce = -np.log(np.clip(probs[np.arange(len(label)), label], 1e-9, None))
            ce_loss = float(ce.mean())
            mse_loss = ((value - value_t) ** 2).mean()
            val_loss += float(ce_loss + 0.1 * mse_loss) * len(label)
            val_batch_count += 1
            _print_progress(f"Epoch {epoch+1} valid", val_batch_count, val_total_batches)
        val_loss /= max(len(val_idx), 1)

        val_correct = 0
        val_total = 0
        for obs, legal_mask, label, value_t in _batch(ds, val_idx, cfg["batch_size"]):
            logits, value, _ = model.forward(obs)
            logits = model.masked_logits(logits, legal_mask)
            preds = logits.argmax(axis=1)
            val_correct += int((preds == label).sum())
            val_total += len(label)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1}/{cfg['num_epochs']} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.2%} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.2%} | "
              f"best_val: {best_val:.4f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "action_vocab": actions,
            "config": cfg,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "view_mode": getattr(args, "view_mode", "all"),
            "view_name": getattr(args, "view_name", ""),
            "num_logs": len(log_paths),
        }
        np.savez(out_dir / "last.npz", **ckpt)
        if val_loss <= best_val:
            best_val = val_loss
            np.savez(out_dir / "best.npz", **ckpt)

    metrics = {
        "num_samples": len(ds),
        "best_val_loss": best_val,
        "num_logs": len(log_paths),
        "view_mode": getattr(args, "view_mode", "all"),
        "view_name": getattr(args, "view_name", ""),
    }
    metrics["sample_stats"] = summarize_samples(samples)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="log.jsonl")
    parser.add_argument("--raw-json", default=None, help="Raw tenhou/majsoul json. If set, convert first.")
    parser.add_argument("--raw-dir", default=None, help="Directory of tenhou6 json files. Convert and train on all files.")
    parser.add_argument("--pre-converted-dir", default=None, help="已转换的MJAI JSONL文件目录，直接训练不转换")
    parser.add_argument("--converted-out", default="artifacts/converted/train_log.jsonl")
    parser.add_argument("--converted-dir", default="artifacts/converted/train")
    parser.add_argument("--libriichi-bin", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="跳过已转换的文件（仅与 --raw-dir 一起使用）")
    parser.add_argument("--view-mode", choices=["all", "single", "me"], default="all")
    parser.add_argument("--view-name", default="keqing1")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--out-dir", default="artifacts/sl")
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

