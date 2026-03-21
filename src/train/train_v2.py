from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

_TorchBase = nn.Module if nn is not None else object

from mahjong_env.replay import (
    build_supervised_samples,
    extract_actor_names,
    read_mjai_jsonl,
    replay_validate_label_legal,
)
from model.policy_value import MultiTaskModel
from model.vocab import build_action_vocab
from train.data_stats import summarize_samples
from train.dataset import OBS_DIM, SupervisedMjaiDataset


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
) -> Tuple[np.ndarray, float]:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    adv = value_t - value_t.mean()
    w = np.exp(np.clip(beta * adv, -3.0, 3.0)).astype(np.float32)
    w = w / np.clip(w.mean(), 1e-6, None)
    grad_logits = ((probs - onehot) * w[:, None]) / max(len(labels), 1)
    grad_logits[legal_mask <= 0] = 0.0
    ce = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, None))
    return grad_logits, float((ce * w).mean())


def _collect_log_paths(args: argparse.Namespace) -> List[str]:
    raw_json = getattr(args, "raw_json", None)
    raw_dir = getattr(args, "raw_dir", None)
    pre_converted_dir = getattr(args, "pre_converted_dir", None)
    log_path = getattr(args, "log_path", "log.jsonl")
    converted_out = getattr(args, "converted_out", "artifacts/converted/train_log.jsonl")
    converted_dir = Path(getattr(args, "converted_dir", "artifacts/converted/train"))
    libriichi_bin = getattr(args, "libriichi_bin", None)

    from convert.libriichi_bridge import convert_raw_to_mjai
    from convert.validate_mjai import validate_mjai_jsonl

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
            # Some folders may contain helper/report json files that are not tenhou6 game logs.
            # Keep this list short and explicit to avoid silently skipping real data.
            ignore_names = {"browser_export_report.json"}
            src_files = sorted(x for x in raw_dir_path.glob("*.json") if x.name not in ignore_names)
        for src in src_files:
            out = converted_dir / f"{src.stem}.jsonl"
            convert_raw_to_mjai(str(src), str(out), libriichi_bin)
            errors = validate_mjai_jsonl(out)
            if errors:
                raise RuntimeError(f"converted mjai validation failed for {src.name}: {errors}")
            out_paths.append(str(out))
        if not out_paths:
            raise RuntimeError(f"no json files found in raw-dir: {raw_dir}")
        return out_paths

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


def _batch(ds: SupervisedMjaiDataset, indices: np.ndarray, batch_size: int):
    for i in range(0, len(indices), batch_size):
        idxs = indices[i : i + batch_size]
        items = [ds[int(j)] for j in idxs]
        obs = np.stack([x[0] for x in items], axis=0)
        legal = np.stack([x[1] for x in items], axis=0)
        label = np.array([x[2] for x in items], dtype=np.int64)
        value = np.array([x[3] for x in items], dtype=np.float32)
        yield obs, legal, label, value


def _compute_rank_proxy(value_t: np.ndarray, num_ranks: int) -> np.ndarray:
    rank = (value_t * (num_ranks - 1)).astype(np.int32)
    rank = np.clip(rank, 0, num_ranks - 1)
    return rank


def _onehot_rank(rank: np.ndarray, num_ranks: int) -> np.ndarray:
    onehot = np.zeros((len(rank), num_ranks), dtype=np.float32)
    onehot[np.arange(len(rank)), rank] = 1.0
    return onehot


def _get_lr(epoch: int, base_lr: float, num_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        t = epoch / max(1, warmup_epochs)
        return base_lr * (0.1 + 0.9 * t)
    phase = max(0, epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * phase))


def _clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    total_norm = math.sqrt(sum(np.sum(g * g) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {k: g * scale for k, g in grads.items()}
    return grads


def _batch_count(n_items: int, batch_size: int) -> int:
    if n_items <= 0:
        return 0
    return (n_items + batch_size - 1) // batch_size


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
        print("", file=sys.stdout, flush=True)


def _make_output_dir(base_out_dir: str, model_name: str | None = None) -> Path:
    """生成带时间戳和模型名的输出目录"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if model_name:
        dir_name = f"{model_name}_{timestamp}"
    else:
        dir_name = f"train_{timestamp}"
    
    out_dir = Path(base_out_dir) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")
    return out_dir


class _TorchMultiTaskModel(_TorchBase):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, num_ranks: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.aux = nn.Linear(hidden_dim, num_ranks)

    def forward(self, obs: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc3(F.relu(self.fc2(h))) + h)
        h = F.relu(self.fc5(F.relu(self.fc4(h))) + h)
        return self.policy(h), self.value(h), self.aux(h)


def _torch_state_to_numpy_dict(model: "_TorchMultiTaskModel") -> Dict[str, np.ndarray]:
    state = {
        "W1": model.fc1.weight.detach().cpu().numpy().T.astype(np.float32),
        "b1": model.fc1.bias.detach().cpu().numpy().astype(np.float32),
        "W2": model.fc2.weight.detach().cpu().numpy().T.astype(np.float32),
        "b2": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "W3": model.fc3.weight.detach().cpu().numpy().T.astype(np.float32),
        "b3": model.fc3.bias.detach().cpu().numpy().astype(np.float32),
        "W4": model.fc4.weight.detach().cpu().numpy().T.astype(np.float32),
        "b4": model.fc4.bias.detach().cpu().numpy().astype(np.float32),
        "W5": model.fc5.weight.detach().cpu().numpy().T.astype(np.float32),
        "b5": model.fc5.bias.detach().cpu().numpy().astype(np.float32),
        "Wp": model.policy.weight.detach().cpu().numpy().T.astype(np.float32),
        "bp": model.policy.bias.detach().cpu().numpy().astype(np.float32),
        "Wv": model.value.weight.detach().cpu().numpy().T.astype(np.float32),
        "bv": model.value.bias.detach().cpu().numpy().astype(np.float32),
        "Waux": model.aux.weight.detach().cpu().numpy().T.astype(np.float32),
        "baux": model.aux.bias.detach().cpu().numpy().astype(np.float32),
    }
    return state


def _load_npz_checkpoint(path: str) -> Dict:
    ckpt_npz = np.load(path, allow_pickle=True)
    return {
        k: ckpt_npz[k].item() if ckpt_npz[k].dtype == object else ckpt_npz[k]
        for k in ckpt_npz.files
    }


def _train_torch(
    args: argparse.Namespace,
    cfg: Dict,
    ds: SupervisedMjaiDataset,
    actions: List[str],
    samples: List,
    log_paths: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict:
    if torch is None:
        raise RuntimeError("PyTorch 不可用，请先安装 torch（支持 CUDA 版本）。")

    requested_device = getattr(args, "device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你指定了 --device cuda，但当前环境未检测到 CUDA。")
    else:
        device = requested_device

    num_ranks = int(cfg.get("num_ranks", 5))
    model = _TorchMultiTaskModel(
        obs_dim=OBS_DIM,
        hidden_dim=int(cfg["hidden_dim"]),
        action_dim=len(actions),
        num_ranks=num_ranks,
    ).to(device)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint:
        ckpt = _load_npz_checkpoint(init_checkpoint)
        msd = ckpt["model_state_dict"]
        with torch.no_grad():
            model.fc1.weight.copy_(torch.from_numpy(msd["W1"].T).to(device=device, dtype=torch.float32))
            model.fc1.bias.copy_(torch.from_numpy(msd["b1"]).to(device=device, dtype=torch.float32))
            model.fc2.weight.copy_(torch.from_numpy(msd["W2"].T).to(device=device, dtype=torch.float32))
            model.fc2.bias.copy_(torch.from_numpy(msd["b2"]).to(device=device, dtype=torch.float32))
            model.fc3.weight.copy_(torch.from_numpy(msd["W3"].T).to(device=device, dtype=torch.float32))
            model.fc3.bias.copy_(torch.from_numpy(msd["b3"]).to(device=device, dtype=torch.float32))
            model.fc4.weight.copy_(torch.from_numpy(msd["W4"].T).to(device=device, dtype=torch.float32))
            model.fc4.bias.copy_(torch.from_numpy(msd["b4"]).to(device=device, dtype=torch.float32))
            model.fc5.weight.copy_(torch.from_numpy(msd["W5"].T).to(device=device, dtype=torch.float32))
            model.fc5.bias.copy_(torch.from_numpy(msd["b5"]).to(device=device, dtype=torch.float32))
            model.policy.weight.copy_(torch.from_numpy(msd["Wp"].T).to(device=device, dtype=torch.float32))
            model.policy.bias.copy_(torch.from_numpy(msd["bp"]).to(device=device, dtype=torch.float32))
            model.value.weight.copy_(torch.from_numpy(msd["Wv"].T).to(device=device, dtype=torch.float32))
            model.value.bias.copy_(torch.from_numpy(msd["bv"].reshape(-1)).to(device=device, dtype=torch.float32))
            if "Waux" in msd and msd["Waux"] is not None:
                model.aux.weight.copy_(torch.from_numpy(msd["Waux"].T).to(device=device, dtype=torch.float32))
            if "baux" in msd and msd["baux"] is not None:
                model.aux.bias.copy_(torch.from_numpy(msd["baux"]).to(device=device, dtype=torch.float32))
        print(f"Loaded init checkpoint: {init_checkpoint}")

    model_name = cfg.get("model_name", "sl_v2")
    out_dir = _make_output_dir(args.out_dir, model_name)
    base_lr = float(cfg["learning_rate"])
    momentum = float(cfg["momentum"])
    grad_clip = float(cfg["grad_clip"])
    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    num_epochs = int(cfg["num_epochs"])
    batch_size = int(cfg["batch_size"])
    policy_adv_beta = float(cfg.get("policy_adv_beta", 0.5))
    aux_weight = float(cfg.get("aux_weight", 0.1))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=float(cfg["weight_decay"]),
    )
    best_val = float("inf")

    for epoch in range(num_epochs):
        lr = _get_lr(epoch, base_lr, num_epochs, warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_total_batches = _batch_count(len(train_idx), batch_size)
        val_total_batches = _batch_count(len(val_idx), batch_size)
        print(f"\nEpoch {epoch + 1}/{num_epochs} | lr={lr:.6f} | device={device}")

        model.train()
        train_loss = 0.0
        train_policy_acc = 0.0
        train_value_mse = 0.0
        num_train_samples = 0

        for train_step, (obs, legal_mask, label, value_t) in enumerate(_batch(ds, train_idx, batch_size), start=1):
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
            legal_t = torch.from_numpy(legal_mask).to(device=device, dtype=torch.float32)
            label_t = torch.from_numpy(label).to(device=device, dtype=torch.long)
            value_t_t = torch.from_numpy(value_t).to(device=device, dtype=torch.float32)

            logits_raw, value_pred, aux_logits = model(obs_t)
            logits = logits_raw.masked_fill(legal_t <= 0, -1e9)
            probs = torch.softmax(logits, dim=1)

            with torch.no_grad():
                adv = value_t_t - value_t_t.mean()
                w = torch.exp(torch.clamp(policy_adv_beta * adv, -3.0, 3.0))
                w = w / torch.clamp(w.mean(), min=1e-6)
                rank = torch.clamp((value_t_t * (num_ranks - 1)).long(), 0, num_ranks - 1)

            ce_per = F.cross_entropy(logits, label_t, reduction="none")
            ce_loss = (ce_per * w).mean()
            mse_loss = F.mse_loss(value_pred.reshape(-1), value_t_t)
            aux_loss = F.cross_entropy(aux_logits, rank)
            loss = ce_loss + 0.1 * mse_loss + aux_weight * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += float(loss.item()) * len(label)
            train_value_mse += float(mse_loss.item()) * len(label)
            pred = torch.argmax(logits, dim=1)
            train_policy_acc += int((pred == label_t).sum().item())
            num_train_samples += len(label)
            _print_progress("  train", train_step, train_total_batches)

        train_loss /= max(num_train_samples, 1)
        train_policy_acc /= max(num_train_samples, 1)
        train_value_mse /= max(num_train_samples, 1)

        model.eval()
        val_loss = 0.0
        val_policy_acc = 0.0
        val_value_mse = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for val_step, (obs, legal_mask, label, value_t) in enumerate(_batch(ds, val_idx, batch_size), start=1):
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                legal_t = torch.from_numpy(legal_mask).to(device=device, dtype=torch.float32)
                label_t = torch.from_numpy(label).to(device=device, dtype=torch.long)
                value_t_t = torch.from_numpy(value_t).to(device=device, dtype=torch.float32)

                logits_raw, value_pred, aux_logits = model(obs_t)
                logits = logits_raw.masked_fill(legal_t <= 0, -1e9)
                ce_loss = F.cross_entropy(logits, label_t)
                mse_loss = F.mse_loss(value_pred.reshape(-1), value_t_t)
                rank = torch.clamp((value_t_t * (num_ranks - 1)).long(), 0, num_ranks - 1)
                aux_loss = F.cross_entropy(aux_logits, rank)
                loss = ce_loss + 0.1 * mse_loss + aux_weight * aux_loss

                val_loss += float(loss.item()) * len(label)
                val_value_mse += float(mse_loss.item()) * len(label)
                pred = torch.argmax(logits, dim=1)
                val_policy_acc += int((pred == label_t).sum().item())
                num_val_samples += len(label)
                _print_progress("  valid", val_step, val_total_batches)

        val_loss /= max(num_val_samples, 1)
        val_policy_acc /= max(num_val_samples, 1)
        val_value_mse /= max(num_val_samples, 1)

        ckpt = {
            "model_state_dict": _torch_state_to_numpy_dict(model),
            "action_vocab": actions,
            "config": cfg,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_policy_acc": train_policy_acc,
            "val_policy_acc": val_policy_acc,
            "train_value_mse": train_value_mse,
            "val_value_mse": val_value_mse,
            "view_mode": getattr(args, "view_mode", "all"),
            "view_name": getattr(args, "view_name", ""),
            "num_logs": len(log_paths),
            "backend": "torch",
            "device": device,
        }
        np.savez(out_dir / "last.npz", **ckpt)
        if val_loss <= best_val:
            best_val = val_loss
            np.savez(out_dir / "best.npz", **ckpt)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"policy_acc={val_policy_acc:.4f} value_mse={val_value_mse:.4f} lr={lr:.6f} device={device}"
        )

    metrics = {
        "num_samples": len(ds),
        "best_val_loss": best_val,
        "num_logs": len(log_paths),
        "view_mode": getattr(args, "view_mode", "all"),
        "view_name": getattr(args, "view_name", ""),
        "backend": "torch",
        "device": device,
    }
    metrics["sample_stats"] = summarize_samples(samples)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics

def train(args: argparse.Namespace) -> Dict:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    log_paths = _collect_log_paths(args)
    name_filter = _view_name_filter(args)

    samples = []
    available_names: Set[str] = set()
    skipped_files = []
    for p in log_paths:
        try:
            events = read_mjai_jsonl(p)
            available_names.update(extract_actor_names(events))
            samples.extend(build_supervised_samples(events, actor_name_filter=name_filter))
        except Exception as e:
            skipped_files.append((p, str(e)[:100]))
    
    if skipped_files:
        print(f"跳过 {len(skipped_files)} 个有问题的文件:")
        for p, err in skipped_files[:5]:
            print(f"  {Path(p).name}: {err}")
        if len(skipped_files) > 5:
            print(f"  ... 还有 {len(skipped_files) - 5} 个")
    
    if not samples:
        if name_filter is not None:
            raise RuntimeError(f"no samples for view-name={list(name_filter)[0]}, available={sorted(available_names)}")
        raise RuntimeError("no supervised samples collected")
    errors = replay_validate_label_legal(samples)
    if errors:
        print(f"验证发现 {len(errors)} 个不合法的样本，已移除")
        for err in errors[:3]:
            print(f"  {err[:100]}")
        # 移除不合法的样本
        bad_indices = set()
        for err in errors:
            if ": " in err:
                try:
                    sample_num = int(err.split(":")[0].replace("sample#", ""))
                    bad_indices.add(sample_num)
                except:
                    pass
        samples = [s for i, s in enumerate(samples) if i not in bad_indices]
        print(f"剩余样本数: {len(samples)}")

    actions, stoi = build_action_vocab()
    ds = SupervisedMjaiDataset(samples, stoi)
    n_train = int(len(ds) * cfg["train_split"])
    n_val = len(ds) - n_train
    all_idx = np.arange(len(ds))
    np.random.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:] if n_val > 0 else all_idx[:]

    backend = getattr(args, "backend", "torch")
    if backend == "torch":
        return _train_torch(args, cfg, ds, actions, samples, log_paths, train_idx, val_idx)

    num_ranks = cfg.get("num_ranks", 5)
    model = MultiTaskModel(
        obs_dim=OBS_DIM,
        hidden_dim=cfg["hidden_dim"],
        action_dim=len(actions),
        num_ranks=num_ranks,
    )
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint:
        ckpt = _load_npz_checkpoint(init_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded init checkpoint: {init_checkpoint}")
    best_val = float("inf")
    model_name = cfg.get("model_name", "sl_v2_numpy")
    out_dir = _make_output_dir(args.out_dir, model_name)

    base_lr = float(cfg["learning_rate"])
    momentum = float(cfg["momentum"])
    grad_clip = float(cfg["grad_clip"])
    wd = float(cfg["weight_decay"])
    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    num_epochs = int(cfg["num_epochs"])
    batch_size = int(cfg["batch_size"])
    policy_adv_beta = float(cfg.get("policy_adv_beta", 0.5))
    aux_weight = float(cfg.get("aux_weight", 0.1))

    velocity: Dict[str, np.ndarray] = {}
    for name, param in model.state_dict().items():
        velocity[name] = np.zeros_like(param)

    for epoch in range(num_epochs):
        lr = _get_lr(epoch, base_lr, num_epochs, warmup_epochs)
        train_total_batches = _batch_count(len(train_idx), batch_size)
        val_total_batches = _batch_count(len(val_idx), batch_size)
        print(f"\nEpoch {epoch + 1}/{num_epochs} | lr={lr:.6f}")

        train_loss = 0.0
        train_policy_acc = 0.0
        train_value_mse = 0.0
        num_train_samples = 0

        for train_step, (obs, legal_mask, label, value_t) in enumerate(_batch(ds, train_idx, batch_size), start=1):
            logits_raw, value, aux_logits, h = model.forward_with_cache(obs)
            logits = model.masked_logits(logits_raw, legal_mask)
            probs = _softmax(logits)
            probs_aux = _softmax(aux_logits)

            grad_logits, ce_loss = _weighted_policy_grad(probs, label, legal_mask, value_t, policy_adv_beta)
            grad_value = ((value.reshape(-1) - value_t) / max(len(label), 1)).reshape(-1, 1)

            rank = _compute_rank_proxy(value_t, num_ranks)
            rank_onehot = _onehot_rank(rank, num_ranks)
            grad_aux = (probs_aux - rank_onehot) / max(len(label), 1)

            rank = _compute_rank_proxy(value_t, num_ranks)
            rank_onehot = _onehot_rank(rank, num_ranks)

            # ---- backbone gradients ----
            # grad_h has shape (batch, hidden_dim)
            grad_h = grad_logits @ model.Wp.T + grad_value @ model.Wv.T + grad_aux @ model.Waux.T

            # Block 2: h -> relu -> W4 -> relu -> W5 -> (+h) -> relu -> output
            # grad passed to Block2 output: grad_h (masked by relu)
            grad_h_after_relu2 = grad_h.copy()
            grad_h_after_relu2[h <= 0] = 0.0

            # h5 = relu(h + h @ W4 @ W5) -- h_shortcut2 = h, h4 = relu(h @ W4), h5_lin = h4 @ W5
            # d/dh (h + h5_lin) = grad_h_after_relu2
            # grad into shortcut (h): grad_h_after_relu2
            # grad into h5_lin: grad_h_after_relu2
            grad_h5_lin = grad_h_after_relu2.copy()
            grad_h4 = grad_h5_lin @ model.encoder.W5.T
            grad_h4[h <= 0] = 0.0
            grad_h4_and_shortcut = grad_h4 + grad_h_after_relu2
            grad_h4_and_shortcut[h <= 0] = 0.0
            grad_h3 = grad_h4_and_shortcut @ model.encoder.W3.T
            grad_h3[h <= 0] = 0.0
            grad_h3_and_shortcut = grad_h3 + grad_h4_and_shortcut
            grad_h3_and_shortcut[h <= 0] = 0.0
            grad_h2 = grad_h3_and_shortcut @ model.encoder.W2.T
            grad_h2[h <= 0] = 0.0
            grad_h1 = grad_h2 + grad_h3_and_shortcut
            grad_h1[h <= 0] = 0.0
            grad_h0 = grad_h1 @ model.encoder.W1.T

            grad_encoder = {
                "W1": obs.T @ grad_h1 + wd * model.encoder.W1,
                "b1": grad_h1.sum(axis=0),
                "W2": h.T @ grad_h2 + wd * model.encoder.W2,
                "b2": grad_h2.sum(axis=0),
                "W3": h.T @ grad_h3 + wd * model.encoder.W3,
                "b3": grad_h3.sum(axis=0),
                "W4": h.T @ grad_h4 + wd * model.encoder.W4,
                "b4": grad_h4.sum(axis=0),
                "W5": h.T @ grad_h5_lin + wd * model.encoder.W5,
                "b5": grad_h5_lin.sum(axis=0),
            }

            grads = {
                **grad_encoder,
                "Wp": h.T @ grad_logits + wd * model.Wp,
                "bp": grad_logits.sum(axis=0),
                "Wv": h.T @ grad_value + wd * model.Wv,
                "bv": grad_value.sum(axis=0),
                "Waux": h.T @ grad_aux + wd * model.Waux,
                "baux": grad_aux.sum(axis=0),
            }

            grads = _clip_grad_norm(grads, grad_clip)

            for name, grad in grads.items():
                velocity[name] = momentum * velocity[name] - lr * grad
                if name in {"W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"}:
                    setattr(model.encoder, name, getattr(model.encoder, name) + velocity[name])
                elif name in {"Wp", "bp", "Wv", "bv", "Waux", "baux"}:
                    setattr(model, name, getattr(model, name) + velocity[name])

            mse_loss = float(((value - value_t) ** 2).mean())
            aux_loss = float(-np.sum(rank_onehot * np.log(np.clip(probs_aux, 1e-9, None))) / max(len(label), 1))
            batch_loss = ce_loss + 0.1 * mse_loss + aux_weight * aux_loss

            train_loss += batch_loss * len(label)
            train_value_mse += mse_loss * len(label)
            pred_labels = logits.argmax(axis=1)
            train_policy_acc += int((pred_labels == label).sum())
            num_train_samples += len(label)
            _print_progress("  train", train_step, train_total_batches)

        train_loss /= max(num_train_samples, 1)
        train_policy_acc /= max(num_train_samples, 1)
        train_value_mse /= max(num_train_samples, 1)

        val_loss = 0.0
        val_policy_acc = 0.0
        val_value_mse = 0.0
        num_val_samples = 0

        for val_step, (obs, legal_mask, label, value_t) in enumerate(_batch(ds, val_idx, batch_size), start=1):
            logits_out, value, aux_logits = model.forward(obs)
            logits = model.masked_logits(logits_out, legal_mask)
            probs = _softmax(logits)
            probs_aux = _softmax(aux_logits)

            ce = -np.log(np.clip(probs[np.arange(len(label)), label], 1e-9, None))
            ce_loss = float(ce.mean())
            mse_loss = float(((value - value_t) ** 2).mean())

            rank = _compute_rank_proxy(value_t, num_ranks)
            rank_onehot = _onehot_rank(rank, num_ranks)
            aux_loss = float(-np.sum(rank_onehot * np.log(np.clip(probs_aux, 1e-9, None))) / max(len(label), 1))

            batch_loss = ce_loss + 0.1 * mse_loss + aux_weight * aux_loss

            val_loss += batch_loss * len(label)
            val_value_mse += mse_loss * len(label)
            pred_labels = logits.argmax(axis=1)
            val_policy_acc += int((pred_labels == label).sum())
            num_val_samples += len(label)
            _print_progress("  valid", val_step, val_total_batches)

        val_loss /= max(num_val_samples, 1)
        val_policy_acc /= max(num_val_samples, 1)
        val_value_mse /= max(num_val_samples, 1)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "action_vocab": actions,
            "config": cfg,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_policy_acc": train_policy_acc,
            "val_policy_acc": val_policy_acc,
            "train_value_mse": train_value_mse,
            "val_value_mse": val_value_mse,
            "view_mode": getattr(args, "view_mode", "all"),
            "view_name": getattr(args, "view_name", ""),
            "num_logs": len(log_paths),
        }
        np.savez(out_dir / "last.npz", **ckpt)
        if val_loss <= best_val:
            best_val = val_loss
            np.savez(out_dir / "best.npz", **ckpt)

        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} policy_acc={val_policy_acc:.4f} value_mse={val_value_mse:.4f} lr={lr:.6f}")

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
    parser.add_argument("--view-mode", choices=["all", "single", "me"], default="all")
    parser.add_argument("--view-name", default="keqing1")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--out-dir", default="artifacts/models")
    parser.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--init-checkpoint", default=None, help="Path to existing .npz checkpoint for continued training.")
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
