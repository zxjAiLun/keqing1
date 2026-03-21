"""
训练脚本 for ModelV4 (增强版)

参考 train_v2.py 的结构，支持:
- CUDA 训练
- 动态输出目录 (model_name + timestamp)
- 继续训练 (checkpoint)
- 使用 converted/ds* 数据集
- 训练可视化 (loss/accuracy 曲线)
- 四家视角训练 (view-mode)

view-mode 选项:
- all: 默认，收集所有玩家的样本
- all4: 分别以四家视角收集样本，每局游戏产生4份样本
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mahjong_env.replay import (
    ReplaySample,
    build_supervised_samples,
    extract_actor_names,
    read_mjai_jsonl,
)
from model.vocab import build_action_vocab
import riichienv.convert as cvt

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，训练可视化将不可用")


# ============================================================================
# 常量
# ============================================================================

# 动作数量会在训练时根据实际动作种类动态确定
NUM_ACTIONS = 38  # 默认值，会在训练时更新

# 37 牌类型
N_TILE_TYPES = 37

# 特征维度
TILE_PLANE_DIM = 32
SCALAR_DIM = 24


# ============================================================================
# 数据集
# ============================================================================

class V4SupervisedDataset(Dataset):
    """
    用于 ModelV4 的监督学习数据集

    返回:
        tile_features: (37, tile_plane_dim)
        scalar_features: (scalar_dim,)
        legal_mask: (num_actions,)
        action: int
        value: float
    """

    def __init__(
        self,
        samples: List[ReplaySample],
        action_stoi: Dict[str, int],
        tile_plane_dim: int = TILE_PLANE_DIM,
        scalar_dim: int = SCALAR_DIM,
    ):
        self.samples = samples
        self.action_stoi = action_stoi
        self.tile_plane_dim = tile_plane_dim
        self.scalar_dim = scalar_dim
        self.num_actions = len(action_stoi)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        s = self.samples[idx]

        # 编码状态
        tile_features, scalar_features = vectorize_state_v4(s.state, s.actor)

        # Legal mask
        legal_mask = np.zeros(self.num_actions, dtype=np.float32)
        for a in s.legal_actions:
            action_name = action_to_token_name(a)
            if action_name in self.action_stoi:
                legal_mask[self.action_stoi[action_name]] = 1.0

        # Action label
        label_name = action_to_token_name(s.label_action)
        action_idx = self.action_stoi.get(label_name, 0)

        # Value target
        value = float(s.value_target)

        return (
            tile_features.astype(np.float32),
            scalar_features.astype(np.float32),
            legal_mask,
            action_idx,
            value,
        )


def collate_fn(batch):
    """DataLoader 的批处理函数"""
    tile_features = np.stack([b[0] for b in batch], axis=0)
    scalar_features = np.stack([b[1] for b in batch], axis=0)
    legal_masks = np.stack([b[2] for b in batch], axis=0)
    actions = np.array([b[3] for b in batch], dtype=np.int64)
    values = np.array([b[4] for b in batch], dtype=np.float32)
    # 转换为 torch tensor
    tile_features = torch.from_numpy(tile_features)
    scalar_features = torch.from_numpy(scalar_features)
    legal_masks = torch.from_numpy(legal_masks)
    actions = torch.from_numpy(actions)
    values = torch.from_numpy(values)
    return tile_features, scalar_features, legal_masks, actions, values


# ============================================================================
# 特征编码 (与 FeatureEncoder 对应)
# ============================================================================

def tile_to_type(tile_id: int) -> int:
    """
    将 tile id (0-135) 映射到牌类型索引 (0-36)

    37 牌类型:
    - 0-8:   1-9m (万子)
    - 9-17:  1-9p (筒子)
    - 18-26: 1-9s (索子)
    - 27-33: E/S/W/N/P/F/C (字牌)
    - 34:    aka5 5m
    - 35:    aka5 5p
    - 36:    aka5 5s

    riichienv tile id 分布:
    - 万子: 0-35 (1-9m * 4)
    - 筒子: 36-71 (1-9p * 4)
    - 索子: 72-107 (1-9s * 4)
    - 字牌: 108-135 (7种字牌 * 4)

    aka5 tile id: 17(5m), 53(5p), 89(5s)
    """
    # aka5 先检查
    if tile_id == 17:
        return 34  # aka5 5m
    if tile_id == 53:
        return 35  # aka5 5p
    if tile_id == 89:
        return 36  # aka5 5s

    # 标准牌: tile_type = tile_id // 4
    return tile_id // 4


def parse_tile(tile_str: str) -> int:
    """
    将 tile string 转换为 tile id

    使用 riichienv.convert.mpsz_to_tid
    - 数牌: "1m", "5p", "9s" 等
    - 字牌: "1z", "2z", ... "7z" (不是 "E", "N" 等)
    """
    if not tile_str or tile_str == "?":
        return -1

    # 字牌映射 (E->1z, S->2z, etc.)
    zi_map = {"E": "1z", "S": "2z", "W": "3z", "N": "4z", "P": "5z", "F": "6z", "C": "7z"}
    if tile_str in zi_map:
        tile_str = zi_map[tile_str]

    try:
        return cvt.mpsz_to_tid(tile_str)
    except Exception:
        return -1


def vectorize_state_v4(state: Dict, actor: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 state dict 编码为 ModelV4 的输入特征

    Args:
        state: snapshot() 返回的字典
        actor: 当前玩家索引

    Returns:
        tile_features: (37, tile_plane_dim)
        scalar_features: (scalar_dim,)
    """
    tile_features = np.zeros((N_TILE_TYPES, TILE_PLANE_DIM), dtype=np.float32)
    scalar_features = np.zeros(SCALAR_DIM, dtype=np.float32)

    # =====================================================================
    # 牌类型平面特征
    # =====================================================================

    # 通道 0: 自己的手牌
    hand_tiles = state.get("hand", [])
    for tile_str in hand_tiles:
        tile_id = parse_tile(tile_str)
        if tile_id >= 0:
            tile_type = tile_to_type(tile_id)
            if tile_type < N_TILE_TYPES:
                tile_features[tile_type, 0] = 1.0

    # 通道 1: 宝牌指示器
    dora_markers = state.get("dora_markers", [])
    for tile_str in dora_markers:
        tile_id = parse_tile(tile_str)
        if tile_id >= 0:
            tile_type = tile_to_type(tile_id)
            if tile_type < N_TILE_TYPES:
                tile_features[tile_type, 1] = 1.0

    # 通道 2-5: 各玩家舍牌
    discards = state.get("discards", [])
    for pid in range(min(4, len(discards))):
        channel = 2 + pid
        tile_type_counts = Counter()
        for tile_str in discards[pid]:
            tile_id = parse_tile(tile_str)
            if tile_id >= 0:
                tile_type = tile_to_type(tile_id)
                if tile_type < N_TILE_TYPES:
                    tile_type_counts[tile_type] += 1
        # 归一化
        max_count = max(tile_type_counts.values()) if tile_type_counts else 1
        for tile_type, count in tile_type_counts.items():
            tile_features[tile_type, channel] = count / max_count

    # =====================================================================
    # 标量特征
    # =====================================================================

    scalar_idx = 0

    # 0: 亲家标志
    oya = state.get("oya", 0)
    scalar_features[scalar_idx] = 1.0 if oya == actor else 0.0
    scalar_idx += 1

    # 1: 场风 (0-3)
    bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
    bakaze = state.get("bakaze", "E")
    scalar_features[scalar_idx] = bakaze_map.get(bakaze, 0) / 3.0
    scalar_idx += 1

    # 2: 本场
    honba = state.get("honba", 0)
    scalar_features[scalar_idx] = min(honba / 10.0, 1.0)
    scalar_idx += 1

    # 3: 立直棒
    kyotaku = state.get("kyotaku", 0)
    scalar_features[scalar_idx] = min(kyotaku / 5.0, 1.0)
    scalar_idx += 1

    # 4-7: 各玩家副露次数 (0=门清)
    melds = state.get("melds", [])
    for pid in range(4):
        n_melds = len(melds[pid]) if pid < len(melds) else 0
        scalar_features[scalar_idx + pid] = min(n_melds / 4.0, 1.0)
    scalar_idx += 4

    # 8-11: 各玩家巡目 (通过舍牌数估算)
    for pid in range(4):
        n_discards = len(discards[pid]) if pid < len(discards) else 0
        scalar_features[scalar_idx + pid] = min(n_discards / 20.0, 1.0)
    scalar_idx += 4

    # 12: 与其他三家分数差 (取最大差值)
    scores = state.get("scores", [25000, 25000, 25000, 25000])
    my_score = scores[actor] if actor < len(scores) else 25000
    max_diff = 0
    for i, score in enumerate(scores):
        if i != actor:
            diff = abs(score - my_score) / 50000.0
            max_diff = max(max_diff, diff)
    scalar_features[scalar_idx] = min(max_diff, 1.0)
    scalar_idx += 1

    # 13: 向听数 (使用 state 中的 shanten 或估算)
    shanten = state.get("shanten", 4)
    scalar_features[scalar_idx] = min(shanten / 8.0, 1.0)
    scalar_idx += 1

    # 14: 是否听牌
    is_tenpai = 1.0 if shanten <= 0 else 0.0
    scalar_features[scalar_idx] = is_tenpai
    scalar_idx += 1

    # 15: 有效进张数 (使用 waits_count 或估算)
    waits_count = state.get("waits_count", 0)
    scalar_features[scalar_idx] = waits_count / 34.0
    scalar_idx += 1

    # 16-19: 预留
    # ...

    return tile_features, scalar_features


def action_to_token_name(action: Dict) -> str:
    """将 action dict 转换为 token name"""
    if not action:
        return "none"

    action_type = action.get("type", "")
    if action_type == "dahai":
        tile = action.get("pai", "").replace(" ", "")  # 去除空格
        return f"dahai_{tile}"
    elif action_type in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        return action_type
    elif action_type == "reach":
        return "reach"
    elif action_type == "hora":
        return "hora"
    elif action_type == "ryukyoku":
        return "ryukyoku"
    elif action_type == "none":
        return "none"
    else:
        return action_type


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_progress(prefix: str, step: int, total: int) -> None:
    """打印进度条"""
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
    """创建带时间戳的输出目录"""
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


def _load_checkpoint(path: str, model: nn.Module, device: str) -> bool:
    """加载检查点

    Returns:
        True if loaded successfully, False otherwise
    """
    # 首先检查文件是否存在
    import os
    if not os.path.exists(path):
        print(f"错误: 检查点文件不存在: {path}")
        return False

    try:
        ckpt = torch.load(path, map_location=device)

        # 检查必要字段
        if "model_state_dict" not in ckpt:
            print(f"错误: 检查点缺少 'model_state_dict' 字段")
            return False

        model.load_state_dict(ckpt["model_state_dict"])
        print(f"已加载检查点: {path}")
        print(f"  - 包含字段: {list(ckpt.keys())}")
        return True
    except Exception as e:
        print(f"错误: 无法加载检查点 {path}: {e}")
        return False


def _plot_training_history(history: Dict, out_dir: Path) -> None:
    """绘制并保存训练曲线"""
    if not HAS_MATPLOTLIB:
        return

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate 曲线
    axes[2].plot(epochs, history["lr"], label="LR", marker="o", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 保存历史数据为 CSV
    import csv
    csv_path = out_dir / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        for i in range(len(epochs)):
            row = {k: v[i] for k, v in history.items()}
            writer.writerow(row)
    print(f"训练历史已保存到: {csv_path}")


def _collect_log_paths(args: argparse.Namespace) -> List[str]:
    """收集要训练的数据文件"""
    pre_converted_dir = getattr(args, "pre_converted_dir", None)

    if pre_converted_dir:
        pre_path = Path(pre_converted_dir)
        if not pre_path.exists():
            raise RuntimeError(f"pre-converted-dir 不存在: {pre_converted_dir}")

        # 支持目录或单个文件
        if pre_path.is_file():
            return [str(pre_path)]
        else:
            # 递归搜索所有 jsonl 和 mjson 文件
            jsonl_files = sorted(pre_path.glob("**/*.jsonl"))
            mjson_files = sorted(pre_path.glob("**/*.mjson"))
            all_files = jsonl_files + mjson_files
            if not all_files:
                raise RuntimeError(f"pre-converted-dir 中没有找到 .jsonl 或 .mjson 文件: {pre_converted_dir}")
            print(f"从已转换目录加载数据: {pre_converted_dir} ({len(all_files)} 个文件, jsonl={len(jsonl_files)}, mjson={len(mjson_files)})")
            return [str(f) for f in all_files]

    raise RuntimeError("必须指定 --pre-converted-dir")


# ============================================================================
# 训练函数
# ============================================================================

def _fix_mjai_events(events):
    """修复 mjai events 中的一些格式问题"""
    fixed = []
    for event in events:
        # 修复 ankan 缺少 pai 的问题 - pai 应该等于 consumed 的第一张
        if event.get("type") == "ankan" and "pai" not in event:
            consumed = event.get("consumed", [])
            if consumed:
                event = dict(event)
                event["pai"] = consumed[0]
        fixed.append(event)
    return fixed


def train_torch(
    args: argparse.Namespace,
    cfg: Dict,
    log_paths: List[str],
) -> None:
    """PyTorch 训练"""
    device = args.device if hasattr(args, 'device') else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用设备: {device}")

    # 1. 收集数据
    all_samples = []
    actions_set = set()
    total_files = len(log_paths)

    # view-mode: all, all4
    view_mode = getattr(args, "view_mode", "all")
    print(f"视角模式: {view_mode}")

    print(f"加载数据: {total_files} 个文件")
    for i, log_path in enumerate(log_paths):
        try:
            events = read_mjai_jsonl(log_path)
            events = _fix_mjai_events(events)

            if view_mode == "all":
                # 默认模式: 收集所有玩家的样本
                samples = build_supervised_samples(events, actor_name_filter=None)
                all_samples.extend(samples)
            elif view_mode == "all4":
                # 四家视角模式: 分别收集四家视角的样本
                # 注意: 这里使用 actor_filter={0,1,2,3} 来确保收集所有玩家的样本
                # build_supervised_samples 会对每个有手牌可见的玩家创建样本
                samples = build_supervised_samples(events, actor_filter={0, 1, 2, 3})
                all_samples.extend(samples)
            else:
                raise ValueError(f"未知的 view-mode: {view_mode}")

            for s in samples:
                action_name = action_to_token_name(s.label_action)
                actions_set.add(action_name)

        except Exception as e:
            print(f"  加载失败: {e}")

        _print_progress("  loading", i + 1, total_files)

    _print_progress("  loading", total_files, total_files)

    # 统计各视角样本数
    actor_counts = Counter(s.actor for s in all_samples)
    print(f"\n总样本数: {len(all_samples)}")
    print(f"各视角样本数: {dict(sorted(actor_counts.items()))}")
    print(f"动作种类: {len(actions_set)}")

    # 2. 创建动作表
    actions = sorted(actions_set)
    action_stoi = {a: i for i, a in enumerate(actions)}
    print(f"动作列表: {actions}")

    # 3. 创建数据集
    dataset = V4SupervisedDataset(
        all_samples,
        action_stoi,
        tile_plane_dim=cfg.get("tile_plane_dim", TILE_PLANE_DIM),
        scalar_dim=cfg.get("scalar_dim", SCALAR_DIM),
    )

    # 4. 划分训练/验证集
    train_split = cfg.get("train_split", 0.9)
    indices = np.random.permutation(len(dataset))
    train_size = int(len(indices) * train_split)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    # 5. 创建模型
    from v4model import ModelV4

    model = ModelV4(
        tile_plane_dim=cfg.get("tile_plane_dim", TILE_PLANE_DIM),
        scalar_dim=cfg.get("scalar_dim", SCALAR_DIM),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_actions=len(actions),
        num_res_blocks=cfg.get("num_res_blocks", 3),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 加载检查点 (继续训练)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        if not _load_checkpoint(checkpoint, model, device):
            raise RuntimeError(f"指定的 --checkpoint 无法加载: {checkpoint}")

    # 7. 创建输出目录
    model_name = cfg.get("model_name", "modelv4")
    # 如果使用 all4 视角模式，在目录名中标注
    if view_mode == "all4":
        model_name = f"{model_name}_all4"
    out_dir = _make_output_dir(args.out_dir, model_name)

    # 8. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # 9. 学习率调度
    num_epochs = cfg.get("num_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # 10. 训练循环
    batch_size = cfg.get("batch_size", 256)
    grad_clip = cfg.get("grad_clip", 1.0)
    best_val_loss = float("inf")

    # 训练历史记录
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 使用Subset实现训练集采样
        from torch.utils.data import Subset
        train_subset = Subset(dataset, train_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=0,
        )

        train_total_batches = len(train_loader)
        for batch_idx, (tile_feat, scalar_feat, legal_mask, actions_batch, values) in enumerate(train_loader):
            tile_feat = tile_feat.to(device)
            scalar_feat = scalar_feat.to(device)
            legal_mask = legal_mask.to(device)
            actions_batch = actions_batch.to(device)

            optimizer.zero_grad()

            policy_logits, value_pred = model(tile_feat, scalar_feat)

            # 应用 legal mask
            policy_logits = policy_logits.masked_fill(legal_mask <= 0, -1e9)

            # Policy loss
            policy_loss = F.cross_entropy(policy_logits, actions_batch)

            loss = policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            preds = policy_logits.argmax(dim=1)
            train_correct += (preds == actions_batch).sum().item()
            train_total += actions_batch.size(0)

        _print_progress("  train", train_total_batches, train_total_batches)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=0,
            )

            val_total_batches = len(val_loader)
            for tile_feat, scalar_feat, legal_mask, actions_batch, values in val_loader:
                tile_feat = tile_feat.to(device)
                scalar_feat = scalar_feat.to(device)
                legal_mask = legal_mask.to(device)
                actions_batch = actions_batch.to(device)

                policy_logits, value_pred = model(tile_feat, scalar_feat)
                policy_logits = policy_logits.masked_fill(legal_mask <= 0, -1e9)

                loss = F.cross_entropy(policy_logits, actions_batch)

                val_loss += loss.item()
                preds = policy_logits.argmax(dim=1)
                val_correct += (preds == actions_batch).sum().item()
                val_total += actions_batch.size(0)

        _print_progress("  valid", val_total_batches, val_total_batches)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # 记录历史
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # 保存检查点
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "actions": actions,
        }
        torch.save(ckpt, out_dir / "last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

    print(f"\n训练完成! 输出目录: {out_dir}")

    # 保存训练历史
    if HAS_MATPLOTLIB:
        _plot_training_history(history, out_dir)
        print(f"训练曲线已保存到: {out_dir / 'training_curve.png'}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="训练 ModelV4 (增强版)")
    parser.add_argument("--config", type=str, default="configs/v4.yaml", help="配置文件路径")
    parser.add_argument("--pre-converted-dir", type=str, required=True, help="已转换的数据目录")
    parser.add_argument("--out-dir", type=str, default="artifacts/models", help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="初始检查点 (用于继续训练)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--view-mode",
        type=str,
        default="all",
        choices=["all", "all4"],
        help="视角模式: all=收集所有玩家样本, all4=分别以四家视角收集样本",
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"配置文件不存在: {args.config}, 使用默认配置")
        cfg = {
            "tile_plane_dim": TILE_PLANE_DIM,
            "scalar_dim": SCALAR_DIM,
            "hidden_dim": 256,
            "num_res_blocks": 3,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "num_epochs": 10,
            "grad_clip": 1.0,
            "train_split": 0.9,
            "model_name": "modelv4",
        }

    # 收集数据
    log_paths = _collect_log_paths(args)

    # 开始训练
    train_torch(args, cfg, log_paths)


if __name__ == "__main__":
    main()
