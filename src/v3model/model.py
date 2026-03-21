"""
增强型模型架构

本模块实现高可解释性的多任务模型，支持向听数预测和立直决策。

核心设计原则：
1. 可解释性优先 - 每个输出都有明确语义
2. 多任务学习 - Policy, Value, Shanten, Riichi四个输出头
3. 可分离损失函数 - 便于单独调试
4. 模块化设计 - 便于扩展新功能

Example:
    >>> from model_enhanced import EnhancedModel
    >>> model = EnhancedModel()
    >>> features = torch.randn(1, 13)  # features_enhanced的13维特征
    >>> outputs = model(features, return_all_heads=True)
    >>> print(outputs['shanten'])  # 预测向听数
    >>> print(outputs['riichi'])  # 立直概率
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path


class ResNetEncoder(nn.Module):
    """
    ResNet编码器
    
    使用残差连接的特征编码器，将输入特征映射到256维隐藏空间
    
    Architecture:
        Input (13) -> Linear -> ReLU -> 
        [Linear -> ReLU -> Linear -> ReLU] * num_layers ->
        Output (256)
    
    Attributes:
        input_dim: 输入特征维度（默认13，对应features_enhanced）
        hidden_dim: 隐藏层维度（默认256）
        num_layers: 残差块数量（默认5）
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def _make_residual_block(self, hidden_dim: int) -> nn.Module:
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, input_dim)
        
        Returns:
            编码后的特征 (batch_size, hidden_dim)
        """
        # 输入投影
        h = self.input_layer(x)
        
        # 残差连接
        for block in self.residual_blocks:
            h = h + block(h)
        
        # 输出投影
        h = self.output_layer(h)
        
        return h


class PolicyHead(nn.Module):
    """
    策略预测头
    
    输出动作概率分布，用于选择最优动作
    
    Architecture:
        Input (256) -> Linear(256->256) -> ReLU -> Linear(256->num_actions)
    
    Attributes:
        hidden_dim: 输入维度（默认256）
        num_actions: 动作数量（默认200）
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_actions: int = 200
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 编码器输出 (batch_size, 256)
        
        Returns:
            动作logits (batch_size, num_actions)
        """
        return self.net(h)


class ValueHead(nn.Module):
    """
    价值评估头
    
    输出局面价值评估，范围[-1, 1]
    
    Architecture:
        Input (256) -> Linear(256->128) -> ReLU -> Linear(128->1) -> Tanh
    
    Attributes:
        hidden_dim: 输入维度（默认256）
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 编码器输出 (batch_size, 256)
        
        Returns:
            局面价值 (batch_size, 1)，范围 [-1, 1]
        """
        return self.net(h)


class ShantenHead(nn.Module):
    """
    向听数预测头（可解释性）
    
    输出预测的向听数，用于验证模型是否理解手牌质量
    
    Architecture:
        Input (256) -> Linear(256->128) -> ReLU -> Linear(128->1)
    
    Attributes:
        hidden_dim: 输入维度（默认256）
    
    Note:
        向听数范围：-1（和了）到8+（多向听）
        输出是连续值，便于回归训练
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 编码器输出 (batch_size, 256)
        
        Returns:
            预测向听数 (batch_size, 1)
            范围：-1（和了）到8+（多向听）
        """
        return self.net(h)


class RiichiHead(nn.Module):
    """
    立直决策头（可解释性）
    
    输出立直概率，用于判断是否应该立直
    
    Architecture:
        Input (256) -> Linear(256->128) -> ReLU -> Linear(128->1) -> Sigmoid
    
    Attributes:
        hidden_dim: 输入维度（默认256）
    
    Note:
        输出范围：[0, 1]，表示立直概率
        可以直接判断模型的立直倾向
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 编码器输出 (batch_size, 256)
        
        Returns:
            立直概率 (batch_size, 1)，范围 [0, 1]
        """
        return self.net(h)


class EnhancedModel(nn.Module):
    """
    增强型模型
    
    支持多任务输出，每个输出都有明确的语义含义
    
    Architecture:
        Input Features (13) -> ResNetEncoder -> Shared Representation (256)
                                                      |
                              +--------------------+--------------------+
                              |                    |                    |
                              v                    v                    v
                         Policy Head         Value Head         Shanten Head
                         (200 actions)       (value)            (shanten)
                                                               |
                                                               v
                                                          Riichi Head
                                                          (riichi prob)
    
    Attributes:
        config: 模型配置字典
    
    Example:
        >>> model = EnhancedModel()
        >>> features = torch.randn(4, 13)  # batch_size=4
        >>> 
        >>> # 获取策略输出（用于动作选择）
        >>> policy_logits = model(features)
        >>> 
        >>> # 获取所有输出（用于训练和分析）
        >>> outputs = model(features, return_all_heads=True)
        >>> print(f"向听数: {outputs['shanten']}")
        >>> print(f"立直概率: {outputs['riichi']}")
        >>> print(f"局面价值: {outputs['value']}")
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # 配置
        self.config = config or {}
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.num_layers = self.config.get('num_layers', 5)
        self.num_actions = self.config.get('num_actions', 200)
        
        # 编码器
        self.encoder = ResNetEncoder(
            input_dim=13,  # features_enhanced的维度
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # 输出头
        self.policy_head = PolicyHead(
            hidden_dim=self.hidden_dim,
            num_actions=self.num_actions
        )
        self.value_head = ValueHead(hidden_dim=self.hidden_dim)
        self.shanten_head = ShantenHead(hidden_dim=self.hidden_dim)
        self.riichi_head = RiichiHead(hidden_dim=self.hidden_dim)
        
        # 版本信息
        self.version = "1.0.0"
    
    def forward(
        self,
        features: torch.Tensor,
        return_all_heads: bool = False
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, 13)
                对应features_enhanced的13维特征向量
            return_all_heads: 是否返回所有Head的输出
        
        Returns:
            如果return_all_heads=False:
                policy_logits: 策略输出 (batch_size, num_actions)
            如果return_all_heads=True:
                包含所有输出的字典:
                    - policy_logits: 策略logits
                    - policy_probs: 策略概率分布
                    - value: 局面价值 [-1, 1]
                    - shanten: 预测向听数
                    - riichi: 立直概率 [0, 1]
                    - all_heads: 所有Head输出的列表
        
        Example:
            >>> features = torch.randn(4, 13)
            >>> 
            >>> # 快速获取策略输出
            >>> logits = model(features)
            >>> 
            >>> # 获取所有输出
            >>> outputs = model(features, return_all_heads=True)
            >>> print(outputs['shanten'])  # 预测向听数
        """
        # 编码
        h = self.encoder(features)
        
        # 各Head输出
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        shanten = self.shanten_head(h)
        riichi = self.riichi_head(h)
        
        if return_all_heads:
            return {
                'policy_logits': policy_logits,
                'policy_probs': F.softmax(policy_logits, dim=-1),
                'value': value,
                'shanten': shanten,
                'riichi': riichi,
                'all_heads': [policy_logits, value, shanten, riichi]
            }
        
        return policy_logits
    
    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算损失
        
        Args:
            batch: 包含以下键的字典
                - features: 特征向量 (batch_size, 13)
                - actions: 动作标签 (batch_size,)
                - rewards: 奖励 (batch_size, 1)
                - true_shanten: 真实向听数 (batch_size, 1)
                - should_riichi: 是否应该立直 (batch_size, 1)
        
        Returns:
            total_loss: 总损失
            loss_dict: 各损失的字典（便于调试）
        
        Example:
            >>> batch = {
            ...     'features': torch.randn(4, 13),
            ...     'actions': torch.randint(0, 200, (4,)),
            ...     'rewards': torch.randn(4, 1),
            ...     'true_shanten': torch.tensor([[0], [1], [-1], [2]]),
            ...     'should_riichi': torch.tensor([[1], [0], [0], [1]]).float()
            ... }
            >>> total, losses = model.compute_losses(batch)
            >>> print(f"总损失: {total.item():.4f}")
            >>> print(f"策略损失: {losses['policy'].item():.4f}")
            >>> print(f"向听损失: {losses['shanten'].item():.4f}")
        """
        # 前向传播
        outputs = self.forward(batch['features'], return_all_heads=True)
        
        # 计算各损失
        p_loss = policy_loss(outputs['policy_logits'], batch['actions'])
        v_loss = value_loss(outputs['value'], batch['rewards'])
        s_loss = shanten_loss(outputs['shanten'], batch['true_shanten'])
        r_loss = riichi_loss(outputs['riichi'], batch['should_riichi'])
        
        # 组合损失
        total, loss_dict = total_loss(p_loss, v_loss, s_loss, r_loss)
        
        return total, loss_dict
    
    def save(self, path: str | Path) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        
        Example:
            >>> model.save('checkpoints/model_v1.pth')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'version': self.version
        }, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'EnhancedModel':
        """
        加载模型
        
        Args:
            path: 模型路径
        
        Returns:
            加载的模型实例
        
        Example:
            >>> model = EnhancedModel.load('checkpoints/model_v1.pth')
        """
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# ============== 损失函数 ==============

def policy_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    策略损失：交叉熵
    
    Args:
        logits: 策略logits (batch_size, num_actions)
        actions: 动作标签 (batch_size,)
    
    Returns:
        交叉熵损失
    """
    return F.cross_entropy(logits, actions)


def value_loss(values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    价值损失：均方误差
    
    Args:
        values: 预测价值 (batch_size,) or (batch_size, 1)
        rewards: 真实奖励 (batch_size,) or (batch_size, 1)
    
    Returns:
        均方误差损失
    """
    # 确保形状一致
    if values.dim() > 1:
        values = values.squeeze(-1)
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    return F.mse_loss(values, rewards)


def shanten_loss(
    pred_shanten: torch.Tensor,
    true_shanten: torch.Tensor
) -> torch.Tensor:
    """
    向听损失：均方误差
    
    关键用途：
    - 直接反映模型对向听数的理解
    - 损失值高说明模型不会判断手牌质量
    - 便于单独调试和可视化
    
    Args:
        pred_shanten: 预测向听数 (batch_size,) or (batch_size, 1)
        true_shanten: 真实向听数 (batch_size,) or (batch_size, 1)
    
    Returns:
        均方误差损失
    """
    # 确保形状一致
    if pred_shanten.dim() > 1:
        pred_shanten = pred_shanten.squeeze(-1)
    if true_shanten.dim() > 1:
        true_shanten = true_shanten.squeeze(-1)
    return F.mse_loss(pred_shanten, true_shanten)


def riichi_loss(
    pred_riichi: torch.Tensor,
    should_riichi: torch.Tensor
) -> torch.Tensor:
    """
    立直损失：二分类交叉熵
    
    关键用途：
    - 直接反映模型的立直决策
    - 可以对比预测概率与实际决策
    - 便于理解模型的立直时机判断
    
    Args:
        pred_riichi: 预测立直概率 (batch_size,) or (batch_size, 1)
        should_riichi: 是否应该立直 (batch_size,) or (batch_size, 1)
    
    Returns:
        二分类交叉熵损失
    """
    # 确保形状一致
    if pred_riichi.dim() > 1:
        pred_riichi = pred_riichi.squeeze(-1)
    if should_riichi.dim() > 1:
        should_riichi = should_riichi.squeeze(-1)
    return F.binary_cross_entropy(pred_riichi, should_riichi)


def total_loss(
    policy_loss_val: torch.Tensor,
    value_loss_val: torch.Tensor,
    shanten_loss_val: torch.Tensor,
    riichi_loss_val: torch.Tensor,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    组合损失函数
    
    Args:
        policy_loss_val: 策略损失
        value_loss_val: 价值损失
        shanten_loss_val: 向听损失
        riichi_loss_val: 立直损失
        weights: 各损失的权重，默认 {
            'policy': 1.0,
            'value': 0.5,
            'shanten': 0.3,
            'riichi': 0.3
        }
    
    Returns:
        total: 总损失
        loss_dict: 各损失的字典（便于调试）
    
    Example:
        >>> total, losses = total_loss(p_loss, v_loss, s_loss, r_loss)
        >>> print(f"总损失: {total.item():.4f}")
        >>> print(f"各损失权重: {losses}")
    """
    if weights is None:
        weights = {
            'policy': 1.0,
            'value': 0.5,
            'shanten': 0.3,
            'riichi': 0.3
        }
    
    total = (
        weights['policy'] * policy_loss_val +
        weights['value'] * value_loss_val +
        weights['shanten'] * shanten_loss_val +
        weights['riichi'] * riichi_loss_val
    )
    
    loss_dict = {
        'total': total,
        'policy': policy_loss_val,
        'value': value_loss_val,
        'shanten': shanten_loss_val,
        'riichi': riichi_loss_val
    }
    
    return total, loss_dict


# ============== 可解释性工具 ==============

@dataclass
class InterpretableOutput:
    """
    可解释的模型输出
    
    包含模型的完整输出信息，便于生成人类可读的解释
    
    Attributes:
        policy_logits: 策略logits
        policy_probs: 策略概率分布
        value: 局面价值
        pred_shanten: 预测向听数
        pred_riichi: 预测立直概率
        true_shanten: 真实向听数（可选）
        legal_actions: 合法动作列表（可选）
        selected_action: 最终选择（可选）
        features: 输入特征（可选）
        losses: 各损失值（可选）
    """
    policy_logits: Optional[torch.Tensor] = None
    policy_probs: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    pred_shanten: Optional[torch.Tensor] = None
    pred_riichi: Optional[torch.Tensor] = None
    true_shanten: Optional[torch.Tensor] = None
    legal_actions: Optional[List[Any]] = None
    selected_action: Optional[Any] = None
    features: Optional[Any] = None
    losses: Optional[Dict[str, float]] = None
    
    def explain(self) -> str:
        """
        生成人类可读的解释
        
        Returns:
            包含决策依据的字符串
        
        Example:
            >>> output = InterpretableOutput(...)
            >>> print(output.explain())
            预测向听数: 0.23
            真实向听数: 0
            特征向听数: 0
            立直概率: 0.87
            → 模型倾向于立直
            局面价值: 0.45
            → 局面优势
            推荐动作: Dahai(pai='5m')
        """
        lines = []
        
        # 向听数解释
        pred_s = self.pred_shanten.item() if self.pred_shanten is not None else 0
        lines.append(f"预测向听数: {pred_s:.2f}")
        
        if self.true_shanten is not None:
            true_s = self.true_shanten.item() if isinstance(self.true_shanten, torch.Tensor) else self.true_shanten
            lines.append(f"真实向听数: {true_s}")
        
        if self.features is not None and hasattr(self.features, 'shanten'):
            feat_s = self.features.shanten
            lines.append(f"特征向听数: {feat_s}")
        
        lines.append("")
        
        # 立直决策解释
        pred_r = self.pred_riichi.item() if self.pred_riichi is not None else 0
        lines.append(f"立直概率: {pred_r:.3f}")
        
        if pred_r > 0.7:
            lines.append("→ 模型倾向于立直")
        elif pred_r < 0.3:
            lines.append("→ 模型倾向于不立直")
        else:
            lines.append("→ 模型在立直与否之间犹豫")
        
        lines.append("")
        
        # 价值评估解释
        val = self.value.item() if self.value is not None else 0
        lines.append(f"局面价值: {val:.3f}")
        
        if val > 0.5:
            lines.append("→ 局面优势")
        elif val < -0.5:
            lines.append("→ 局面劣势")
        else:
            lines.append("→ 局面均势")
        
        # 策略决策解释
        if self.policy_probs is not None and self.legal_actions:
            probs = self.policy_probs.detach().cpu().numpy()
            top_idx = probs.argmax()
            if top_idx < len(self.legal_actions):
                top_action = self.legal_actions[top_idx]
                lines.append("")
                lines.append(f"推荐动作: {top_action}")
        
        # 损失信息
        if self.losses:
            lines.append("")
            lines.append("损失分解:")
            for name, value in self.losses.items():
                if name != 'total' and isinstance(value, (int, float)):
                    lines.append(f"  {name}: {value:.4f}")
        
        return "\n".join(lines)


# ============== 训练工具 ==============

def create_optimizer(
    model: EnhancedModel,
    lr: float = 1e-4,
    weight_decay: float = 1e-5
) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型实例
        lr: 学习率
        weight_decay: 权重衰减
    
    Returns:
        Adam优化器
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def train_step(
    model: EnhancedModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer
) -> Tuple[float, Dict[str, float]]:
    """
    单步训练
    
    Args:
        model: 模型实例
        batch: 批次数据
        optimizer: 优化器
    
    Returns:
        total_loss: 总损失
        loss_dict: 各损失字典
    """
    model.train()
    optimizer.zero_grad()
    
    total_loss, loss_dict = model.compute_losses(batch)
    
    total_loss.backward()
    optimizer.step()
    
    # 转换为Python float便于日志记录
    loss_dict_float = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in loss_dict.items()
    }
    
    return total_loss.item(), loss_dict_float


def validate(
    model: EnhancedModel,
    batch: Dict[str, torch.Tensor]
) -> Tuple[float, Dict[str, float]]:
    """
    验证
    
    Args:
        model: 模型实例
        batch: 批次数据
    
    Returns:
        total_loss: 总损失
        loss_dict: 各损失字典
    """
    model.eval()
    
    with torch.no_grad():
        total_loss, loss_dict = model.compute_losses(batch)
    
    # 转换为Python float便于日志记录
    loss_dict_float = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in loss_dict.items()
    }
    
    return total_loss.item(), loss_dict_float
