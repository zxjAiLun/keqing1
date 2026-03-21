"""
ModelV4 - 基于 riichienv 的麻雀 AI 模型

架构设计参考 Mortal:
- ResNet Encoder + ChannelAttention (提取特征)
- Policy Head (动作概率)
- Value Head (状态价值)

输入:
- tile_features: (batch, 37, tile_plane_dim) - 牌类型平面
- scalar_features: (batch, scalar_dim) - 标量特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 第一步: 定义基础模块
# ============================================================================

class ResidualBlock(nn.Module):
    """
    残差块 - 基础 ResNet 结构

    输入: (batch, channels, height, width) 或 (batch, channels)
    输出: (batch, channels, height, width) 或 (batch, channels)
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual  # 残差连接
        out = F.relu(out)
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力机制 - 参考 SE-Net

    对每个通道进行加权，强调重要特征

    结构:
    1. Squeeze: Global Average Pooling
    2. Excitation: FC -> ReLU -> FC -> Sigmoid
    3. Scale: 通道权重相乘
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        # Squeeze: 将 (B, C, H, W) -> (B, C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Excitation: (B, C) -> (B, C // reduction) -> (B, C)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        batch, channels, _, _ = x.size()

        # Squeeze
        y = self.gap(x).view(batch, channels)

        # Excitation
        y = self.fc(y).view(batch, channels, 1, 1)

        # Scale
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    """
    SE-ResNet Block - 带有通道注意力的残差块
    """

    def __init__(self, channels: int, reduction: int = 8, dropout: float = 0.1):
        super().__init__()
        self.res_block = ResidualBlock(channels, dropout)
        self.channel_attention = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res_block(x)
        out = self.channel_attention(out)
        return out


# ============================================================================
# 第二步: 定义 Encoder
# ============================================================================

class Encoder(nn.Module):
    """
    特征编码器 - 将牌类型平面和标量特征编码为隐藏向量

    输入:
    - tile_features: (batch, 37, tile_plane_dim) - 牌类型平面
    - scalar_features: (batch, scalar_dim) - 标量特征

    输出:
    - encoded: (batch, hidden_dim) - 编码后的特征向量
    """

    def __init__(
        self,
        tile_plane_dim: int = 32,
        scalar_dim: int = 24,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ------------------------------------------------------------------
        # 1. 将 tile_features (37, tile_plane_dim) 转换为 (37, hidden_dim)
        # ------------------------------------------------------------------
        # 使用 2D 卷积处理: (B, C, H, W) = (B, tile_plane_dim, 37, 1)
        self.tile_proj = nn.Sequential(
            nn.Conv2d(tile_plane_dim, hidden_dim, kernel_size=(37, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 2. ResNet Blocks + ChannelAttention
        # ------------------------------------------------------------------
        self.res_blocks = nn.ModuleList([
            SEBlock(hidden_dim, reduction=8, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # ------------------------------------------------------------------
        # 3. 处理 scalar_features 并融合
        # ------------------------------------------------------------------
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # ------------------------------------------------------------------
        # 4. 融合层
        # ------------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, tile_features: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            tile_features: (batch, 37, tile_plane_dim)
            scalar_features: (batch, scalar_dim)

        Returns:
            encoded: (batch, hidden_dim)
        """
        batch = tile_features.size(0)

        # ------------------------------------------------------------------
        # 1. 处理 tile_features
        # ------------------------------------------------------------------
        # 转换为 (B, C, H, W) 格式: (B, tile_plane_dim, 37, 1)
        tile_x = tile_features.unsqueeze(-1).permute(0, 2, 1, 3)

        # 投影到 hidden_dim
        tile_x = self.tile_proj(tile_x)  # (B, hidden_dim, 1, 1)
        tile_x = tile_x.squeeze(-1).squeeze(-1)  # (B, hidden_dim)

        # ------------------------------------------------------------------
        # 2. ResNet Blocks
        # ------------------------------------------------------------------
        # 转换为 (B, hidden_dim, 1, 1) 格式
        tile_x = tile_x.unsqueeze(-1).unsqueeze(-1)

        for res_block in self.res_blocks:
            tile_x = res_block(tile_x)

        tile_x = tile_x.squeeze(-1).squeeze(-1)  # (B, hidden_dim)

        # ------------------------------------------------------------------
        # 3. 处理 scalar_features
        # ------------------------------------------------------------------
        scalar_x = self.scalar_net(scalar_features)  # (B, hidden_dim)

        # ------------------------------------------------------------------
        # 4. 融合
        # ------------------------------------------------------------------
        fused = torch.cat([tile_x, scalar_x], dim=1)  # (B, hidden_dim * 2)
        encoded = self.fusion(fused)  # (B, hidden_dim)

        return encoded


# ============================================================================
# 第三步: 定义 Policy Head
# ============================================================================

class PolicyHead(nn.Module):
    """
    策略头 - 输出动作概率

    输入: (batch, hidden_dim)
    输出: (batch, num_actions)
    """

    def __init__(self, hidden_dim: int = 256, num_actions: int = 38):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 第四步: 定义 Value Head
# ============================================================================

class ValueHead(nn.Module):
    """
    价值头 - 输出状态价值

    输入: (batch, hidden_dim)
    输出: (batch, 1)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 第五步: 定义完整 ModelV4
# ============================================================================

class ModelV4(nn.Module):
    """
    完整的 ModelV4 模型

    输入:
    - tile_features: (batch, 37, tile_plane_dim) - 牌类型平面
    - scalar_features: (batch, scalar_dim) - 标量特征

    输出:
    - policy_logits: (batch, num_actions) - 动作 logit
    - value: (batch, 1) - 状态价值
    """

    def __init__(
        self,
        tile_plane_dim: int = 32,
        scalar_dim: int = 24,
        hidden_dim: int = 256,
        num_actions: int = 38,
        num_res_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            tile_plane_dim=tile_plane_dim,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )

        self.policy_head = PolicyHead(hidden_dim, num_actions)
        self.value_head = ValueHead(hidden_dim)

    def forward(
        self,
        tile_features: torch.Tensor,
        scalar_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            tile_features: (batch, 37, tile_plane_dim)
            scalar_features: (batch, scalar_dim)

        Returns:
            policy_logits: (batch, num_actions)
            value: (batch, 1)
        """
        encoded = self.encoder(tile_features, scalar_features)
        policy_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return policy_logits, value


def test_model():
    """测试模型"""
    from v4model.features import FeatureEncoder
    from riichienv import RiichiEnv

    # 创建模型
    model = ModelV4(
        tile_plane_dim=32,
        scalar_dim=24,
        hidden_dim=256,
        num_actions=38
    )

    print("=== ModelV4 结构 ===")
    print(model)

    # 测试前向传播
    batch_size = 2
    tile_features = torch.randn(batch_size, 37, 32)
    scalar_features = torch.randn(batch_size, 24)

    policy_logits, value = model(tile_features, scalar_features)

    print(f"\n=== 输出形状 ===")
    print(f"policy_logits: {policy_logits.shape} (batch, num_actions)")
    print(f"value: {value.shape} (batch, 1)")

    # 测试使用真实数据
    print(f"\n=== 使用真实数据测试 ===")
    encoder = FeatureEncoder(tile_plane_dim=32, scalar_dim=24)

    env = RiichiEnv(game_mode='4p-red-half')
    obs_dict = env.reset()

    # 获取第一个玩家的观察
    obs = list(obs_dict.values())[0]
    tile_feat, scalar_feat = encoder.encode(obs)

    tile_feat = torch.FloatTensor(tile_feat).unsqueeze(0)
    scalar_feat = torch.FloatTensor(scalar_feat).unsqueeze(0)

    policy_logits, value = model(tile_feat, scalar_feat)

    print(f"policy_logits: {policy_logits.shape}")
    print(f"value: {value.item():.4f}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 参数量 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


if __name__ == "__main__":
    test_model()
