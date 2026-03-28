"""MahjongModel v5 — Conv1d ResNet + dual head (policy + value).

输入：
  tile_feat : Tensor (B, C_TILE=128, 34)   — 牌维度特征
  scalar    : Tensor (B, N_SCALAR=16)       — 标量特征

输出：
  policy_logits : Tensor (B, ACTION_SPACE=45)  — 未 softmax
  value         : Tensor (B, 1)                — Tanh 输出
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v5model.action_space import ACTION_SPACE
from v5model.features import C_TILE, N_SCALAR


class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        scale = x.mean(dim=-1)          # (B, C)
        scale = self.fc(scale)          # (B, C)
        return x * scale.unsqueeze(-1)  # (B, C, L)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
        )
        self.se = SEBlock(channels)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.net(x)))


class MahjongModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        c_tile: int = C_TILE,
        n_scalar: int = N_SCALAR,
        action_space: int = ACTION_SPACE,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 升维投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(c_tile, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-3),
            nn.Mish(),
        )

        # 残差塔
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # 全局池化后拼接标量
        self.scalar_proj = nn.Sequential(
            nn.Linear(n_scalar, 32),
            nn.Mish(),
        )

        # 共享全连接
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_space)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        tile_feat: torch.Tensor,
        scalar: torch.Tensor,
    ):
        # tile_feat: (B, C_TILE, 34)
        x = self.input_proj(tile_feat)    # (B, hidden_dim, 34)
        x = self.res_blocks(x)            # (B, hidden_dim, 34)
        x = x.mean(dim=-1)               # (B, hidden_dim) — global avg pool

        s = self.scalar_proj(scalar)      # (B, 32)
        x = torch.cat([x, s], dim=-1)    # (B, hidden_dim + 32)
        x = self.fc_shared(x)            # (B, hidden_dim)

        policy_logits = self.policy_head(x)  # (B, action_space)
        value = self.value_head(x)           # (B, 1)
        return policy_logits, value
