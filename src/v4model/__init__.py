"""
V4Model - 基于 riichienv 的麻雀 AI 模型

架构设计参考 Mortal:
- ResNet Encoder + ChannelAttention (提取特征)
- Policy Head (动作概率)
- Value Head (状态价值)

特征工程:
- 37 tile types (包含 aka5) 作为行
- 多个特征通道作为列
- 组成 2D feature map 供 ResNet 处理
"""

from .features import FeatureEncoder, N_TILE_TYPES
from .model import ModelV4, Encoder, PolicyHead, ValueHead
from .bot import V4Bot, create_v4_bot

__all__ = ["FeatureEncoder", "N_TILE_TYPES", "ModelV4", "Encoder", "PolicyHead", "ValueHead", "V4Bot", "create_v4_bot"]
