"""
V3 增强模型包

包含高可解释性的多任务模型架构和相关工具。
"""

from .model import (
    EnhancedModel,
    ResNetEncoder,
    PolicyHead,
    ValueHead,
    ShantenHead,
    RiichiHead,
    policy_loss,
    value_loss,
    shanten_loss,
    riichi_loss,
    total_loss,
    create_optimizer,
    train_step,
    validate,
)

from .features import (
    EnhancedFeatures,
    extract_enhanced_features,
    vectorize_enhanced_features,
    WaitsQuality,
    GamePhase,
)
from .dataset import EnhancedDataset, create_dataloader
from .bot import EnhancedBot

__all__ = [
    # Model
    "EnhancedModel",
    "ResNetEncoder",
    "PolicyHead",
    "ValueHead",
    "ShantenHead",
    "RiichiHead",
    # Loss functions
    "policy_loss",
    "value_loss",
    "shanten_loss",
    "riichi_loss",
    "total_loss",
    # Training utilities
    "create_optimizer",
    "train_step",
    "validate",
    # Features
    "extract_enhanced_features",
    "vectorize_enhanced_features",
    # Dataset
    "EnhancedDataset",
    "create_dataloader",
    # Bot
    "EnhancedBot",
]
