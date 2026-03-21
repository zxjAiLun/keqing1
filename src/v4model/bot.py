"""
ModelV4 Bot

基于 ModelV4 的麻将 AI Bot
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
import numpy as np

from riichienv import RiichiEnv, Action
import riichienv.convert as cvt

sys.path.insert(0, str(Path(__file__).parent.parent))

from v4model import ModelV4, FeatureEncoder
from v4model.features import N_TILE_TYPES


def _action_to_token_name(action: Union[Action, Dict]) -> str:
    """将 Action 对象或 dict 转换为 token name"""
    # 处理 Action 对象
    if isinstance(action, Action):
        action_type_str = str(action.action_type).split('.')[-1]  # e.g., "DISCARD"
        action_type_lower = action_type_str.lower()
        
        if action_type_lower == "discard":
            tile_id = action.tile
            if tile_id is not None:
                tile_str = cvt.tid_to_mpsz(tile_id)
                return f"dahai_{tile_str}"
            return "dahai_unknown"
        elif action_type_lower in {"chi", "pon", "daiminkan", "ankan", "kakan", "pass", "ron", "tsumo"}:
            return action_type_lower
        elif action_type_lower == "riichi":
            return "reach"
        elif action_type_lower == "hora":
            return "hora"
        else:
            return action_type_lower
    
    # 处理 dict (兼容旧格式)
    if isinstance(action, dict):
        action_type = action.get("type", "")
        if action_type == "dahai":
            tile = action.get("pai", "").replace(" ", "")
            return f"dahai_{tile}"
        return action_type
    
    return "unknown"


class V4Bot:
    """ModelV4 机器人"""

    def __init__(
        self,
        player_id: int,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.player_id = player_id
        self.device = device

        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        self.actions = checkpoint.get("actions", [])

        # 创建模型
        num_actions = len(self.actions)
        self.model = ModelV4(
            tile_plane_dim=32,
            scalar_dim=24,
            hidden_dim=256,
            num_res_blocks=3,
            num_actions=num_actions,
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 特征编码器
        self.feature_encoder = FeatureEncoder()

        # 动作名称到索引的映射
        self.action_stoi = {a: i for i, a in enumerate(self.actions)}

    def act(self, obs) -> Action:
        """根据观察选择动作"""
        with torch.no_grad():
            # 编码特征
            tile_features, scalar_features = self.feature_encoder.encode(obs)

            # 转换为 tensor
            tile_tensor = torch.from_numpy(tile_features).float().unsqueeze(0).to(self.device)
            scalar_tensor = torch.from_numpy(scalar_features).float().unsqueeze(0).to(self.device)

            # 获取合法动作
            legal_actions = obs.legal_actions()

            # 创建 legal mask
            legal_mask = np.zeros(len(self.actions), dtype=np.float32)
            for a in legal_actions:
                action_name = _action_to_token_name(a)
                if action_name in self.action_stoi:
                    legal_mask[self.action_stoi[action_name]] = 1.0

            # 模型推理
            policy_logits, _ = self.model(tile_tensor, scalar_tensor)

            # 应用 mask
            legal_mask_tensor = torch.from_numpy(legal_mask).to(self.device)
            policy_logits = policy_logits.masked_fill(legal_mask_tensor <= 0, -1e9)

            # 选择概率最高的动作
            probs = torch.softmax(policy_logits, dim=-1)
            action_idx = probs.argmax(dim=-1).item()

            selected_action_name = self.actions[action_idx]

            # 找到对应的 Action 对象
            for a in legal_actions:
                if _action_to_token_name(a) == selected_action_name:
                    return a

            # 如果没找到，返回第一个合法动作（fallback）
            return legal_actions[0] if legal_actions else None


def create_v4_bot(player_id: int, model_path: str, device: str = "cuda") -> V4Bot:
    """工厂函数：创建 V4Bot"""
    return V4Bot(player_id=player_id, model_path=model_path, device=device)
