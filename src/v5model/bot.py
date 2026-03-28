"""V5Bot — 基于 MahjongModel v5 的 mjai 协议推理 Bot。

用法：
    bot = V5Bot(player_id=0, model_path="best.pth", device="cuda")
    action = bot.react(events)  # events: List[dict] mjai 事件
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.state import GameState, apply_event
from v5model.action_space import (
    ACTION_SPACE,
    NONE_IDX,
    action_to_idx,
    build_legal_mask,
    chi_type_idx,
)
from v5model.features import encode
from v5model.model import MahjongModel


def _find_best_legal(policy_logits: np.ndarray, legal_actions: list) -> dict:
    """根据 policy logits 在合法动作中选分数最高的。"""
    best_score = -1e18
    best_action = legal_actions[0]
    seen_chi: dict = {}  # chi_idx -> action，保留第一个匹配的

    for a in legal_actions:
        idx = action_to_idx(a)
        if idx == NONE_IDX:
            score = policy_logits[NONE_IDX]
        else:
            score = policy_logits[idx]
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


class V5Bot:
    def __init__(
        self,
        player_id: int,
        model_path: str | Path,
        device: str = "cuda",
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
    ):
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = MahjongModel(
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.game_state = GameState()

    def reset(self):
        self.game_state = GameState()

    @torch.no_grad()
    def react(self, event: dict) -> Optional[dict]:
        """处理单个 mjai 事件，返回需要响应时的动作 dict，否则返回 None。"""
        actor = self.player_id
        state = self.game_state

        # 判断是否需要本 bot 响应
        needs_response = False
        legal_actions: list = []

        etype = event.get("type", "")

        if etype == "tsumo" and event.get("actor") == actor:
            # 自家摸牌，需要弃牌/立直/自摸
            apply_event(state, event)
            snap = state.snapshot(actor)
            legal_actions = enumerate_legal_actions(state, actor)
            needs_response = True
        elif etype == "dahai" and event.get("actor") != actor:
            # 他家弃牌，可能需要鸣牌/荣和/pass
            apply_event(state, event)
            snap = state.snapshot(actor)
            legal_actions = enumerate_legal_actions(state, actor)
            needs_response = bool(legal_actions)
        else:
            apply_event(state, event)
            return None

        if not needs_response or not legal_actions:
            return None

        snap = state.snapshot(actor)
        tile_feat, scalar = encode(snap, actor)

        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)  # (1, C, 34)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)   # (1, S)

        policy_logits, _ = self.model(tile_t, scalar_t)
        logits_np = policy_logits.squeeze(0).cpu().numpy()  # (45,)

        # 将非法动作设为 -1e9
        mask = np.array(build_legal_mask(legal_actions), dtype=np.float32)
        logits_np = np.where(mask > 0, logits_np, -1e9)

        chosen = _find_best_legal(logits_np, legal_actions)
        # 转为 mjai dict 格式
        if hasattr(chosen, "to_dict"):
            return chosen.to_dict()
        if hasattr(chosen, "__dict__"):
            return {k: v for k, v in chosen.__dict__.items() if v is not None}
        return dict(chosen)
