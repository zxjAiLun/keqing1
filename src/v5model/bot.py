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
        style_vec: Optional[List[float]] = None,
        verbose: bool = False,
    ):
        self.player_id = player_id
        self.verbose = verbose
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # style_vec: [speed, riichi, value, defense]，均为 [-1, +1]，None 则全 0
        self.style_vec = list(style_vec) if style_vec is not None else [0.0, 0.0, 0.0, 0.0]

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        # 从 checkpoint 自动推断 n_scalar（兼容旧版 N_SCALAR=16）
        n_scalar = state_dict["scalar_proj.0.weight"].shape[1]
        self.model = MahjongModel(
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            n_scalar=n_scalar,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.decision_log: list = []
        self.game_state = GameState()
        try:
            import riichi as _riichi
            self._riichi_state = _riichi.state.PlayerState(player_id)
        except Exception:
            self._riichi_state = None

    def reset(self):
        self.decision_log.clear()
        self.game_state = GameState()
        if self._riichi_state is not None:
            try:
                import riichi as _riichi
                self._riichi_state = _riichi.state.PlayerState(self.player_id)
            except Exception:
                pass

    @torch.no_grad()
    def react(self, event: dict, gt_action: Optional[dict] = None) -> Optional[dict]:
        """处理单个 mjai 事件，返回需要响应时的动作 dict，否则返回 None。"""
        actor = self.player_id
        state = self.game_state

        # 判断是否需要本 bot 响应
        needs_response = False
        legal_actions: list = []

        etype = event.get("type", "")

        import json as _json
        _payload = _json.dumps(event, ensure_ascii=False)
        if self._riichi_state is not None:
            try:
                self._riichi_state.update(_payload)
            except Exception:
                pass

        if etype == "tsumo" and event.get("actor") == actor:
            # 自家摸牌，需要弃牌/立直/自摸
            apply_event(state, event)
            needs_response = True
        elif etype == "dahai" and event.get("actor") != actor:
            # 他家弃牌，可能需要鸣牌/荣和/pass
            apply_event(state, event)
            needs_response = True
        elif etype in ("chi", "pon", "daiminkan") and event.get("actor") == actor:
            # 自家副露后需要打牌（riichienv 在同一 step 发此事件让玩家决策打牌）
            apply_event(state, event)
            needs_response = True
        elif etype == "reach" and event.get("actor") == actor:
            # 自家立直宣告后需要打出立直宣言牌
            apply_event(state, event)
            needs_response = True
        else:
            apply_event(state, event)
            return None

        # 先注入 shanten/waits，再枚举合法动作（shanten 影响 reach 是否合法）
        snap = state.snapshot(actor)
        injected = False
        if self._riichi_state is not None:
            try:
                snap["shanten"] = int(self._riichi_state.shanten)
                snap["waits_count"] = int(sum(self._riichi_state.waits))
                snap["waits_tiles"] = list(self._riichi_state.waits)
                injected = True
            except Exception:
                pass
        if not injected:
            # fallback：用 riichienv 自算 shanten/waits
            from mahjong_env.replay import _calc_shanten_waits
            hand_list = snap.get("hand", [])
            melds_list = (snap.get("melds") or [[], [], [], []])[actor]
            shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(hand_list, melds_list)
            snap["shanten"] = shanten
            snap["waits_count"] = waits_cnt
            snap["waits_tiles"] = waits_tiles
        legal_actions = enumerate_legal_actions(snap, actor)

        # 如果合法动作只有 none（无实质选择），直接返回 none 不做推理
        non_none = [a for a in legal_actions if a.type != "none"]
        if not needs_response or not legal_actions:
            return None
        if not non_none:
            return {"type": "none", "actor": actor}
        tile_feat, scalar = encode(snap, actor)
        n_scalar = self.model.scalar_proj[0].weight.shape[1]
        if n_scalar >= 20:
            scalar[16:20] = self.style_vec  # 注入风格向量
        scalar = scalar[:n_scalar]

        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)  # (1, C, 34)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)   # (1, S)

        policy_logits, _ = self.model(tile_t, scalar_t)
        logits_np = policy_logits.squeeze(0).cpu().numpy()  # (45,)

        # Action dataclass → mjai dict
        legal_dicts = [a.to_mjai() for a in legal_actions]

        # 将非法动作设为 -1e9
        mask = np.array(build_legal_mask(legal_dicts), dtype=np.float32)
        logits_np = np.where(mask > 0, logits_np, -1e9)

        chosen = _find_best_legal(logits_np, legal_dicts)


        # 记录决策日志（供 HTML 导出）
        scored = sorted(
            [{"action": a, "logit": float(logits_np[action_to_idx(a)])} for a in legal_dicts],
            key=lambda x: x["logit"], reverse=True,
        )
        self.decision_log.append({
            "step": len(self.decision_log),
            "bakaze": snap.get("bakaze", ""),
            "kyoku": snap.get("kyoku", 0),
            "honba": snap.get("honba", 0),
            "scores": snap.get("scores", []),
            "hand": snap.get("hand", []),
            "discards": snap.get("discards", []),
            "dora_markers": snap.get("dora_markers", []),
            "reached": snap.get("reached", []),
            "candidates": scored,
            "chosen": chosen,
            "gt_action": gt_action,
        })

        if self.verbose:
            print(f"[Bot {self.player_id}] 决策:")
            scored = []
            for a in legal_dicts:
                idx = action_to_idx(a)
                scored.append((logits_np[idx], a))
            scored.sort(key=lambda x: x[0], reverse=True)
            for logit, a in scored:
                marker = " <-- 选择" if a == chosen else ""
                print(f"  {logit:+.3f}  {a}{marker}")

        return chosen
