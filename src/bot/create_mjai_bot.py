#!/usr/bin/env python3
"""
创建 mjai Docker 格式的 bot.py

将我们的 MjaiPolicyBot 适配到 mjai Bot 接口
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mjai.bot.base import Bot as MjaiBaseBot
from bot.mjai_bot import _NumpylessMultiTaskModel, _to_scalar
from bot.features import vectorize_state_py
from mahjong_env.legal_actions import enumerate_legal_actions
from model.vocab import action_to_token


class MjaiDockerBot(MjaiBaseBot):
    """
    mjai Docker 格式的 Bot，继承自 mjai.bot.base.Bot

    使用纯 Python 推理（无 NumPy 依赖）
    """

    def __init__(self, player_id: int, checkpoint_json: str):
        super().__init__(player_id)
        self.player_id = player_id
        self.checkpoint_json = checkpoint_json
        self.action_vocab, self.stoi, self.model = self._load_checkpoint(checkpoint_json)

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt = json.load(open(checkpoint_path, "r", encoding="utf-8"))
        action_vocab = ckpt.get("action_vocab", [])
        stoi = {a: i for i, a in enumerate(action_vocab)}
        model = _NumpylessMultiTaskModel.from_state_dict_json(ckpt["model_state_dict"])
        return action_vocab, stoi, model

    def _choose_action(self):
        obs = vectorize_state_py(self, self.player_id)
        logits = self.model.forward_logits(obs)

        best_a = None
        best_logit = float("-inf")
        for a in self._get_legal_actions():
            if a.get("type") == "none":
                continue
            tok = self._action_to_token(a)
            idx = self.stoi.get(tok)
            if idx is None:
                continue
            v = logits[idx]
            if v > best_logit:
                best_logit = v
                best_a = a

        if best_a is not None:
            return best_a
        return {"type": "none"}

    def _get_legal_actions(self):
        actions = []
        if self.can_discard:
            for tile in self.discardable_tiles:
                actions.append({
                    "type": "dahai",
                    "pai": tile,
                    "actor": self.player_id,
                    "tsumogiri": tile == self.last_self_tsumo,
                })
        if self.can_pon:
            tile = self.last_kawa_tile
            consumed = [tile, tile]
            actions.append({
                "type": "pon",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": tile,
                "consumed": consumed,
            })
        if self.can_chi:
            tile = self.last_kawa_tile
            color = tile[1] if len(tile) > 1 else ""
            num = int(tile[0]) if tile[0].isdigit() else 0
            if self.can_chi_high and num >= 2:
                consumed = [f"{num-2}{color}", f"{num-1}{color}"]
                actions.append({
                    "type": "chi",
                    "actor": self.player_id,
                    "target": self.target_actor,
                    "pai": tile,
                    "consumed": consumed,
                })
            if self.can_chi_low and num <= 8:
                consumed = [f"{num+1}{color}", f"{num+2}{color}"]
                actions.append({
                    "type": "chi",
                    "actor": self.player_id,
                    "target": self.target_actor,
                    "pai": tile,
                    "consumed": consumed,
                })
        if self.can_agari or self.can_tsumo_agari:
            actions.append({"type": "hora", "actor": self.player_id, "target": self.target_actor, "pai": self.last_self_tsumo})
        if self.can_agari or self.can_ron_agari:
            actions.append({"type": "hora", "actor": self.player_id, "target": self.target_actor, "pai": self.last_kawa_tile})
        if self.can_ryukyoku:
            actions.append({"type": "ryukyoku", "actor": self.player_id})
        if self.can_kan or self.can_ankan:
            actions.append({"type": "ankan", "actor": self.player_id, "consumed": ["?", "?", "?"]})
        if self.can_kakan:
            actions.append({"type": "kakan", "actor": self.player_id, "pai": "?"})
        if self.can_pass:
            actions.append({"type": "none"})
        return actions

    def _action_to_token(self, action):
        a = action.copy()
        a.pop("actor", None)
        return action_to_token(a)

    def think(self) -> str:
        action = self._choose_action()
        if action.get("type") == "none":
            return self.action_nothing()
        elif action.get("type") == "dahai":
            return self.action_discard(action["pai"])
        elif action.get("type") == "hora":
            if action.get("pai") == self.last_self_tsumo:
                return self.action_tsumo_agari()
            else:
                return self.action_ron_agari()
        elif action.get("type") == "pon":
            return self.action_pon(action["consumed"])
        elif action.get("type") == "chi":
            return self.action_chi(action["consumed"])
        elif action.get("type") == "reach":
            return self.action_riichi()
        elif action.get("type") == "ankan":
            return self.action_ankan(action["consumed"])
        elif action.get("type") == "kakan":
            return self.action_kakan(action["pai"])
        elif action.get("type") == "ryukyoku":
            return self.action_ryukyoku()
        else:
            return self.action_nothing()


def main():
    if len(sys.argv) < 3:
        print("Usage: python create_mjai_bot.py <player_id> <checkpoint_json>", file=sys.stderr)
        sys.exit(1)

    player_id = int(sys.argv[1])
    checkpoint_json = sys.argv[2]

    bot = MjaiDockerBot(player_id, checkpoint_json)
    bot.start()


if __name__ == "__main__":
    main()
