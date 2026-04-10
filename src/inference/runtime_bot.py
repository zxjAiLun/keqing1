from __future__ import annotations

import json as _json
from pathlib import Path
from typing import List, Optional

import torch

from inference import (
    DefaultActionScorer,
    DefaultDecisionContextBuilder,
    DefaultRuntimeReviewExporter,
    KeqingModelAdapter,
)
from mahjong_env.legal_actions import enumerate_legal_actions as _enumerate_legal_actions
from mahjong_env.state import GameState

enumerate_legal_actions = _enumerate_legal_actions


def inject_shanten_waits(
    snap: dict,
    *,
    hand_list: list,
    melds_list: list,
    model_version: str,
) -> None:
    from mahjong_env.replay import _calc_shanten_waits
    from mahjong_env.tiles import tile_to_34 as _tile_to_34

    shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(hand_list, melds_list)
    if model_version in {"keqingv3", "keqingv31"}:
        from keqingv3.progress_oracle import calc_standard_shanten_from_counts

        counts34 = [0] * 34
        for tile in hand_list:
            idx34 = _tile_to_34(tile)
            if 0 <= idx34 < 34:
                counts34[idx34] += 1
        shanten = int(calc_standard_shanten_from_counts(tuple(counts34)))
    snap["shanten"] = shanten
    snap["waits_count"] = waits_cnt
    snap["waits_tiles"] = waits_tiles


class RuntimeBot:
    """共享运行时推理 bot 壳。

    v1/v2/v3 的差异由 checkpoint 和 KeqingModelAdapter 自动识别，
    battle/replay 不应再把它挂在某个特定模型版本命名空间下。
    """

    def __init__(
        self,
        player_id: int,
        model_path: str | Path,
        device: str = "cuda",
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        style_vec: Optional[List[float]] = None,
        verbose: bool = False,
        beam_k: int = 3,
        beam_lambda: float = 1.0,
        score_delta_lambda: float = 0.20,
        win_prob_lambda: float = 0.20,
        dealin_prob_lambda: float = 0.25,
        model_version: Optional[str] = None,
    ):
        self.player_id = player_id
        self.verbose = verbose
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.style_vec = (
            list(style_vec) if style_vec is not None else [0.0, 0.0, 0.0, 0.0]
        )
        self.beam_k = beam_k
        self.beam_lambda = beam_lambda
        self.score_delta_lambda = score_delta_lambda
        self.win_prob_lambda = win_prob_lambda
        self.dealin_prob_lambda = dealin_prob_lambda

        self._adapter = KeqingModelAdapter.from_checkpoint(
            model_path,
            device=self.device,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            model_version=model_version,
        )
        self._model_version = self._adapter.model_version
        # Keep a mutable hook to the raw feature encoder. react() temporarily
        # assigns this into adapter._encode so tests/runtime can override
        # self._encode without creating encode -> _encode -> encode recursion.
        self._encode = self._adapter._encode
        self.model = self._adapter.model

        self.decision_log: list = []
        self.game_state = GameState()
        try:
            import riichi as _riichi

            self._riichi_state = _riichi.state.PlayerState(player_id)
        except Exception:
            self._riichi_state = None
        self._rebuild_helpers()

    def _rebuild_helpers(self) -> None:
        self._context_builder = DefaultDecisionContextBuilder(
            model_version=self._model_version,
            riichi_state=self._riichi_state,
            inject_shanten_waits=inject_shanten_waits,
            enumerate_legal_actions_fn=lambda snap, seat: enumerate_legal_actions(snap, seat),
        )
        self._scorer = DefaultActionScorer(
            adapter=self._adapter,
            beam_k=self.beam_k,
            beam_lambda=self.beam_lambda,
            style_lambda=self.style_vec[0],
            score_delta_lambda=self.score_delta_lambda,
            win_prob_lambda=self.win_prob_lambda,
            dealin_prob_lambda=self.dealin_prob_lambda,
        )
        self._review_exporter = DefaultRuntimeReviewExporter()

    def reset(self):
        self.decision_log.clear()
        self.game_state = GameState()
        if self._riichi_state is not None:
            try:
                import riichi as _riichi

                self._riichi_state = _riichi.state.PlayerState(self.player_id)
            except Exception:
                pass
        self._rebuild_helpers()

    @torch.no_grad()
    def react(self, event: dict, gt_action: Optional[dict] = None) -> Optional[dict]:
        actor = self.player_id
        state = self.game_state

        payload = _json.dumps(event, ensure_ascii=False)
        if self._riichi_state is not None:
            try:
                self._riichi_state.update(payload)
            except Exception:
                pass

        ctx = self._context_builder.build(state, actor, event)
        if ctx is None:
            return None
        legal_dicts = ctx.legal_actions

        non_none = [a for a in legal_dicts if a.get("type") != "none"]
        if not legal_dicts:
            return None
        if not non_none:
            return {"type": "none", "actor": actor}

        self._adapter._encode = self._encode
        decision = self._scorer.score(ctx)
        chosen = decision.chosen

        if chosen.get("type") == "none":
            has_hora = any(a.get("type") == "hora" for a in legal_dicts)
            if has_hora:
                state.players[actor].doujun_furiten = True
                state.players[actor].furiten = True

        decision_entry = self._review_exporter.build_decision_entry(
            step=len(self.decision_log),
            ctx=ctx,
            decision=decision,
            gt_action=gt_action,
            actor=actor,
        )
        self.decision_log.append(decision_entry)

        if self.verbose:
            print(f"[Bot {self.player_id}] 决策:")
            for c in decision_entry["candidates"]:
                a = c["action"]
                logit = c["logit"]
                marker = " <-- 选择" if a == chosen else ""
                print(f"  {logit:+.3f}  {a}{marker}")

        return chosen


__all__ = [
    "RuntimeBot",
    "inject_shanten_waits",
    "enumerate_legal_actions",
]
