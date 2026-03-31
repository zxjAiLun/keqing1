"""KeqingBot — 基于 MahjongModel keqingv1 的 mjai 协议推理 Bot。

用法：
    bot = KeqingBot(player_id=0, model_path="best.pth", device="cuda")
    action = bot.react(events)  # events: List[dict] mjai 事件
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.state import GameState, apply_event
from keqingv1.action_space import (
    ACTION_SPACE,
    NONE_IDX,
    action_to_idx,
    build_legal_mask,
    chi_type_idx,
)
def _find_best_legal(
    policy_logits: np.ndarray,
    legal_actions: list,
    value: float = 0.0,
    style_lambda: float = 0.0,
    aux_bonus: float = 0.0,
) -> dict:
    """根据 policy logits 在合法动作中选分数最高的。

    style_lambda > 0：进攻型（对非 none 动作加 lambda * value，倾向高 EV 选择）
    style_lambda < 0：防守型（对非 none 动作减分，倾向 pass/none）
    style_lambda = 0：纯 policy，不做风格干预
    """
    best_score = -1e18
    best_action = legal_actions[0]

    for a in legal_actions:
        idx = action_to_idx(a)
        score = policy_logits[NONE_IDX if idx == NONE_IDX else idx]
        if a.get("type") != "none":
            score += style_lambda * value
            score += aux_bonus
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def _legal_score(
    policy_logits: np.ndarray,
    action: dict,
    value: float = 0.0,
    style_lambda: float = 0.0,
    aux_bonus: float = 0.0,
) -> float:
    """与 _find_best_legal 保持一致的单动作打分。"""
    idx = action_to_idx(action)
    score = float(policy_logits[NONE_IDX if idx == NONE_IDX else idx])
    if action.get("type") != "none":
        score += style_lambda * value
        score += aux_bonus
    return score


def _get_aux_outputs(
    model: torch.nn.Module,
) -> dict[str, float]:
    if not hasattr(model, "get_last_aux_outputs"):
        return {"score_delta": 0.0, "win_prob": 0.0, "dealin_prob": 0.0}
    try:
        aux = model.get_last_aux_outputs()
    except Exception:
        return {"score_delta": 0.0, "win_prob": 0.0, "dealin_prob": 0.0}
    return {
        "score_delta": float(aux["score_delta"].squeeze().detach().cpu().item()),
        "win_prob": float(torch.sigmoid(aux["win_prob"].squeeze()).detach().cpu().item()),
        "dealin_prob": float(torch.sigmoid(aux["dealin_prob"].squeeze()).detach().cpu().item()),
    }


def _aux_bonus(
    aux_outputs: dict[str, float],
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> float:
    return (
        score_delta_lambda * aux_outputs.get("score_delta", 0.0)
        + win_prob_lambda * aux_outputs.get("win_prob", 0.0)
        - dealin_prob_lambda * aux_outputs.get("dealin_prob", 0.0)
    )


def _eval_snapshot_outputs(
    model: torch.nn.Module,
    encode_fn,
    device: torch.device,
    snap: dict,
    actor: int,
) -> tuple[float, dict[str, float]]:
    tile_feat, scalar = encode_fn(snap, actor)
    tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(device)
    scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value_t = model(tile_t, scalar_t)
    value = float(value_t.squeeze().cpu().item())
    return value, _get_aux_outputs(model)


def _dahai_beam_search(
    model: torch.nn.Module,
    encode_fn,
    device: torch.device,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    legal_dahai: list,
    beam_k: int,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> dict:
    """对 dahai 候选做 value beam search。

    取 policy top-k 候选牌，各模拟打出后用 value head 重排，
    返回 policy_logit + beam_lambda * value 最高的动作。
    """
    # 按 policy logit 排序，取 top-k
    sorted_dahai = sorted(
        legal_dahai,
        key=lambda a: policy_logits[action_to_idx(a)],
        reverse=True,
    )[:beam_k]

    best_score = -1e18
    best_action = sorted_dahai[0]
    value_scores: dict = {}

    for a in sorted_dahai:
        pai = a.get("pai", "")
        # 构造打出这张牌后的 fake snapshot
        fake_snap = dict(snap)
        hand = list(snap.get("hand", []))
        # 移除手牌中第一张匹配的牌（赤宝牌等价处理）
        from mahjong_env.tiles import normalize_tile

        norm_pai = normalize_tile(pai)
        removed = False
        new_hand = []
        for t in hand:
            if not removed and normalize_tile(t) == norm_pai:
                removed = True
            else:
                new_hand.append(t)
        if not removed:
            new_hand = hand  # 找不到就不修改
        fake_snap = dict(snap)
        fake_snap["hand"] = new_hand
        # 更新舍牌
        discards = [list(d) for d in snap.get("discards", [[], [], [], []])]
        discards[actor] = discards[actor] + [pai]
        fake_snap["discards"] = discards
        # 打牌后不应再保留当前摸牌高亮
        fake_snap["tsumo_pai"] = None

        value, aux = _eval_snapshot_outputs(model, encode_fn, device, fake_snap, actor)
        score = (
            float(policy_logits[action_to_idx(a)])
            + beam_lambda * value
            + _aux_bonus(aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )
        value_scores[action_to_idx(a)] = score
        if score > best_score:
            best_score = score
            best_action = a

    return best_action, value_scores


def _meld_value_eval(
    model: torch.nn.Module,
    encode_fn,
    device: torch.device,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    meld_actions: list,
    none_actions: list,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> dict:
    """对副露候选（chi/pon/kan）做 value beam search，与 none 对比后返回最优动作。

    对每个副露动作模拟做副露后的状态，用 value head 评估；
    none 仅用 policy logit，不加 value（无动作无后续状态）。
    """
    from mahjong_env.tiles import normalize_tile

    best_score = -1e18
    best_action = none_actions[0] if none_actions else meld_actions[0]
    value_scores: dict = {}

    # 计算 none baseline：policy logit + beam_lambda * v_none（与 meld 评估对称）
    none_score = -1e18
    if none_actions:
        v_none, aux_none = _eval_snapshot_outputs(model, encode_fn, device, snap, actor)
        none_bonus = _aux_bonus(aux_none, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        for a in none_actions:
            s = float(policy_logits[action_to_idx(a)]) + beam_lambda * v_none
            if a.get("type") != "none":
                s += none_bonus
            value_scores[action_to_idx(a)] = s
            if s > none_score:
                none_score = s
                best_score = s
                best_action = a

    for a in meld_actions:
        meld_type = a.get("type", "")
        consumed = a.get("consumed", [])
        pai = a.get("pai", "")

        # 构造副露后的 fake_snap
        fake_snap = dict(snap)
        hand = list(snap.get("hand", []))
        new_hand = list(hand)
        for c in consumed:
            norm_c = normalize_tile(c)
            for i, t in enumerate(new_hand):
                if normalize_tile(t) == norm_c:
                    new_hand.pop(i)
                    break
        fake_snap["hand"] = new_hand

        # 更新 melds（追加完整 dict，与 state.snapshot() / encode() 格式一致）
        melds = [list(m) for m in snap.get("melds", [[], [], [], []])]
        melds[actor] = melds[actor] + [
            {
                "type": meld_type,
                "pai": normalize_tile(pai),
                "consumed": [normalize_tile(c) for c in consumed],
            }
        ]
        fake_snap["melds"] = melds

        value, aux = _eval_snapshot_outputs(model, encode_fn, device, fake_snap, actor)
        score = (
            float(policy_logits[action_to_idx(a)])
            + beam_lambda * value
            + _aux_bonus(aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )
        value_scores[action_to_idx(a)] = score
        if score > best_score:
            best_score = score
            best_action = a

    return best_action, value_scores


def _reach_value_eval(
    model: torch.nn.Module,
    encode_fn,
    device: torch.device,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    reach_action: dict,
    other_actions: list,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> dict:
    """对 reach 决策做 value 评估：模拟立直后状态与其他动作对比。"""
    # reach 后状态：手牌不变，但 reached 标记开启
    fake_snap = dict(snap)
    reached = list(snap.get("reached", [False, False, False, False]))
    reached[actor] = True
    fake_snap["reached"] = reached

    reach_value, reach_aux = _eval_snapshot_outputs(model, encode_fn, device, fake_snap, actor)
    reach_score = (
        float(policy_logits[action_to_idx(reach_action)])
        + beam_lambda * reach_value
        + _aux_bonus(reach_aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
    )

    value_scores: dict = {action_to_idx(reach_action): reach_score}
    best_score = reach_score
    best_action = reach_action
    for a in other_actions:
        s = float(policy_logits[action_to_idx(a)])
        value_scores[action_to_idx(a)] = s
        if s > best_score:
            best_score = s
            best_action = a

    return best_action, value_scores


class KeqingBot:
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
        # style_vec: [speed, riichi, value, defense]，均为 [-1, +1]，None 则全 0
        self.style_vec = (
            list(style_vec) if style_vec is not None else [0.0, 0.0, 0.0, 0.0]
        )
        # beam_k > 0 时，dahai 阶段对 top-k 候选做 value beam search
        self.beam_k = beam_k
        self.beam_lambda = beam_lambda
        self.score_delta_lambda = score_delta_lambda
        self.win_prob_lambda = win_prob_lambda
        self.dealin_prob_lambda = dealin_prob_lambda

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        c_tile = state_dict["input_proj.0.weight"].shape[1]
        n_scalar = state_dict["scalar_proj.0.weight"].shape[1]
        inferred_version = model_version or (
            "keqingv3" if (c_tile == 57 and n_scalar == 56) else "keqingv1"
        )
        if inferred_version == "keqingv3":
            features_mod = importlib.import_module("keqingv3.features")
            model_mod = importlib.import_module("keqingv3.model")
        else:
            features_mod = importlib.import_module("keqingv1.features")
            model_mod = importlib.import_module("keqingv1.model")
        self._model_version = inferred_version
        self._encode = features_mod.encode
        model_cls = model_mod.MahjongModel
        self.model = model_cls(
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            c_tile=c_tile,
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

        # tsumo：决策时手里有 14 张（apply 后），waits 应基于 14 张计算 → apply 后算
        # dahai 响应/副露/reach：决策时手牌尚未变化 → apply 前先记录手牌用于 waits 计算
        _pre_apply_hand: Optional[list] = None
        _pre_apply_melds: Optional[list] = None
        if etype == "tsumo" and event.get("actor") == actor:
            # 自家摸牌，需要弃牌/立直/自摸
            apply_event(state, event)
            needs_response = True
        elif etype == "dahai" and event.get("actor") != actor:
            # 他家弃牌，可能需要鸣牌/荣和/pass；决策时自家手牌未变，apply 前记录
            _pre_snap = state.snapshot(actor)
            _pre_apply_hand = _pre_snap.get("hand", [])
            _pre_apply_melds = (_pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
            needs_response = True
        elif etype == "kakan" and event.get("actor") != actor:
            _pre_snap = state.snapshot(actor)
            _pre_apply_hand = _pre_snap.get("hand", [])
            _pre_apply_melds = (_pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
            needs_response = True
        elif etype in ("chi", "pon", "daiminkan") and event.get("actor") == actor:
            # 自家副露后需要打牌；副露前手牌是决策时刻，apply 前记录
            _pre_snap = state.snapshot(actor)
            _pre_apply_hand = _pre_snap.get("hand", [])
            _pre_apply_melds = (_pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
            needs_response = True
        elif etype == "reach" and event.get("actor") == actor:
            # 自家立直宣告后需要打出立直宣言牌；宣告前手牌是决策时刻，apply 前记录
            _pre_snap = state.snapshot(actor)
            _pre_apply_hand = _pre_snap.get("hand", [])
            _pre_apply_melds = (_pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
            needs_response = True
        else:
            apply_event(state, event)
            return None

        # 先注入 shanten/waits，再枚举合法动作（shanten 影响 reach 是否合法）
        snap = state.snapshot(actor)
        injected = False
        if self._riichi_state is not None and _pre_apply_hand is None:
            # tsumo 场景：_riichi_state 在 apply 后已更新到 14 张状态，直接用
            try:
                snap["shanten"] = int(self._riichi_state.shanten)
                snap["waits_count"] = int(sum(self._riichi_state.waits))
                snap["waits_tiles"] = list(self._riichi_state.waits)
                injected = True
            except Exception:
                pass
        if not injected:
            # fallback 或非 tsumo 场景：用决策时刻手牌计算 waits
            from mahjong_env.replay import _calc_shanten_waits

            if _pre_apply_hand is not None:
                # dahai响应/副露/reach：用 apply 前手牌
                hand_list = _pre_apply_hand
                melds_list = _pre_apply_melds or []
            else:
                # tsumo fallback：apply 后 14 张
                hand_list = snap.get("hand", [])
                melds_list = (snap.get("melds") or [[], [], [], []])[actor]
            shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(
                hand_list, melds_list
            )
            snap["shanten"] = shanten
            snap["waits_count"] = waits_cnt
            snap["waits_tiles"] = waits_tiles
        # 计算并注入振听状态
        waits_tiles = snap.get("waits_tiles")
        p = state.players[actor]
        if waits_tiles is not None:
            from mahjong_env.tiles import tile_to_34 as _t34, normalize_tile as _norm

            wait_set = {i for i, w in enumerate(waits_tiles) if w}
            # 舍牌振听：自家舍牌中有进张牌
            actor_disc_set = {
                _t34(_norm(d["pai"] if isinstance(d, dict) else d))
                for d in state.players[actor].discards
            }
            p.sutehai_furiten = bool(wait_set & actor_disc_set)
            # 立直振听：立直后上一次摸到进张但打出去了（摸切了进张）
            if p.reached and state.last_tsumo[actor] is None:
                last_disc = p.discards[-1]["pai"] if p.discards else None
                if last_disc and _t34(_norm(last_disc)) in wait_set:
                    # 打出了进张牌（摸切或手切）
                    p.riichi_furiten = True
            p.furiten = p.sutehai_furiten or p.riichi_furiten or p.doujun_furiten
        snap["furiten"] = [pp.furiten for pp in state.players]
        snap["sutehai_furiten"] = [pp.sutehai_furiten for pp in state.players]
        snap["riichi_furiten"] = [pp.riichi_furiten for pp in state.players]
        snap["doujun_furiten"] = [pp.doujun_furiten for pp in state.players]
        legal_actions = enumerate_legal_actions(snap, actor)

        # 如果合法动作只有 none（无实质选择），直接返回 none 不做推理
        non_none = [a for a in legal_actions if a.type != "none"]
        if not needs_response or not legal_actions:
            return None
        if not non_none:
            return {"type": "none", "actor": actor}
        tile_feat, scalar = self._encode(snap, actor)

        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)  # (1, C, 34)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)  # (1, S)

        policy_logits, value_t = self.model(tile_t, scalar_t)
        logits_np = policy_logits.squeeze(0).cpu().numpy()  # (45,)
        value_scalar = float(value_t.squeeze().cpu().item())  # scalar
        aux_outputs = _get_aux_outputs(self.model)
        aux_bonus = _aux_bonus(
            aux_outputs,
            self.score_delta_lambda,
            self.win_prob_lambda,
            self.dealin_prob_lambda,
        )

        # Action dataclass → mjai dict
        legal_dicts = [a.to_mjai() for a in legal_actions]

        # 将非法动作设为 -1e9
        mask = np.array(build_legal_mask(legal_dicts), dtype=np.float32)
        logits_np = np.where(mask > 0, logits_np, -1e9)

        # beam search：分动作类型做 value head 重排
        legal_dahai = [a for a in legal_dicts if a.get("type") == "dahai"]
        legal_meld = [
            a
            for a in legal_dicts
            if a.get("type") in ("chi", "pon", "daiminkan", "ankan", "kakan")
        ]
        legal_reach = [a for a in legal_dicts if a.get("type") == "reach"]
        legal_none = [a for a in legal_dicts if a.get("type") == "none"]

        beam_value_scores: dict = {}  # action_idx -> beam combined score (for logging)
        if self.beam_k > 0 and legal_meld:
            # 副露决策：对 chi/pon/kan 做 value 评估，与 none 对比
            non_meld = [
                a
                for a in legal_dicts
                if a.get("type") not in ("chi", "pon", "daiminkan", "ankan", "kakan")
            ]
            chosen, beam_value_scores = _meld_value_eval(
                self.model,
                self._encode,
                self.device,
                snap,
                actor,
                logits_np,
                legal_meld,
                legal_none,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
            # meld beam 之外的合法动作（如 hora）仍需参与最终比较
            if non_meld:
                fallback = _find_best_legal(
                    logits_np,
                    non_meld,
                    value=value_scalar,
                    style_lambda=self.style_vec[0],
                    aux_bonus=aux_bonus,
                )
                if _legal_score(
                    logits_np,
                    fallback,
                    value=value_scalar,
                    style_lambda=self.style_vec[0],
                    aux_bonus=aux_bonus,
                ) > _legal_score(
                    logits_np,
                    chosen,
                    value=value_scalar,
                    style_lambda=self.style_vec[0],
                    aux_bonus=aux_bonus,
                ):
                    chosen = fallback
        elif self.beam_k > 0 and legal_reach:
            # 立直决策：对 reach 做 value 评估
            non_reach = [a for a in legal_dicts if a.get("type") != "reach"]
            chosen, beam_value_scores = _reach_value_eval(
                self.model,
                self._encode,
                self.device,
                snap,
                actor,
                logits_np,
                legal_reach[0],
                non_reach,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
        elif self.beam_k > 0 and len(legal_dahai) > 1:
            # 打牌决策：top-k dahai value beam search
            chosen, beam_value_scores = _dahai_beam_search(
                self.model,
                self._encode,
                self.device,
                snap,
                actor,
                logits_np,
                legal_dahai,
                beam_k=self.beam_k,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
        else:
            chosen = _find_best_legal(
                logits_np,
                legal_dicts,
                value=value_scalar,
                style_lambda=self.style_vec[0],
                aux_bonus=aux_bonus,
            )

        # 同巡振听：如果有 hora 选项但选了 none，则设置同巡振听
        if chosen.get("type") == "none":
            has_hora = any(a.get("type") == "hora" for a in legal_dicts)
            if has_hora:
                state.players[actor].doujun_furiten = True
                state.players[actor].furiten = True

        # 记录决策日志（供 HTML 导出）
        scored = sorted(
            [
                {
                    "action": a,
                    "logit": float(logits_np[action_to_idx(a)]),
                    **(
                        {"beam_score": float(beam_value_scores[action_to_idx(a)])}
                        if action_to_idx(a) in beam_value_scores
                        else {}
                    ),
                }
                for a in legal_dicts
            ],
            key=lambda x: x["logit"],
            reverse=True,
        )
        tsumo_pai = event.get("pai") if etype == "tsumo" else None
        self.decision_log.append(
            {
                "step": len(self.decision_log),
                "bakaze": snap.get("bakaze", ""),
                "kyoku": snap.get("kyoku", 0),
                "honba": snap.get("honba", 0),
                "oya": snap.get("oya", 0),
                "scores": snap.get("scores", []),
                "hand": snap.get("hand", []),
                "discards": snap.get("discards", []),
                "melds": snap.get("melds", []),
                "dora_markers": snap.get("dora_markers", []),
                "reached": snap.get("reached", []),
                "actor_to_move": snap.get(
                    "actor_to_move", event.get("actor", self.player_id)
                ),
                "last_discard": snap.get("last_discard"),
                "tsumo_pai": tsumo_pai,
                "candidates": scored,
                "chosen": chosen,
                "value": value_scalar,
                "aux_outputs": aux_outputs,
                "gt_action": gt_action,
            }
        )

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
