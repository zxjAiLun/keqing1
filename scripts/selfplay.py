#!/usr/bin/env python3
"""自对战脚本：加载共享运行时 bot 跑 N 局全Bot对战，统计顺位/胜率，可选保存牌谱/.npz。

用法:
    python scripts/selfplay.py --model best.pth --games 100
    python scripts/selfplay.py --model best.pth --games 1000 --save-games 10 --output-dir selfplay_out
    python scripts/selfplay.py --model best.pth --games 100 --save-all-games --save-npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gateway.battle import BattleConfig, BattleManager, BattleRoom, _shuffle_wall
from inference.bot_registry import create_runtime_bot
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState
from mahjong_env.tiles import normalize_tile


# ---------------------------------------------------------------------------
# 统计信息
# ---------------------------------------------------------------------------


class PlayerStatInfo:
    def __init__(self) -> None:
        self.rounds = 0
        self.wins = 0
        self.tsumo_wins = 0
        self.ron_wins = 0
        self.deal_ins = 0
        self.riichi_rounds = 0
        self.open_meld_rounds = 0
        self.ryukyoku_rounds = 0
        self.ryukyoku_tenpai_rounds = 0
        self.damaten_wins = 0
        self.total_win_points = 0
        self.total_deal_in_points = 0
        self.total_score = 0
        self.total_win_turns = 0
        self.yaku_counts: Counter = Counter()

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "wins": self.wins,
            "tsumo_wins": self.tsumo_wins,
            "ron_wins": self.ron_wins,
            "deal_ins": self.deal_ins,
            "riichi_rounds": self.riichi_rounds,
            "open_meld_rounds": self.open_meld_rounds,
            "ryukyoku_rounds": self.ryukyoku_rounds,
            "ryukyoku_tenpai_rounds": self.ryukyoku_tenpai_rounds,
            "damaten_wins": self.damaten_wins,
            "total_win_points": self.total_win_points,
            "total_deal_in_points": self.total_deal_in_points,
            "total_score": self.total_score,
            "total_win_turns": self.total_win_turns,
            "yaku_counts": dict(self.yaku_counts),
        }

    def record_round(
        self,
        *,
        actor: int,
        events: List[dict],
        final_score: int,
        turn_count: int,
        start_score: int = 25000,
    ) -> None:
        self.rounds += 1
        self.total_score += final_score

        riichi_declared = any(
            ev.get("type") == "reach" and ev.get("actor") == actor for ev in events
        )
        if riichi_declared:
            self.riichi_rounds += 1

        if any(
            ev.get("type") in ("chi", "pon", "daiminkan", "kakan")
            and ev.get("actor") == actor
            for ev in events
        ):
            self.open_meld_rounds += 1

        hora_ev = next((ev for ev in events if ev.get("type") == "hora"), None)
        if hora_ev is not None and hora_ev.get("actor") == actor:
            self.wins += 1
            self.total_win_points += int((hora_ev.get("deltas") or [0, 0, 0, 0])[actor])
            self.total_win_turns += turn_count
            if hora_ev.get("is_tsumo"):
                self.tsumo_wins += 1
            else:
                self.ron_wins += 1
            if not riichi_declared:
                self.damaten_wins += 1
            for yaku_key in _extract_counted_yaku_keys(hora_ev):
                self.yaku_counts[yaku_key] += 1

        if (
            hora_ev is not None
            and not hora_ev.get("is_tsumo")
            and hora_ev.get("target") == actor
        ):
            self.deal_ins += 1
            self.total_deal_in_points += -int(
                (hora_ev.get("deltas") or [0, 0, 0, 0])[actor]
            )

        ryukyoku_ev = next((ev for ev in events if ev.get("type") == "ryukyoku"), None)
        if ryukyoku_ev is not None:
            self.ryukyoku_rounds += 1
            deltas = ryukyoku_ev.get("deltas") or [0, 0, 0, 0]
            if actor < len(deltas) and int(deltas[actor]) == 0:
                self.ryukyoku_tenpai_rounds += 1

    def summary(self, *, total_rounds: int, start_score: int = 25000) -> dict:
        avg_score = self.total_score / max(1, self.rounds)
        avg_win_points = self.total_win_points / max(1, self.wins)
        avg_deal_in_points = self.total_deal_in_points / max(1, self.deal_ins)
        avg_win_turn = self.total_win_turns / max(1, self.wins)
        win_rate = self.wins / max(1, self.rounds)
        deal_in_rate = self.deal_ins / max(1, self.rounds)
        tsumo_rate = self.tsumo_wins / max(1, self.rounds)
        ron_rate = self.ron_wins / max(1, self.rounds)
        tsumo_share = self.tsumo_wins / max(1, self.wins)
        damaten_rate = self.damaten_wins / max(1, self.rounds)
        damaten_share = self.damaten_wins / max(1, self.wins)
        avg_round_delta = avg_score - start_score
        point_efficiency = win_rate * avg_win_points
        deal_in_loss = deal_in_rate * avg_deal_in_points

        return {
            **self.to_dict(),
            "win_rate": win_rate,
            "hora_rate": win_rate,
            "deal_in_rate": deal_in_rate,
            "tsumo_rate": tsumo_rate,
            "ron_rate": ron_rate,
            "tsumo_win_share": tsumo_share,
            "damaten_rate": damaten_rate,
            "damaten_win_share": damaten_share,
            "riichi_rate": self.riichi_rounds / max(1, self.rounds),
            "meld_rate": self.open_meld_rounds / max(1, self.rounds),
            "ryukyoku_rate": self.ryukyoku_rounds / max(1, self.rounds),
            "ryukyoku_tenpai_rate": self.ryukyoku_tenpai_rounds
            / max(1, self.ryukyoku_rounds),
            "avg_score": avg_score,
            "avg_round_delta": avg_round_delta,
            "avg_get_point": avg_win_points,
            "avg_win_points": avg_win_points,
            "avg_deal_in_point": avg_deal_in_points,
            "avg_loss_on_deal_in": avg_deal_in_points,
            "avg_daten": avg_win_points,
            "avg_win_turn": avg_win_turn,
            "point_efficiency": point_efficiency,
            "deal_in_loss": deal_in_loss,
            "net_point_efficiency": point_efficiency - deal_in_loss,
            "round_balance": avg_round_delta * self.rounds / max(1, total_rounds),
        }


_EXCLUDED_YAKU_KEYS = {
    "Dora",
    "Ura Dora",
    "Aka Dora",
    "Chankan",
    "Rinshan Kaihou",
    "Haitei",
    "Houtei",
    "Nagashi Mangan",
    "Tenhou",
    "Chiihou",
    "Kokushi Musou",
    "Kokushi Musou 13 Men",
    "Suuankou",
    "Suuankou Tanki",
    "Daisangen",
    "Shousuushii",
    "Daisuushii",
    "Tsuuiisou",
    "Chinroutou",
    "Ryuuiisou",
    "Chuuren Poutou",
    "Junsei Chuuren Poutou",
    "Suukantsu",
}


def _extract_counted_yaku_keys(hora_event: dict) -> List[str]:
    yaku_details = hora_event.get("yaku_details") or []
    keys: List[str] = []
    for detail in yaku_details:
        key = str(detail.get("key") or detail.get("name") or "").strip()
        if not key or key in _EXCLUDED_YAKU_KEYS:
            continue
        keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Bot 决策辅助
# ---------------------------------------------------------------------------


def _bot_decide(bot: object, event: dict) -> Optional[dict]:
    """将单条 mjai 事件喂给 bot，返回 bot 的响应动作（或 None）。"""
    return bot.react(event)


def _broadcast_event(
    bots: List[object],
    event: dict,
    *,
    skip: Optional[int] = None,
    decision_time_by_actor: Optional[Dict[str, float]] = None,
    decision_count_by_actor: Optional[Counter] = None,
) -> Dict[int, Optional[dict]]:
    """向 bots 广播一条事件，并返回各 bot 的单次 react 结果。"""
    responses: Dict[int, Optional[dict]] = {}
    for pid, bot in enumerate(bots):
        if pid == skip:
            continue
        action, elapsed = _timed_react(bot, event)
        if decision_time_by_actor is not None:
            decision_time_by_actor[str(pid)] += elapsed
        if decision_count_by_actor is not None:
            decision_count_by_actor[str(pid)] += 1
        responses[pid] = action
    return responses


def _events_since(room: BattleRoom, start_idx: int) -> List[dict]:
    return room.events[start_idx:]


def _apply_manager_action(
    manager: BattleManager,
    room: BattleRoom,
    actor: int,
    action: dict,
) -> List[dict]:
    """应用动作并返回本次新增的事件列表。"""
    action = _canonicalize_action_for_server(manager, room, actor, action) or action
    prev_events_len = len(room.events)
    atype = action.get("type", "none")

    if atype == "none":
        return []

    if atype == "dahai":
        pai = action.get("pai", "")
        tsumogiri = action.get("tsumogiri", False)
        manager.discard(room, actor, pai, tsumogiri=tsumogiri)
        return _events_since(room, prev_events_len)

    if atype == "reach":
        manager.reach(room, actor)
        return _events_since(room, prev_events_len)

    if atype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
        consumed = action.get("consumed", [])
        pai = action.get("pai", "")
        target = action.get("target", None)
        manager.handle_meld(room, atype, actor, pai, consumed, target=target)
        return _events_since(room, prev_events_len)

    if atype == "hora":
        target = action.get("target", actor)
        pai = action.get("pai", "")
        is_tsumo = target == actor
        manager.hora(room, actor, target, pai, is_tsumo=is_tsumo)
        return _events_since(room, prev_events_len)

    if atype == "ryukyoku":
        manager.ryukyoku(room)
        return _events_since(room, prev_events_len)

    return []


def _canonicalize_action_for_server(
    manager: BattleManager,
    room: BattleRoom,
    actor: int,
    action: Optional[dict],
) -> Optional[dict]:
    if not action:
        return action

    atype = action.get("type", "none")
    if atype in ("none", "reach", "hora", "ryukyoku"):
        return action

    try:
        snap = manager.get_snap_with_shanten(room, actor)
    except BaseException:
        return action
    legal = enumerate_legal_actions(snap, actor)

    if atype == "dahai":
        chosen_pai = action.get("pai", "")
        exact = next((a for a in legal if a.type == "dahai" and a.pai == chosen_pai), None)
        if exact is not None:
            return {
                "type": "dahai",
                "actor": actor,
                "pai": exact.pai,
                "tsumogiri": exact.tsumogiri,
            }
        normalized = next(
            (
                a
                for a in legal
                if a.type == "dahai"
                and normalize_tile(a.pai or "") == normalize_tile(chosen_pai)
            ),
            None,
        )
        if normalized is not None:
            return {
                "type": "dahai",
                "actor": actor,
                "pai": normalized.pai,
                "tsumogiri": normalized.tsumogiri,
            }
        return action

    if atype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
        chosen_pai = action.get("pai", "")
        chosen_target = action.get("target")
        chosen_consumed = action.get("consumed", []) or []
        chosen_consumed_norm = sorted(normalize_tile(t) for t in chosen_consumed)

        exact = next(
            (
                a
                for a in legal
                if a.type == atype
                and normalize_tile(a.pai or "") == normalize_tile(chosen_pai)
                and (chosen_target is None or a.target == chosen_target)
                and sorted(normalize_tile(t) for t in (a.consumed or []))
                == chosen_consumed_norm
            ),
            None,
        )
        if exact is None:
            exact = next(
                (
                    a
                    for a in legal
                    if a.type == atype
                    and normalize_tile(a.pai or "") == normalize_tile(chosen_pai)
                    and (chosen_target is None or a.target == chosen_target)
                ),
                None,
            )
        if exact is not None:
            return {
                "type": atype,
                "actor": actor,
                "pai": exact.pai,
                "consumed": list(exact.consumed or []),
                "target": exact.target,
            }

    return action


def _broadcast_events(
    bots: List[object],
    events: List[dict],
    *,
    skip: Optional[int] = None,
    decision_time_by_actor: Optional[Dict[str, float]] = None,
    decision_count_by_actor: Optional[Counter] = None,
) -> Dict[int, Optional[dict]]:
    """按顺序广播多条事件，返回最后一条事件的 react 结果。"""
    responses: Dict[int, Optional[dict]] = {}
    for ev in events:
        responses = _broadcast_event(
            bots,
            ev,
            skip=skip,
            decision_time_by_actor=decision_time_by_actor,
            decision_count_by_actor=decision_count_by_actor,
        )
    return responses


def _extract_event(events: List[dict], event_type: str) -> Optional[dict]:
    for ev in reversed(events):
        if ev.get("type") == event_type:
            return ev
    return None


def _action_name(action: Optional[dict]) -> str:
    if not action:
        return "none"
    return str(action.get("type", "none"))


def _response_priority(action: Optional[dict]) -> int:
    atype = _action_name(action)
    if atype == "hora":
        return 3
    if atype in ("pon", "daiminkan"):
        return 2
    if atype == "chi":
        return 1
    return 0


def _response_turn_distance(discarder: int, responder: int) -> int:
    return (responder - discarder) % 4


def _select_discard_response(
    discarder: int,
    responses: Dict[int, Optional[dict]],
) -> Tuple[Optional[int], Optional[dict]]:
    best_responder: Optional[int] = None
    best_action: Optional[dict] = None
    best_key: Optional[Tuple[int, int]] = None

    for responder_id, action in responses.items():
        priority = _response_priority(action)
        if priority <= 0:
            continue
        distance = _response_turn_distance(discarder, responder_id)
        if distance == 0:
            continue
        candidate_key = (priority, -distance)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_responder = responder_id
            best_action = action

    return best_responder, best_action


def _next_actor_after_meld_response(
    room: BattleRoom,
    responder_id: int,
    meld_type: str,
) -> int:
    if meld_type == "daiminkan":
        return responder_id
    return int(room.state.actor_to_move)


def _timed_react(bot: object, event: dict) -> Tuple[Optional[dict], float]:
    started = time.perf_counter()
    action = bot.react(event)
    return action, time.perf_counter() - started


def _default_model_label(model_path: str) -> str:
    lower = model_path.lower()
    if "keqingv1" in lower:
        return "keqingv1"
    if "keqingv2" in lower:
        return "keqingv2"
    return Path(model_path).stem


def _resolve_model_path(model_spec: str) -> str:
    path = Path(model_spec)
    if path.exists():
        return str(path)

    candidate = MODEL_ROOT / model_spec / "best.pth"
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"无法解析模型 {model_spec!r}；请传文件路径，或确保 {candidate} 存在"
    )


def _resolve_bot_source(bot_spec: str) -> Tuple[str, str]:
    if bot_spec == "rulebase":
        return "rulebase", "rulebase"
    resolved = _resolve_model_path(bot_spec)
    return resolved, _default_model_label(resolved)


def _default_output_dir(model_label: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path("artifacts/selfplay_benchmarks") / f"{model_label}_{timestamp}")


def _resolve_seat_models(
    model: str,
    seat_models: Optional[List[str]],
    seat_labels: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    if seat_models:
        if len(seat_models) != 4:
            raise ValueError("--seat-models 必须提供 4 个路径")
        paths = [_resolve_model_path(path) for path in seat_models]
    else:
        paths = [_resolve_model_path(model)] * 4

    if seat_labels:
        if len(seat_labels) != 4:
            raise ValueError("--seat-labels 必须提供 4 个标签")
        labels = seat_labels
    else:
        labels = [_default_model_label(path) for path in paths]

    return paths, labels


def _resolve_seat_bots(
    model: str,
    seat_bots: Optional[List[str]],
    seat_models: Optional[List[str]],
    seat_labels: Optional[List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    if seat_bots:
        if len(seat_bots) != 4:
            raise ValueError("--seat-bots 必须提供 4 个 bot 类型")
        sources: List[str] = []
        labels: List[str] = []
        kinds: List[str] = []
        for bot_spec in seat_bots:
            if bot_spec not in {"keqingv1", "keqingv2", "keqingv3", "rulebase"}:
                raise ValueError(
                    "--seat-bots 仅支持 keqingv1/keqingv2/keqingv3/rulebase"
                )
            source, default_label = _resolve_bot_source(bot_spec)
            sources.append(source)
            labels.append(default_label)
            kinds.append(bot_spec)
        if seat_labels:
            if len(seat_labels) != 4:
                raise ValueError("--seat-labels 必须提供 4 个标签")
            labels = seat_labels
        return sources, labels, kinds

    paths, labels = _resolve_seat_models(model, seat_models, seat_labels)
    return paths, labels, labels[:]


def _build_machine_names(model_labels: List[str]) -> List[str]:
    label_counts: Counter = Counter()
    machine_names: List[str] = []
    for label in model_labels:
        label_counts[label] += 1
        machine_names.append(f"{label}-{label_counts[label]}号机")
    return machine_names


def _seat_assignment(
    machine_count: int,
    *,
    fixed_seats: bool,
) -> List[int]:
    indices = list(range(machine_count))
    if not fixed_seats:
        random.shuffle(indices)
    return indices


def _abnormal_action_score(
    chosen_action_counts: Dict[str, int],
    fallback_counts: Dict[str, int],
    *,
    action_weights: Optional[Dict[str, float]] = None,
    fallback_weight: float = 10.0,
) -> Tuple[float, Dict[str, float]]:
    weights = action_weights or {
        "response_hora": 8.0,
        "response_daiminkan": 6.0,
        "response_kakan": 6.0,
        "response_ankan": 6.0,
        "response_pon": 3.0,
        "response_chi": 2.0,
        "daiminkan": 5.0,
        "ankan": 4.0,
        "kakan": 5.0,
        "pon": 2.0,
        "chi": 1.0,
        "reach": 1.0,
        "meld_followup_dahai": 1.0,
    }
    components: Dict[str, float] = {}
    for action_name, weight in weights.items():
        count = int(chosen_action_counts.get(action_name, 0))
        if count > 0:
            components[action_name] = count * weight
    for fallback_name, count in fallback_counts.items():
        count_i = int(count)
        if count_i > 0:
            components[f"fallback:{fallback_name}"] = count_i * fallback_weight
    return sum(components.values()), components


# ---------------------------------------------------------------------------
# 单局 / 整场运行
# ---------------------------------------------------------------------------

MAX_TURNS = 300  # 防死循环
MODEL_ROOT = Path("artifacts/models")


def run_one_kyoku(
    manager: BattleManager,
    room: BattleRoom,
    bots: List[object],
    seed: Optional[int] = None,
) -> dict:
    """跑完一局（从 start_kyoku 到 hora/ryukyoku），返回结果 dict。"""
    manager.start_kyoku(room, seed=seed)
    event_type_counts: Counter = Counter()
    chosen_action_counts: Counter = Counter()
    chosen_action_counts_by_actor: Dict[str, Counter] = defaultdict(Counter)
    fallback_counts: Counter = Counter()
    decision_time_by_actor: Dict[str, float] = defaultdict(float)
    decision_count_by_actor: Counter = Counter()

    # 重置所有 bot 状态，replay start_kyoku 事件
    for bot in bots:
        bot.game_state = GameState()
        bot.decision_log = []

    # 广播 start_game / start_kyoku 给所有 bot
    for ev in room.events:
        event_type_counts[ev.get("type", "unknown")] += 1
        _broadcast_event(
            bots,
            ev,
            decision_time_by_actor=decision_time_by_actor,
            decision_count_by_actor=decision_count_by_actor,
        )

    oya = room.state.oya
    actor = oya
    turns = 0

    while room.phase == "playing" and turns < MAX_TURNS:
        turns += 1

        # 摸牌
        tile = manager.draw(room, actor)

        if tile is None:
            # 牌山摸完，流局
            tenpai = []
            for pid in range(4):
                snap = room.state.snapshot(pid)
                from mahjong_env.replay import _calc_shanten_waits

                hand_list = snap.get("hand", [])
                melds_list = (snap.get("melds") or [[], [], [], []])[pid]
                shanten, _, _, _ = _calc_shanten_waits(hand_list, melds_list)
                if shanten <= 0:
                    tenpai.append(pid)
            new_events = _apply_manager_action(
                manager, room, actor, {"type": "ryukyoku", "actor": actor}
            )
            if not new_events:
                prev_events_len = len(room.events)
                manager.ryukyoku(room, tenpai=tenpai)
                new_events = _events_since(room, prev_events_len)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            _broadcast_events(bots, new_events)
            break

        tsumo_ev = room.events[-1]
        event_type_counts["tsumo"] += 1

        # actor bot 响应摸牌
        action, elapsed = _timed_react(bots[actor], tsumo_ev)
        decision_time_by_actor[str(actor)] += elapsed
        decision_count_by_actor[str(actor)] += 1

        # 其他 bot 也收到 tsumo 事件（但不能响应）
        _broadcast_event(
            bots,
            tsumo_ev,
            skip=actor,
            decision_time_by_actor=decision_time_by_actor,
            decision_count_by_actor=decision_count_by_actor,
        )

        if action is None:
            fallback_counts["tsumo_none_to_dahai"] += 1
            action = {"type": "dahai", "actor": actor, "pai": tile}
        action = _canonicalize_action_for_server(manager, room, actor, action)
        chosen_action_counts[_action_name(action)] += 1
        chosen_action_counts_by_actor[str(actor)][_action_name(action)] += 1

        atype = action.get("type", "none")

        # hora（自摸）
        if atype == "hora":
            new_events = _apply_manager_action(manager, room, actor, action)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            _broadcast_events(
                bots,
                new_events,
                decision_time_by_actor=decision_time_by_actor,
                decision_count_by_actor=decision_count_by_actor,
            )
            break

        # 暗杠 / 加杠：继续保留当前 actor，后续可能岭上摸牌
        if atype in ("ankan", "kakan"):
            new_events = _apply_manager_action(manager, room, actor, action)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            kakan_responses = _broadcast_events(
                bots,
                new_events,
                decision_time_by_actor=decision_time_by_actor,
                decision_count_by_actor=decision_count_by_actor,
            )

            if atype == "kakan":
                chankan_candidates: Dict[int, Optional[dict]] = {}
                for pid, resp in kakan_responses.items():
                    if pid == actor or resp is None:
                        continue
                    rtype = resp.get("type", "none")
                    if rtype != "none":
                        resp = _canonicalize_action_for_server(manager, room, pid, resp)
                        rtype = resp.get("type", "none")
                    chosen_action_counts["response_" + rtype] += 1
                    chosen_action_counts_by_actor[str(pid)]["response_" + rtype] += 1
                    if rtype == "hora":
                        chankan_candidates[pid] = resp

                responder_id, resp = _select_discard_response(actor, chankan_candidates)
                if responder_id is not None and resp is not None:
                    manager.cancel_kakan(room)
                    new_resp_events = _apply_manager_action(manager, room, responder_id, resp)
                    event_type_counts.update(
                        ev.get("type", "unknown") for ev in new_resp_events
                    )
                    _broadcast_events(
                        bots,
                        new_resp_events,
                        decision_time_by_actor=decision_time_by_actor,
                        decision_count_by_actor=decision_count_by_actor,
                    )
                    break

                accepted_events = manager.accept_kakan(room)
                event_type_counts.update(ev.get("type", "unknown") for ev in accepted_events)
                _broadcast_events(
                    bots,
                    accepted_events,
                    decision_time_by_actor=decision_time_by_actor,
                    decision_count_by_actor=decision_count_by_actor,
                )
            continue

        # ryukyoku（九种九牌）
        if atype == "ryukyoku":
            new_events = _apply_manager_action(manager, room, actor, action)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            _broadcast_events(
                bots,
                new_events,
                decision_time_by_actor=decision_time_by_actor,
                decision_count_by_actor=decision_count_by_actor,
            )
            break

        # reach
        if atype == "reach":
            new_events = _apply_manager_action(manager, room, actor, action)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            reach_ev = _extract_event(new_events, "reach")
            reach_responses = _broadcast_events(
                bots,
                [reach_ev] if reach_ev is not None else [],
                decision_time_by_actor=decision_time_by_actor,
                decision_count_by_actor=decision_count_by_actor,
            )
            # reach 后继续打牌
            action = reach_responses.get(actor)
            if action is None or action.get("type") != "dahai":
                fallback_counts["reach_followup_to_dahai"] += 1
                hand = list(room.state.players[actor].hand.keys())
                action = {
                    "type": "dahai",
                    "actor": actor,
                    "pai": hand[0] if hand else tile,
                }
            action = _canonicalize_action_for_server(manager, room, actor, action)
            chosen_action_counts["reach_followup_" + _action_name(action)] += 1
            chosen_action_counts_by_actor[str(actor)][
                "reach_followup_" + _action_name(action)
            ] += 1

        # dahai（含 reach 后打牌）
        if action.get("type") == "dahai":
            new_events = _apply_manager_action(manager, room, actor, action)
            event_type_counts.update(ev.get("type", "unknown") for ev in new_events)
            dahai_ev = _extract_event(new_events, "dahai")
            if dahai_ev is None:
                continue

            # 广播 dahai 给所有 bot
            responses = _broadcast_event(
                bots,
                dahai_ev,
                decision_time_by_actor=decision_time_by_actor,
                decision_count_by_actor=decision_count_by_actor,
            )

            # 其他玩家响应（副露/荣和）
            response_candidates: Dict[int, Optional[dict]] = {}
            responded = False
            for offset in range(1, 4):
                responder_id = (actor + offset) % 4
                resp = responses.get(responder_id)
                if resp is None:
                    continue
                rtype = resp.get("type", "none")
                if rtype != "none":
                    resp = _canonicalize_action_for_server(
                        manager, room, responder_id, resp
                    )
                    rtype = resp.get("type", "none")
                chosen_action_counts["response_" + rtype] += 1
                chosen_action_counts_by_actor[str(responder_id)]["response_" + rtype] += 1
                if rtype == "none":
                    continue
                response_candidates[responder_id] = resp

            responder_id, resp = _select_discard_response(actor, response_candidates)
            if responder_id is not None and resp is not None:
                rtype = resp.get("type", "none")
                if rtype == "hora":
                    new_resp_events = _apply_manager_action(manager, room, responder_id, resp)
                    event_type_counts.update(
                        ev.get("type", "unknown") for ev in new_resp_events
                    )
                    _broadcast_events(
                        bots,
                        new_resp_events,
                        decision_time_by_actor=decision_time_by_actor,
                        decision_count_by_actor=decision_count_by_actor,
                    )
                    responded = True
                elif rtype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
                    new_resp_events = _apply_manager_action(
                        manager, room, responder_id, resp
                    )
                    event_type_counts.update(
                        ev.get("type", "unknown") for ev in new_resp_events
                    )
                    meld_responses = _broadcast_events(
                        bots,
                        new_resp_events,
                        decision_time_by_actor=decision_time_by_actor,
                        decision_count_by_actor=decision_count_by_actor,
                    )

                    if rtype in ("chi", "pon"):
                        meld_action = meld_responses.get(responder_id)
                        if meld_action is None or meld_action.get("type") != "dahai":
                            fallback_counts["meld_followup_to_dahai"] += 1
                            hand = list(room.state.players[responder_id].hand.keys())
                            meld_action = {
                                "type": "dahai",
                                "actor": responder_id,
                                "pai": hand[0] if hand else "",
                            }
                        meld_action = _canonicalize_action_for_server(
                            manager, room, responder_id, meld_action
                        )
                        chosen_action_counts["meld_followup_" + _action_name(meld_action)] += 1
                        chosen_action_counts_by_actor[str(responder_id)][
                            "meld_followup_" + _action_name(meld_action)
                        ] += 1
                        meld_follow_events = _apply_manager_action(
                            manager, room, responder_id, meld_action
                        )
                        event_type_counts.update(
                            ev.get("type", "unknown") for ev in meld_follow_events
                        )
                        _broadcast_events(
                            bots,
                            meld_follow_events,
                            decision_time_by_actor=decision_time_by_actor,
                            decision_count_by_actor=decision_count_by_actor,
                        )

                    actor = _next_actor_after_meld_response(room, responder_id, rtype)
                    responded = True

            if room.phase == "ended":
                break
            meta_events = [ev for ev in new_events if ev.get("type") != "dahai"]
            if meta_events:
                _broadcast_events(
                    bots,
                    meta_events,
                    decision_time_by_actor=decision_time_by_actor,
                    decision_count_by_actor=decision_count_by_actor,
                )
            if not responded:
                actor = (actor + 1) % 4

    return {
        "scores": room.state.scores[:],
        "events": list(room.events),
        "turns": turns,
        "phase": room.phase,
        "event_type_counts": dict(event_type_counts),
        "chosen_action_counts": dict(chosen_action_counts),
        "chosen_action_counts_by_actor": {
            actor_id: dict(counts)
            for actor_id, counts in chosen_action_counts_by_actor.items()
        },
        "fallback_counts": dict(fallback_counts),
        "decision_latency": {
            "total_seconds": sum(decision_time_by_actor.values()),
            "decision_count": sum(decision_count_by_actor.values()),
            "avg_seconds": (
                sum(decision_time_by_actor.values())
                / max(1, sum(decision_count_by_actor.values()))
            ),
            "by_actor": {
                actor_id: {
                    "total_seconds": decision_time_by_actor[actor_id],
                    "decision_count": int(decision_count_by_actor[actor_id]),
                    "avg_seconds": (
                        decision_time_by_actor[actor_id]
                        / max(1, decision_count_by_actor[actor_id])
                    ),
                }
                for actor_id in sorted(decision_count_by_actor)
            },
        },
    }


def _merge_counter_dict(
    dst: Dict[str, Counter],
    src: Dict[str, Dict[str, int]],
) -> None:
    for actor_id, counts in src.items():
        dst[str(actor_id)].update(counts)


def run_one_game(
    manager: BattleManager,
    room: BattleRoom,
    bots: List[object],
    seed: Optional[int] = None,
) -> dict:
    kyoku_results: List[dict] = []
    round_results: List[dict] = []
    aggregate_event_counts: Counter = Counter()
    aggregate_action_counts: Counter = Counter()
    aggregate_action_counts_by_actor: Dict[str, Counter] = defaultdict(Counter)
    aggregate_fallback_counts: Counter = Counter()
    aggregate_decision_seconds_by_actor: Dict[str, float] = defaultdict(float)
    aggregate_decision_counts_by_actor: Counter = Counter()
    total_turns = 0
    round_idx = 0

    while True:
        kyoku_seed = None if seed is None else seed + round_idx
        result = run_one_kyoku(manager, room, bots, seed=kyoku_seed)
        total_turns += int(result.get("turns", 0))
        aggregate_event_counts.update(result.get("event_type_counts", {}))
        aggregate_action_counts.update(result.get("chosen_action_counts", {}))
        aggregate_fallback_counts.update(result.get("fallback_counts", {}))
        _merge_counter_dict(
            aggregate_action_counts_by_actor,
            result.get("chosen_action_counts_by_actor", {}),
        )

        decision_latency = result.get("decision_latency", {})
        for actor_id, info in decision_latency.get("by_actor", {}).items():
            aggregate_decision_seconds_by_actor[str(actor_id)] += float(
                info.get("total_seconds", 0.0)
            )
            aggregate_decision_counts_by_actor[str(actor_id)] += int(
                info.get("decision_count", 0)
            )

        events = list(result.get("events", []))
        start_idx = 0
        if kyoku_results:
            start_idx = kyoku_results[-1]["event_end_index"]
        new_events = events[start_idx:]
        kyoku_results.append(
            {
                "bakaze": room.state.bakaze,
                "kyoku": room.state.kyoku,
                "honba": room.state.honba,
                "scores": result["scores"],
                "turns": result.get("turns", 0),
                "event_start_index": start_idx,
                "event_end_index": len(events),
                "events": new_events,
            }
        )

        round_results.append(
            {
                "scores": result["scores"],
                "events": new_events,
                "turns": result.get("turns", 0),
            }
        )

        round_idx += 1
        if manager.is_game_ended(room):
            room.events.append({"type": "end_game", "scores": room.state.scores[:]})
            break
        if not manager.next_kyoku(room):
            room.events.append({"type": "end_game", "scores": room.state.scores[:]})
            break

    return {
        "scores": room.state.scores[:],
        "events": list(room.events),
        "turns": total_turns,
        "rounds": len(round_results),
        "kyoku_results": kyoku_results,
        "round_results": round_results,
        "event_type_counts": dict(aggregate_event_counts),
        "chosen_action_counts": dict(aggregate_action_counts),
        "chosen_action_counts_by_actor": {
            actor_id: dict(counts)
            for actor_id, counts in aggregate_action_counts_by_actor.items()
        },
        "fallback_counts": dict(aggregate_fallback_counts),
        "decision_latency": {
            "total_seconds": sum(aggregate_decision_seconds_by_actor.values()),
            "decision_count": sum(aggregate_decision_counts_by_actor.values()),
            "avg_seconds": (
                sum(aggregate_decision_seconds_by_actor.values())
                / max(1, sum(aggregate_decision_counts_by_actor.values()))
            ),
            "by_actor": {
                actor_id: {
                    "total_seconds": aggregate_decision_seconds_by_actor[actor_id],
                    "decision_count": int(aggregate_decision_counts_by_actor[actor_id]),
                    "avg_seconds": (
                        aggregate_decision_seconds_by_actor[actor_id]
                        / max(1, aggregate_decision_counts_by_actor[actor_id])
                    ),
                }
                for actor_id in sorted(aggregate_decision_counts_by_actor)
            },
        },
    }


# ---------------------------------------------------------------------------
# NPZ / 牌谱 保存
# ---------------------------------------------------------------------------


def _events_to_npz(
    events: list,
    *,
    adapter_name: str = "base",
    value_strategy: str = "heuristic",
    encode_module: str = "keqingv1.features",
) -> Optional[Dict[str, np.ndarray]]:
    """从 events 生成 preprocess-owned cache arrays."""
    try:
        from training.preprocess import create_preprocess_adapter, events_to_cached_arrays

        return events_to_cached_arrays(
            events,
            adapter=create_preprocess_adapter(adapter_name),
            value_strategy=value_strategy,
            encode_module=encode_module,
        )
    except Exception:
        return None


def _save_npz(
    path: Path,
    events: list,
    *,
    adapter_name: str = "base",
    value_strategy: str = "heuristic",
    encode_module: str = "keqingv1.features",
) -> bool:
    try:
        from training.preprocess import save_events_to_cache_file

        return save_events_to_cache_file(
            path,
            events,
            adapter_name=adapter_name,
            value_strategy=value_strategy,
            encode_module=encode_module,
        )
    except Exception:
        return False


def _build_cache_export_metadata(
    *,
    adapter_name: str,
    value_strategy: str,
    encode_module: str,
    saved_games: int,
    output_dir: Path,
) -> dict:
    recommended_output_dir = _recommended_preprocess_output_dir(output_dir.parent, adapter_name)
    return {
        "format": "preprocess_cache_wrapper",
        "owner": "training.preprocess",
        "status": "legacy_convenience_export",
        "recommended_flow": [
            "运行 selfplay 保存 .mjson",
            "使用 src/training/preprocess.py 统一离线导出 cache",
            "训练侧从 training.cached_dataset 读取 .npz",
        ],
        "recommended_replays_dir": str(output_dir.parent / "replays"),
        "recommended_preprocess_output_dir": str(recommended_output_dir),
        "recommended_preprocess_command": _build_recommended_preprocess_command(
            output_dir.parent,
            adapter_name,
        ),
        "adapter_name": adapter_name,
        "value_strategy": value_strategy,
        "encode_module": encode_module,
        "saved_games": saved_games,
        "output_dir": str(output_dir),
    }


def _write_cache_export_metadata(npz_dir: Path, metadata: dict) -> None:
    with open(npz_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _recommended_preprocess_output_dir(output_dir: Path, adapter_name: str) -> Path:
    return output_dir / f"preprocessed_{adapter_name}"


def _build_recommended_preprocess_command(output_dir: Path, adapter_name: str) -> str:
    replays_dir = output_dir / "replays"
    processed_dir = _recommended_preprocess_output_dir(output_dir, adapter_name)
    return (
        "uv run python -c "
        "\"from training.preprocess import create_preprocess_adapter, run_preprocess; "
        f"run_preprocess(default_output_dir='{processed_dir}', "
        f"adapter=create_preprocess_adapter('{adapter_name}'))\" "
        f"--data_dirs {replays_dir}"
    )


def _save_mjson(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _extract_player_names(events: List[dict]) -> Optional[List[str]]:
    for ev in events:
        if ev.get("type") == "start_game" and isinstance(ev.get("names"), list):
            return [str(name) for name in ev["names"]]
    return None


def _infer_replay_bot_type(model_path: str) -> str:
    lower = model_path.lower()
    if "keqingv1" in lower:
        return "keqingv1"
    if "v5" in lower:
        return "v5"
    return "keqingv2"


def _persist_replay_ui_artifact(
    events: List[dict],
    *,
    model_path: str,
    player_id: int = 0,
    bot_type: Optional[str] = None,
) -> dict:
    from replay.api import run_replay_single_raw
    from replay.bot import render_replay_json
    from replay.storage import get_storage

    replay_bot_type = bot_type or _infer_replay_bot_type(model_path)
    bot = run_replay_single_raw(
        events,
        player_id=player_id,
        checkpoint=model_path,
        input_type="url",
        bot_type=replay_bot_type,
    )
    decisions = render_replay_json(bot)
    player_names = _extract_player_names(events) or decisions.get("player_names") or None
    replay_id = get_storage().save(
        events=events,
        decisions=decisions,
        bot_type=replay_bot_type,
        player_names=player_names,
    )
    return {
        "replay_id": replay_id,
        "replay_player_id": player_id,
        "replay_bot_type": replay_bot_type,
        "replay_view_url": f"/replay?id={replay_id}",
        "game_board_url": f"/game-replay?id={replay_id}",
    }


def _save_error_artifacts(
    output_dir: Path,
    game_id: int,
    room: BattleRoom,
    exc: BaseException,
    bots: List[object],
) -> Dict[str, str]:
    err_dir = output_dir / "errors"
    err_dir.mkdir(parents=True, exist_ok=True)

    mjson_path = err_dir / f"game_{game_id:05d}.mjson"
    meta_path = err_dir / f"game_{game_id:05d}.json"

    _save_mjson(mjson_path, room.events)
    meta = {
        "game_id": game_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "event_count": len(room.events),
        "server_state": room.state.snapshot(0),
        "tail_events": room.events[-10:],
        "decision_log_tail": {
            str(bot.player_id): bot.decision_log[-5:] for bot in bots
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"mjson": str(mjson_path), "meta": str(meta_path)}


def _save_anomaly_samples(
    output_dir: Path,
    game_results: List[dict],
    top_n: int,
    *,
    min_score: float,
    replay_model_path: Optional[str] = None,
    replay_player_id: int = 0,
    replay_bot_type: Optional[str] = None,
) -> List[dict]:
    if top_n <= 0:
        return []

    ranked = []
    for result in game_results:
        score, components = _abnormal_action_score(
            result.get("chosen_action_counts", {}),
            result.get("fallback_counts", {}),
        )
        if score < min_score:
            continue
        ranked.append(
            {
                "game_id": result["game_id"],
                "score": score,
                "components": components,
                "turns": result.get("turns", 0),
                "events": result["events"],
            }
        )

    ranked.sort(key=lambda x: (x["score"], x["turns"]), reverse=True)
    selected = ranked[:top_n]
    if not selected:
        return []

    export_dir = output_dir / "anomaly_replays"
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in selected:
        mjson_path = export_dir / f"game_{item['game_id']:05d}.mjson"
        meta_path = export_dir / f"game_{item['game_id']:05d}.json"
        _save_mjson(mjson_path, item["events"])
        replay_ui = None
        replay_ui_error = None
        if replay_model_path:
            try:
                replay_ui = _persist_replay_ui_artifact(
                    item["events"],
                    model_path=replay_model_path,
                    player_id=replay_player_id,
                    bot_type=replay_bot_type,
                )
            except Exception as exc:
                replay_ui_error = f"{type(exc).__name__}: {exc}"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "game_id": item["game_id"],
                    "anomaly_score": item["score"],
                    "score_components": item["components"],
                    "turns": item["turns"],
                    "replay_ui": replay_ui,
                    "replay_ui_error": replay_ui_error,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        manifest_item = {
            "game_id": item["game_id"],
            "anomaly_score": item["score"],
            "score_components": item["components"],
            "mjson": str(mjson_path),
            "meta": str(meta_path),
        }
        if replay_ui is not None:
            manifest_item.update(replay_ui)
        if replay_ui_error is not None:
            manifest_item["replay_ui_error"] = replay_ui_error
        manifest.append(manifest_item)

    with open(export_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _save_replay_samples(
    output_dir: Path,
    game_results: List[dict],
    top_n: int,
    *,
    save_all: bool = False,
    replay_model_path: Optional[str] = None,
    replay_player_id: int = 0,
    replay_bot_type: Optional[str] = None,
) -> List[dict]:
    if top_n <= 0 and not save_all:
        return []

    export_dir = output_dir / "replays"
    export_dir.mkdir(parents=True, exist_ok=True)

    if save_all:
        selected = list(game_results)
    else:
        selected = sorted(game_results, key=lambda x: x["interest"], reverse=True)[:top_n]

    manifest = []
    for item in selected:
        mjson_path = export_dir / f"game_{item['game_id']:05d}.mjson"
        meta_path = export_dir / f"game_{item['game_id']:05d}.json"
        _save_mjson(mjson_path, item["events"])

        replay_ui = None
        replay_ui_error = None
        if replay_model_path:
            try:
                replay_ui = _persist_replay_ui_artifact(
                    item["events"],
                    model_path=replay_model_path,
                    player_id=replay_player_id,
                    bot_type=replay_bot_type,
                )
            except Exception as exc:
                replay_ui_error = f"{type(exc).__name__}: {exc}"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "game_id": item["game_id"],
                    "interest": item["interest"],
                    "scores": item.get("scores"),
                    "ranks": item.get("ranks"),
                    "turns": item.get("turns", 0),
                    "rounds": item.get("rounds", 1),
                    "replay_ui": replay_ui,
                    "replay_ui_error": replay_ui_error,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        manifest_item = {
            "game_id": item["game_id"],
            "interest": item["interest"],
            "scores": item.get("scores"),
            "ranks": item.get("ranks"),
            "turns": item.get("turns", 0),
            "rounds": item.get("rounds", 1),
            "mjson": str(mjson_path),
            "meta": str(meta_path),
        }
        if replay_ui is not None:
            manifest_item.update(replay_ui)
        if replay_ui_error is not None:
            manifest_item["replay_ui_error"] = replay_ui_error
        manifest.append(manifest_item)

    with open(export_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
# 多局统计
# ---------------------------------------------------------------------------


def _rank_of_scores(scores: List[int]) -> List[int]:
    """返回每个 player 的顺位（1-4），同分时座位靠前者优先（日麻规则）。"""
    sorted_scores = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    ranks = [0] * 4
    for rank, (pid, _sc) in enumerate(sorted_scores, start=1):
        ranks[pid] = rank
    return ranks


def _score_interest(events: list, scores: List[int]) -> float:
    """为牌谱评分，用于筛选'值得保存'的局。分越高越值得看。"""
    # 指标：有荣和/自摸事件 + 最终得分差距大
    hora_count = sum(1 for e in events if e.get("type") == "hora")
    score_spread = max(scores) - min(scores)
    return hora_count * 1000 + score_spread


def _build_label_summary(
    model_rank_counts: Dict[str, Counter],
    model_total_scores: Dict[str, int],
    model_games: Counter,
    model_stat_infos: Dict[str, PlayerStatInfo],
    model_decision_seconds: Dict[str, float],
    model_decision_counts: Counter,
) -> dict[str, dict]:
    return {
        model_label: {
            "games": int(model_games[model_label]),
            "avg_rank": (
                sum(
                    rank * count
                    for rank, count in model_rank_counts[model_label].items()
                )
                / max(1, model_games[model_label])
            ),
            "avg_score": model_total_scores[model_label]
            / max(1, model_games[model_label]),
            "rank_counts": dict(model_rank_counts[model_label]),
            "top1_rate": model_rank_counts[model_label][1]
            / max(1, model_games[model_label]),
            "top4_rate": model_rank_counts[model_label][4]
            / max(1, model_games[model_label]),
            "yaku_counts": dict(model_stat_infos[model_label].yaku_counts),
            "decision_total_seconds": model_decision_seconds[model_label],
            "decision_count": int(model_decision_counts[model_label]),
            "avg_decision_seconds": model_decision_seconds[model_label]
            / max(1, model_decision_counts[model_label]),
        }
        for model_label in sorted(model_games)
    }


def _format_progress_label_summary(label_summary: dict[str, dict]) -> str:
    parts = []
    for model_label, summary in label_summary.items():
        parts.append(
            f"{model_label}: avg_rank={summary['avg_rank']:.3f} "
            f"avg_score={summary['avg_score']:.1f} "
            f"top1={summary['top1_rate'] * 100:.1f}%"
        )
    return " | ".join(parts)


def _format_progress_benchmark_side_summary(
    benchmark_side_summary: dict[str, dict],
) -> str:
    if not benchmark_side_summary:
        return ""
    solo = benchmark_side_summary.get("solo_side")
    trio = benchmark_side_summary.get("trio_side")
    if not solo or not trio:
        return ""
    return (
        f"单机侧 {solo['label']}: avg_rank={solo['avg_rank']:.3f} avg_score={solo['avg_score']:.1f} "
        f"| 三机侧 {trio['label']}: avg_rank={trio['avg_rank']:.3f} avg_score={trio['avg_score']:.1f}"
    )


def _build_benchmark_side_summary(
    seat_model_labels: List[str],
    label_summary: dict[str, dict],
    model_stats: dict[str, dict],
) -> dict[str, dict]:
    label_counts = Counter(seat_model_labels)
    if len(label_counts) != 2:
        return {}

    solo_items = [item for item in label_counts.items() if item[1] == 1]
    trio_items = [item for item in label_counts.items() if item[1] == 3]
    if len(solo_items) != 1 or len(trio_items) != 1:
        return {}

    solo_label = solo_items[0][0]
    trio_label = trio_items[0][0]
    solo_summary = label_summary.get(solo_label)
    trio_summary = label_summary.get(trio_label)
    solo_stats = model_stats.get(solo_label, {})
    trio_stats = model_stats.get(trio_label, {})
    if not solo_summary or not trio_summary:
        return {}

    return {
        "solo_side": {
            "label": solo_label,
            "count": 1,
            "avg_rank": solo_summary.get("avg_rank", 0.0),
            "avg_score": solo_summary.get("avg_score", 0.0),
            "top1_rate": solo_summary.get("top1_rate", 0.0),
            "hora_rate": solo_stats.get("hora_rate", 0.0),
            "deal_in_rate": solo_stats.get("deal_in_rate", 0.0),
            "riichi_rate": solo_stats.get("riichi_rate", 0.0),
            "meld_rate": solo_stats.get("meld_rate", 0.0),
        },
        "trio_side": {
            "label": trio_label,
            "count": 3,
            "avg_rank": trio_summary.get("avg_rank", 0.0),
            "avg_score": trio_summary.get("avg_score", 0.0),
            "top1_rate": trio_summary.get("top1_rate", 0.0),
            "hora_rate": trio_stats.get("hora_rate", 0.0),
            "deal_in_rate": trio_stats.get("deal_in_rate", 0.0),
            "riichi_rate": trio_stats.get("riichi_rate", 0.0),
            "meld_rate": trio_stats.get("meld_rate", 0.0),
        },
    }


def _build_stats_snapshot(
    *,
    args,
    seat_model_paths: List[str],
    seat_model_labels: List[str],
    rank_counts,
    total_scores: List[int],
    model_rank_counts: Dict[str, Counter],
    model_total_scores: Dict[str, int],
    model_games: Counter,
    player_stat_infos: Dict[str, PlayerStatInfo],
    model_stat_infos: Dict[str, PlayerStatInfo],
    model_decision_seconds: Dict[str, float],
    model_decision_counts: Counter,
    total_rounds_played: int,
    hora_total: int,
    total_turns: int,
    aggregate_event_counts: Counter,
    aggregate_action_counts: Counter,
    aggregate_action_counts_by_actor: Dict[str, Counter],
    aggregate_fallback_counts: Counter,
    aggregate_decision_seconds_by_actor: Dict[str, float],
    aggregate_decision_counts_by_actor: Counter,
    replay_manifest: List[dict],
    anomaly_manifest: List[dict],
    error_games: List[dict],
    elapsed: float,
    completed_games: int,
    cache_export: Optional[dict],
) -> dict:
    total_decision_seconds = sum(aggregate_decision_seconds_by_actor.values())
    total_decision_count = sum(aggregate_decision_counts_by_actor.values())
    label_summary = _build_label_summary(
        model_rank_counts=model_rank_counts,
        model_total_scores=model_total_scores,
        model_games=model_games,
        model_stat_infos=model_stat_infos,
        model_decision_seconds=model_decision_seconds,
        model_decision_counts=model_decision_counts,
    )
    model_stats = {
        model_label: model_stat_infos[model_label].summary(
            total_rounds=total_rounds_played
        )
        for model_label in sorted(model_stat_infos)
    }
    benchmark_side_summary = _build_benchmark_side_summary(
        seat_model_labels=seat_model_labels,
        label_summary=label_summary,
        model_stats=model_stats,
    )
    return {
        "games": args.games,
        "completed_games": completed_games,
        "game_format": args.game_format,
        "total_rounds": total_rounds_played,
        "error_games": error_games,
        "seat_model_paths": seat_model_paths,
        "seat_model_labels": seat_model_labels,
        "rank_counts": dict(rank_counts),
        "avg_scores": [s / max(1, completed_games) for s in total_scores],
        "model_summary": label_summary,
        "player_stats": {
            actor_id: player_stat_infos[actor_id].summary(total_rounds=total_rounds_played)
            for actor_id in sorted(player_stat_infos)
        },
        "model_stats": model_stats,
        "benchmark_side_summary": benchmark_side_summary,
        "hora_per_game": hora_total / max(1, completed_games),
        "avg_turns": total_turns / max(1, completed_games),
        "event_type_counts": dict(aggregate_event_counts),
        "chosen_action_counts": dict(aggregate_action_counts),
        "chosen_action_counts_by_actor": {
            actor_id: dict(counts)
            for actor_id, counts in aggregate_action_counts_by_actor.items()
        },
        "fallback_counts": dict(aggregate_fallback_counts),
        "decision_latency": {
            "total_seconds": total_decision_seconds,
            "decision_count": total_decision_count,
            "avg_seconds": total_decision_seconds / max(1, total_decision_count),
            "by_actor": {
                actor_id: {
                    "total_seconds": aggregate_decision_seconds_by_actor[actor_id],
                    "decision_count": int(aggregate_decision_counts_by_actor[actor_id]),
                    "avg_seconds": (
                        aggregate_decision_seconds_by_actor[actor_id]
                        / max(1, aggregate_decision_counts_by_actor[actor_id])
                    ),
                }
                for actor_id in sorted(aggregate_decision_counts_by_actor)
            },
        },
        "replay_exports": replay_manifest,
        "anomaly_exports": anomaly_manifest,
        "cache_export": cache_export,
        "elapsed_seconds": elapsed,
        "seconds_per_game": elapsed / max(1, completed_games),
    }


def _write_stats_snapshot(output_dir: Path, stats: dict) -> None:
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def _append_progress_line(output_dir: Path, record: dict) -> None:
    with open(output_dir / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="RuntimeBot 全Bot自对战")
    p.add_argument(
        "--model",
        required=True,
        help="模型名或模型权重路径；如 keqingv2 将自动解析到 artifacts/models/keqingv2/best.pth",
    )
    p.add_argument(
        "--seat-models",
        nargs=4,
        default=None,
        metavar=("P0", "P1", "P2", "P3"),
        help="按座位指定 4 个模型路径；未指定时四家共用 --model",
    )
    p.add_argument(
        "--seat-bots",
        nargs=4,
        default=None,
        metavar=("B0", "B1", "B2", "B3"),
        help="按座位指定 4 个 bot；可混用 keqingv1/keqingv2/keqingv3/rulebase。提供后优先于 --seat-models",
    )
    p.add_argument(
        "--seat-labels",
        nargs=4,
        default=None,
        metavar=("L0", "L1", "L2", "L3"),
        help="按座位指定 4 个模型标签；未指定时从模型路径推断",
    )
    p.add_argument("--games", type=int, default=100, help="总局数")
    p.add_argument(
        "--game-format",
        choices=["hanchan", "kyoku"],
        default="hanchan",
        help="对战粒度：完整半庄(hanchan)或单局(kyoku)",
    )
    p.add_argument(
        "--save-games",
        type=int,
        default=10,
        metavar="N",
        help="保存最值得看的 N 局牌谱（按 hora 数和得分差筛选，默认 10）",
    )
    p.add_argument(
        "--save-all-games", action="store_true", help="保存全部牌谱为 .mjson"
    )
    p.add_argument(
        "--save-npz",
        action="store_true",
        help="兼容导出：通过 src/training/preprocess.py 导出 cache .npz，并写 metadata.json；标准训练流程仍建议 .mjson -> 离线 preprocess",
    )
    p.add_argument(
        "--cache-adapter",
        choices=["base", "meld_rank", "v3_aux"],
        default="base",
        help="保存 .npz 时使用的 preprocess adapter",
    )
    p.add_argument(
        "--cache-value-strategy",
        choices=["heuristic", "mc_return"],
        default="heuristic",
        help="保存 .npz 时使用的 value strategy",
    )
    p.add_argument(
        "--cache-encode-module",
        default="keqingv1.features",
        help="保存 .npz 时使用的 encode module",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认 artifacts/selfplay_benchmarks/<model>_<timestamp>",
    )
    p.add_argument("--beam-k", type=int, default=3, help="beam search k (0=禁用)")
    p.add_argument("--beam-lambda", type=float, default=1.0, help="value head 权重")
    p.add_argument("--device", default="cuda", help="推理设备")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument(
        "--fixed-seats",
        action="store_true",
        help="固定模型与座位映射；默认每场随机分配座次",
    )
    p.add_argument(
        "--export-anomaly-games",
        type=int,
        default=0,
        metavar="N",
        help="按高频异常动作分数导出前 N 局牌谱",
    )
    p.add_argument(
        "--anomaly-min-score",
        type=float,
        default=1.0,
        help="异常动作导出的最低分数阈值",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    resolved_model = _resolve_model_path(args.model)
    if args.output_dir is None:
        args.output_dir = _default_output_dir(_default_model_label(resolved_model))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    if args.seed is not None:
        random.seed(args.seed)

    seat_model_paths, seat_model_labels, seat_bot_kinds = _resolve_seat_bots(
        resolved_model, args.seat_bots, args.seat_models, args.seat_labels
    )
    machine_names = _build_machine_names(seat_model_labels)
    print("加载模型阵容:", flush=True)
    for machine_id, (model_path, model_label, machine_name, bot_kind) in enumerate(
        zip(seat_model_paths, seat_model_labels, machine_names, seat_bot_kinds)
    ):
        print(
            f"  机体{machine_id}: {machine_name} ({model_label}) -> {model_path} [{bot_kind}]",
            flush=True,
        )
    machine_bots = [
        create_runtime_bot(
            bot_name=seat_bot_kinds[i],
            player_id=i,
            project_root=Path.cwd(),
            model_path=None if seat_model_paths[i] == "rulebase" else seat_model_paths[i],
            device=args.device,
            verbose=args.verbose,
            beam_k=args.beam_k,
            beam_lambda=args.beam_lambda,
        )
        for i in range(4)
    ]
    format_label = "半庄" if args.game_format == "hanchan" else "单局"
    print(
        f"开始自对战: {args.games} {format_label}，beam_k={args.beam_k}，beam_lambda={args.beam_lambda}",
        flush=True,
    )

    manager = BattleManager()
    rank_counts = defaultdict(int)  # rank -> count
    total_scores = [0] * 4
    hora_total = 0
    total_turns = 0
    aggregate_event_counts: Counter = Counter()
    aggregate_action_counts: Counter = Counter()
    aggregate_action_counts_by_actor: Dict[str, Counter] = defaultdict(Counter)
    aggregate_fallback_counts: Counter = Counter()
    aggregate_decision_seconds_by_actor: Dict[str, float] = defaultdict(float)
    aggregate_decision_counts_by_actor: Counter = Counter()
    model_rank_counts: Dict[str, Counter] = defaultdict(Counter)
    model_total_scores: Dict[str, int] = defaultdict(int)
    model_games: Counter = Counter()
    player_stat_infos: Dict[str, PlayerStatInfo] = {
        str(pid): PlayerStatInfo() for pid in range(4)
    }
    model_stat_infos: Dict[str, PlayerStatInfo] = defaultdict(PlayerStatInfo)
    model_decision_seconds: Dict[str, float] = defaultdict(float)
    model_decision_counts: Counter = Counter()
    game_results = []  # [{game_id, scores, ranks, interest, events}]
    error_games = []
    total_rounds_played = 0

    _write_stats_snapshot(
        output_dir,
        _build_stats_snapshot(
            args=args,
            seat_model_paths=seat_model_paths,
            seat_model_labels=seat_model_labels,
            rank_counts=rank_counts,
            total_scores=total_scores,
            model_rank_counts=model_rank_counts,
            model_total_scores=model_total_scores,
            model_games=model_games,
            player_stat_infos=player_stat_infos,
            model_stat_infos=model_stat_infos,
            model_decision_seconds=model_decision_seconds,
            model_decision_counts=model_decision_counts,
            total_rounds_played=total_rounds_played,
            hora_total=hora_total,
            total_turns=total_turns,
            aggregate_event_counts=aggregate_event_counts,
            aggregate_action_counts=aggregate_action_counts,
            aggregate_action_counts_by_actor=aggregate_action_counts_by_actor,
            aggregate_fallback_counts=aggregate_fallback_counts,
            aggregate_decision_seconds_by_actor=aggregate_decision_seconds_by_actor,
            aggregate_decision_counts_by_actor=aggregate_decision_counts_by_actor,
            replay_manifest=[],
            anomaly_manifest=[],
            error_games=error_games,
            elapsed=0.0,
            completed_games=0,
            cache_export=None,
        ),
    )

    for game_idx in range(args.games):
        seat_assignment = _seat_assignment(4, fixed_seats=args.fixed_seats)
        game_seat_model_paths = [seat_model_paths[machine_id] for machine_id in seat_assignment]
        game_seat_model_labels = [seat_model_labels[machine_id] for machine_id in seat_assignment]
        game_seat_bot_kinds = [seat_bot_kinds[machine_id] for machine_id in seat_assignment]
        game_player_names = [machine_names[machine_id] for machine_id in seat_assignment]
        game_bots = [machine_bots[machine_id] for machine_id in seat_assignment]
        for seat, bot in enumerate(game_bots):
            bot.player_id = seat
            bot.reset()

        config = BattleConfig(
            player_count=4,
            players=[
                {"id": i, "name": game_player_names[i], "type": "bot"}
                for i in range(4)
            ],
        )

        room = manager.create_room(config)
        room.human_player_id = -1
        room.state.bakaze = "E"
        room.state.kyoku = 1
        room.state.honba = 0
        room.state.oya = 0
        room.state.scores = [25000, 25000, 25000, 25000]
        room.state.kyotaku = 0

        seed = random.randint(0, 2**31) if args.seed is None else args.seed + game_idx
        try:
            if args.game_format == "hanchan":
                result = run_one_game(manager, room, game_bots, seed=seed)
            else:
                result = run_one_kyoku(manager, room, game_bots, seed=seed)
        except Exception as exc:
            artifacts = _save_error_artifacts(output_dir, game_idx, room, exc, game_bots)
            error_games.append(
                {
                    "game_id": game_idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "player_names": game_player_names[:],
                    "seat_model_labels": game_seat_model_labels[:],
                    "seat_bot_kinds": game_seat_bot_kinds[:],
                    "seat_assignment": seat_assignment[:],
                    **artifacts,
                }
            )
            print(
                f"  [{game_idx + 1}/{args.games}] ERROR {type(exc).__name__}: {exc} "
                f"-> {artifacts['mjson']}",
                flush=True,
            )
            elapsed = time.perf_counter() - started_at
            completed_games = len(game_results)
            _append_progress_line(
                output_dir,
                {
                    "game_id": game_idx,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                    "completed_games": completed_games,
                    "total_games": args.games,
                    "error_count": len(error_games),
                    "player_names": game_player_names,
                    "seat_model_labels": game_seat_model_labels,
                    "seat_bot_kinds": game_seat_bot_kinds,
                    "seat_assignment": seat_assignment,
                },
            )
            _write_stats_snapshot(
                output_dir,
                _build_stats_snapshot(
                    args=args,
                    seat_model_paths=seat_model_paths,
                    seat_model_labels=seat_model_labels,
                    rank_counts=rank_counts,
                    total_scores=total_scores,
                    model_rank_counts=model_rank_counts,
                    model_total_scores=model_total_scores,
                    model_games=model_games,
                    player_stat_infos=player_stat_infos,
                    model_stat_infos=model_stat_infos,
                    model_decision_seconds=model_decision_seconds,
                    model_decision_counts=model_decision_counts,
                    total_rounds_played=total_rounds_played,
                    hora_total=hora_total,
                    total_turns=total_turns,
                    aggregate_event_counts=aggregate_event_counts,
                    aggregate_action_counts=aggregate_action_counts,
                    aggregate_action_counts_by_actor=aggregate_action_counts_by_actor,
                    aggregate_fallback_counts=aggregate_fallback_counts,
                    aggregate_decision_seconds_by_actor=aggregate_decision_seconds_by_actor,
                    aggregate_decision_counts_by_actor=aggregate_decision_counts_by_actor,
                    replay_manifest=[],
                    anomaly_manifest=[],
                    error_games=error_games,
                    elapsed=elapsed,
                    completed_games=completed_games,
                    cache_export=None,
                ),
            )
            continue

        scores = result["scores"]
        events = result["events"]
        rounds_played = int(result.get("rounds", 1))
        total_rounds_played += rounds_played
        total_turns += result.get("turns", 0)
        aggregate_event_counts.update(result.get("event_type_counts", {}))
        aggregate_action_counts.update(result.get("chosen_action_counts", {}))
        aggregate_fallback_counts.update(result.get("fallback_counts", {}))
        for actor_id, counts in result.get("chosen_action_counts_by_actor", {}).items():
            aggregate_action_counts_by_actor[str(actor_id)].update(counts)
        decision_latency = result.get("decision_latency", {})
        for actor_id, info in decision_latency.get("by_actor", {}).items():
            aggregate_decision_seconds_by_actor[str(actor_id)] += float(
                info.get("total_seconds", 0.0)
            )
            aggregate_decision_counts_by_actor[str(actor_id)] += int(
                info.get("decision_count", 0)
            )
            actor_idx = int(actor_id)
            if 0 <= actor_idx < len(game_seat_model_labels):
                model_label = game_seat_model_labels[actor_idx]
                model_decision_seconds[model_label] += float(
                    info.get("total_seconds", 0.0)
                )
                model_decision_counts[model_label] += int(
                    info.get("decision_count", 0)
                )
        ranks = _rank_of_scores(scores)
        interest = _score_interest(events, scores)
        hora_count = sum(1 for e in events if e.get("type") == "hora")
        hora_total += hora_count

        for pid in range(4):
            rank_counts[ranks[pid]] += 1
            total_scores[pid] += scores[pid]
            model_label = game_seat_model_labels[pid]
            model_rank_counts[model_label][ranks[pid]] += 1
            model_total_scores[model_label] += scores[pid]
            model_games[model_label] += 1
            if args.game_format == "hanchan":
                for round_result in result.get("round_results", []):
                    player_stat_infos[str(pid)].record_round(
                        actor=pid,
                        events=round_result["events"],
                        final_score=round_result["scores"][pid],
                        turn_count=round_result.get("turns", 0),
                    )
                    model_stat_infos[model_label].record_round(
                        actor=pid,
                        events=round_result["events"],
                        final_score=round_result["scores"][pid],
                        turn_count=round_result.get("turns", 0),
                    )
            else:
                player_stat_infos[str(pid)].record_round(
                    actor=pid,
                    events=events,
                    final_score=scores[pid],
                    turn_count=result.get("turns", 0),
                )
                model_stat_infos[model_label].record_round(
                    actor=pid,
                    events=events,
                    final_score=scores[pid],
                    turn_count=result.get("turns", 0),
                )

        game_results.append(
            {
                "game_id": game_idx,
                "scores": scores,
                "ranks": ranks,
                "seat_model_labels": game_seat_model_labels[:],
                "seat_model_paths": game_seat_model_paths[:],
                "seat_bot_kinds": game_seat_bot_kinds[:],
                "player_names": game_player_names[:],
                "seat_assignment": seat_assignment[:],
                "interest": interest,
                "events": events,
                "turns": result.get("turns", 0),
                "rounds": rounds_played,
                "chosen_action_counts": result.get("chosen_action_counts", {}),
                "fallback_counts": result.get("fallback_counts", {}),
            }
        )

        if (game_idx + 1) % max(1, args.games // 10) == 0 or game_idx == args.games - 1:
            done = game_idx + 1
            label_summary = _build_label_summary(
                model_rank_counts=model_rank_counts,
                model_total_scores=model_total_scores,
                model_games=model_games,
                model_stat_infos=model_stat_infos,
                model_decision_seconds=model_decision_seconds,
                model_decision_counts=model_decision_counts,
            )
            progress_model_stats = {
                model_label: model_stat_infos[model_label].summary(
                    total_rounds=total_rounds_played
                )
                for model_label in sorted(model_stat_infos)
            }
            benchmark_side_summary = _build_benchmark_side_summary(
                seat_model_labels=seat_model_labels,
                label_summary=label_summary,
                model_stats=progress_model_stats,
            )
            summary_line = (
                _format_progress_benchmark_side_summary(benchmark_side_summary)
                if benchmark_side_summary
                else _format_progress_label_summary(label_summary)
            )
            print(
                f"  [{done}/{args.games}] hora/{format_label}={hora_total / done:.2f}"
                + (f" | {summary_line}" if summary_line else ""),
                flush=True,
            )

        elapsed = time.perf_counter() - started_at
        completed_games = len(game_results)
        progress_label_summary = _build_label_summary(
            model_rank_counts=model_rank_counts,
            model_total_scores=model_total_scores,
            model_games=model_games,
            model_stat_infos=model_stat_infos,
            model_decision_seconds=model_decision_seconds,
            model_decision_counts=model_decision_counts,
        )
        solo_label = _default_model_label(resolved_model)
        solo_summary = progress_label_summary.get(solo_label, {})
        if solo_summary:
            solo_summary = {
                "label": solo_label,
                **solo_summary,
            }
        _append_progress_line(
            output_dir,
            {
                "game_id": game_idx,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "completed_games": completed_games,
                "total_games": args.games,
                "rounds_played": rounds_played,
                "total_rounds": total_rounds_played,
                "scores": scores,
                "ranks": ranks,
                "hora_count": hora_count,
                "error_count": len(error_games),
                "player_names": game_player_names,
                "seat_model_labels": game_seat_model_labels,
                "seat_bot_kinds": game_seat_bot_kinds,
                "seat_assignment": seat_assignment,
                "label_summary": progress_label_summary,
                "solo_model_summary": solo_summary,
            },
        )
        _write_stats_snapshot(
            output_dir,
            _build_stats_snapshot(
                args=args,
                seat_model_paths=seat_model_paths,
                seat_model_labels=seat_model_labels,
                rank_counts=rank_counts,
                total_scores=total_scores,
                model_rank_counts=model_rank_counts,
                model_total_scores=model_total_scores,
                model_games=model_games,
                player_stat_infos=player_stat_infos,
                model_stat_infos=model_stat_infos,
                model_decision_seconds=model_decision_seconds,
                model_decision_counts=model_decision_counts,
                total_rounds_played=total_rounds_played,
                hora_total=hora_total,
                total_turns=total_turns,
                aggregate_event_counts=aggregate_event_counts,
                aggregate_action_counts=aggregate_action_counts,
                aggregate_action_counts_by_actor=aggregate_action_counts_by_actor,
                aggregate_fallback_counts=aggregate_fallback_counts,
                aggregate_decision_seconds_by_actor=aggregate_decision_seconds_by_actor,
                aggregate_decision_counts_by_actor=aggregate_decision_counts_by_actor,
                replay_manifest=[],
                anomaly_manifest=[],
                error_games=error_games,
                elapsed=elapsed,
                completed_games=completed_games,
                cache_export=None,
            ),
        )

    # 最终统计
    completed_games = len(game_results)
    print("\n=== 自对战结果 ===", flush=True)
    for rank in range(1, 5):
        print(
            f"  {rank}位: {rank_counts[rank]} 次 ({rank_counts[rank] / max(1, completed_games * 4) * 100:.1f}%)",
            flush=True,
        )
    print(f"  平均得分: {[s / max(1, completed_games) for s in total_scores]}", flush=True)
    print(f"  hora/{format_label}: {hora_total / max(1, completed_games):.2f}", flush=True)
    print(f"  总小局数: {total_rounds_played}", flush=True)
    print(
        f"  平均小局数/{format_label}: {total_rounds_played / max(1, completed_games):.2f}",
        flush=True,
    )
    elapsed = time.perf_counter() - started_at
    print(f"  错误局数: {len(error_games)}", flush=True)
    print(f"  总耗时: {elapsed:.2f}s", flush=True)
    print(f"  平均耗时: {elapsed / max(1, completed_games):.2f}s/局", flush=True)
    print(f"  平均巡数: {total_turns / max(1, completed_games):.2f}", flush=True)
    if model_games:
        final_label_summary = _build_label_summary(
            model_rank_counts=model_rank_counts,
            model_total_scores=model_total_scores,
            model_games=model_games,
            model_stat_infos=model_stat_infos,
            model_decision_seconds=model_decision_seconds,
            model_decision_counts=model_decision_counts,
        )
        final_model_stats = {
            model_label: model_stat_infos[model_label].summary(
                total_rounds=total_rounds_played
            )
            for model_label in sorted(model_stat_infos)
        }
        benchmark_side_summary = _build_benchmark_side_summary(
            seat_model_labels=seat_model_labels,
            label_summary=final_label_summary,
            model_stats=final_model_stats,
        )
        print("  模型汇总:", flush=True)
        for model_label, summary in final_label_summary.items():
            top_yaku = ", ".join(
                f"{key}={count}"
                for key, count in sorted(
                    summary.get("yaku_counts", {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ) or "-"
            print(
                f"    {model_label}: games={summary['games']} avg_rank={summary['avg_rank']:.3f} "
                f"avg_score={summary['avg_score']:.1f} top1={summary['top1_rate'] * 100:.1f}% "
                f"top4={summary['top4_rate'] * 100:.1f}% "
                f"think_events={summary['decision_count']} think_time={summary['decision_total_seconds']:.2f}s "
                f"avg_think={summary['avg_decision_seconds']:.4f}s",
                flush=True,
            )
            print(f"      yaku: {top_yaku}", flush=True)
        if benchmark_side_summary:
            print("  1v3聚合:", flush=True)
            solo_side = benchmark_side_summary["solo_side"]
            trio_side = benchmark_side_summary["trio_side"]
            print(
                f"    单机侧 {solo_side['label']}: avg_rank={solo_side['avg_rank']:.3f} "
                f"avg_score={solo_side['avg_score']:.1f} top1={solo_side['top1_rate'] * 100:.1f}% "
                f"hora_rate={solo_side['hora_rate'] * 100:.1f}% deal_in_rate={solo_side['deal_in_rate'] * 100:.1f}%",
                flush=True,
            )
            print(
                f"    三机侧 {trio_side['label']}: avg_rank={trio_side['avg_rank']:.3f} "
                f"avg_score={trio_side['avg_score']:.1f} top1={trio_side['top1_rate'] * 100:.1f}% "
                f"hora_rate={trio_side['hora_rate'] * 100:.1f}% deal_in_rate={trio_side['deal_in_rate'] * 100:.1f}%",
                flush=True,
            )
    if aggregate_action_counts:
        top_actions = ", ".join(
            f"{k}={v}" for k, v in aggregate_action_counts.most_common(8)
        )
        print(f"  动作频次Top: {top_actions}", flush=True)
    if aggregate_action_counts_by_actor:
        for actor_id in sorted(aggregate_action_counts_by_actor):
            actor_top = ", ".join(
                f"{k}={v}"
                for k, v in aggregate_action_counts_by_actor[actor_id].most_common(5)
            )
            print(f"  玩家{actor_id}动作Top: {actor_top}", flush=True)
    if aggregate_fallback_counts:
        fallback_summary = ", ".join(
            f"{k}={v}" for k, v in aggregate_fallback_counts.items()
        )
        print(f"  fallback统计: {fallback_summary}", flush=True)
    total_decision_seconds = sum(aggregate_decision_seconds_by_actor.values())
    total_decision_count = sum(aggregate_decision_counts_by_actor.values())
    print(
        f"  决策耗时: total={total_decision_seconds:.2f}s "
        f"count={total_decision_count} avg={total_decision_seconds / max(1, total_decision_count):.4f}s",
        flush=True,
    )
    for actor_id in sorted(aggregate_decision_counts_by_actor):
        actor_total = aggregate_decision_seconds_by_actor[actor_id]
        actor_count = aggregate_decision_counts_by_actor[actor_id]
        print(
            f"  玩家{actor_id}决策耗时: total={actor_total:.2f}s "
            f"count={actor_count} avg={actor_total / max(1, actor_count):.4f}s",
            flush=True,
        )
    if error_games:
        print(f"  错误日志 -> {output_dir / 'errors'}", flush=True)

    replay_model_for_exports = (
        None if any(kind == "rulebase" for kind in seat_bot_kinds) else resolved_model
    )
    replay_manifest = _save_replay_samples(
        output_dir,
        game_results,
        args.save_games,
        save_all=args.save_all_games,
        replay_model_path=replay_model_for_exports,
    )
    if replay_manifest:
        print(f"  保存 {len(replay_manifest)} 局牌谱 -> {output_dir / 'replays'}", flush=True)
        replay_ready = sum(1 for item in replay_manifest if item.get("replay_id"))
        if replay_ready:
            print(f"  ReplayUI 直连回放已生成 {replay_ready} 局", flush=True)

    anomaly_manifest = _save_anomaly_samples(
        output_dir,
        game_results,
        args.export_anomaly_games,
        min_score=args.anomaly_min_score,
        replay_model_path=replay_model_for_exports,
    )
    if anomaly_manifest:
        print(
            f"  导出高频异常动作对局 {len(anomaly_manifest)} 局 -> "
            f"{output_dir / 'anomaly_replays'}",
            flush=True,
        )
        replay_ready = sum(1 for item in anomaly_manifest if item.get("replay_id"))
        if replay_ready:
            print(f"  ReplayUI 直连回放已生成 {replay_ready} 局", flush=True)

    cache_export = None

    # 保存 npz
    if args.save_npz:
        print(
            "  警告: --save-npz 仅保留为 convenience wrapper；"
            "训练 cache 的权威实现仍在 src/training/preprocess.py",
            flush=True,
        )
        npz_dir = output_dir / "npz"
        npz_dir.mkdir(exist_ok=True)
        saved = 0
        for r in game_results:
            fname = npz_dir / f"game_{r['game_id']:05d}.npz"
            ok = _save_npz(
                fname,
                r["events"],
                adapter_name=args.cache_adapter,
                value_strategy=args.cache_value_strategy,
                encode_module=args.cache_encode_module,
            )
            if ok:
                saved += 1
        cache_export = _build_cache_export_metadata(
            adapter_name=args.cache_adapter,
            value_strategy=args.cache_value_strategy,
            encode_module=args.cache_encode_module,
            saved_games=saved,
            output_dir=npz_dir,
        )
        _write_cache_export_metadata(npz_dir, cache_export)
        print(
            f"  保存 {saved} 局 .npz cache -> {npz_dir} "
            f"(adapter={args.cache_adapter}, value_strategy={args.cache_value_strategy}, "
            f"encode={args.cache_encode_module})",
            flush=True,
        )
        print(
            "  推荐标准训练导出流程：先保存 .mjson，再离线 preprocess。"
            f" 推荐命令 -> {cache_export['recommended_preprocess_command']}",
            flush=True,
        )

    # 保存统计 json
    stats = _build_stats_snapshot(
        args=args,
        seat_model_paths=seat_model_paths,
        seat_model_labels=seat_model_labels,
        rank_counts=rank_counts,
        total_scores=total_scores,
        model_rank_counts=model_rank_counts,
        model_total_scores=model_total_scores,
        model_games=model_games,
        player_stat_infos=player_stat_infos,
        model_stat_infos=model_stat_infos,
        model_decision_seconds=model_decision_seconds,
        model_decision_counts=model_decision_counts,
        total_rounds_played=total_rounds_played,
        hora_total=hora_total,
        total_turns=total_turns,
        aggregate_event_counts=aggregate_event_counts,
        aggregate_action_counts=aggregate_action_counts,
        aggregate_action_counts_by_actor=aggregate_action_counts_by_actor,
        aggregate_fallback_counts=aggregate_fallback_counts,
        aggregate_decision_seconds_by_actor=aggregate_decision_seconds_by_actor,
        aggregate_decision_counts_by_actor=aggregate_decision_counts_by_actor,
        replay_manifest=replay_manifest,
        anomaly_manifest=anomaly_manifest,
        error_games=error_games,
        elapsed=elapsed,
        completed_games=completed_games,
        cache_export=cache_export,
    )
    _write_stats_snapshot(output_dir, stats)
    print(f"  统计 -> {output_dir}/stats.json", flush=True)


if __name__ == "__main__":
    main()
