from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Set

from mahjong.shanten import Shanten
from keqingv3.progress_oracle import (
    NormalProgressInfo,
    analyze_normal_progress_from_counts as _oracle_calc_normal_progress_from_counts,
    calc_shanten_waits_from_counts as _oracle_calc_shanten_waits_from_counts,
)
from mahjong_env.legal_actions import enumerate_legal_action_specs
from mahjong_env.replay_normalizer import (
    is_replay_meta_event,
    normalize_replay_events,
    normalize_replay_label_for_legal_compare,
    replay_label_matches_legal,
    replay_label_to_legal_mjai,
)
from mahjong_env.state import GameState, apply_event
from mahjong_env.tiles import AKA_DORA_TILES, tile_to_34 as _to_34
from mahjong_env.types import MjaiEvent
from mahjong_env.types import ActionSpec

# ---------------------------------------------------------------------------
# generic 3n+1 / 3n+2 standard-hand shanten/waits helpers
# ---------------------------------------------------------------------------


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    if tile.endswith("r"):
        return tile[:-1]
    return tile


_TILE34_TO_STR: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]


def _counts34_from_hand(hand: List[str]) -> List[int]:
    counts = [0] * 34
    for tile in hand:
        t34 = _to_34(tile)
        if 0 <= t34 < 34:
            counts[t34] += 1
    return counts


def _is_suited_sequence_start(tile34: int) -> bool:
    return tile34 < 27 and (tile34 % 9) <= 6


def _is_complete_regular_counts(
    counts: tuple[int, ...],
    cache: Dict[tuple[tuple[int, ...], int, bool], bool],
    melds_needed: Optional[int] = None,
    need_pair: bool = True,
) -> bool:
    if melds_needed is None:
        tile_count = sum(counts)
        if tile_count % 3 != 2:
            return False
        melds_needed = (tile_count - 2) // 3

    key = (counts, melds_needed, need_pair)
    cached = cache.get(key)
    if cached is not None:
        return cached

    first = next((i for i, cnt in enumerate(counts) if cnt > 0), None)
    if first is None:
        result = melds_needed == 0 and not need_pair
        cache[key] = result
        return result

    work = list(counts)
    result = False

    if need_pair and work[first] >= 2:
        work[first] -= 2
        if _is_complete_regular_counts(tuple(work), cache, melds_needed, False):
            result = True
        work[first] += 2

    if not result and melds_needed > 0 and work[first] >= 3:
        work[first] -= 3
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True
        work[first] += 3

    if (
        not result
        and melds_needed > 0
        and _is_suited_sequence_start(first)
        and work[first + 1] > 0
        and work[first + 2] > 0
    ):
        work[first] -= 1
        work[first + 1] -= 1
        work[first + 2] -= 1
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True

    cache[key] = result
    return result


def _find_regular_waits(counts: tuple[int, ...]) -> List[bool]:
    waits = [False] * 34
    if sum(counts) % 3 != 1:
        return waits

    complete_cache: Dict[tuple[tuple[int, ...], int, bool], bool] = {}
    for tile34, cnt in enumerate(counts):
        if cnt >= 4:
            continue
        work = list(counts)
        work[tile34] += 1
        if _is_complete_regular_counts(tuple(work), complete_cache):
            waits[tile34] = True
    return waits


def _find_special_waits(
    counts34: tuple[int, ...],
    *,
    shanten_calc: Shanten,
) -> tuple[int, List[bool]]:
    chiitoi_shanten = int(shanten_calc.calculate_shanten_for_chiitoitsu_hand(list(counts34)))
    kokushi_shanten = int(shanten_calc.calculate_shanten_for_kokushi_hand(list(counts34)))
    special_shanten = min(chiitoi_shanten, kokushi_shanten)
    waits = [False] * 34
    if special_shanten != 0:
        return special_shanten, waits

    for tile34, cnt in enumerate(counts34):
        if cnt >= 4:
            continue
        work = list(counts34)
        work[tile34] += 1
        if (
            (chiitoi_shanten == 0 and int(shanten_calc.calculate_shanten_for_chiitoitsu_hand(work)) == -1)
            or (kokushi_shanten == 0 and int(shanten_calc.calculate_shanten_for_kokushi_hand(work)) == -1)
        ):
            waits[tile34] = True
    return special_shanten, waits


def _calc_shanten_waits(hand: List[str], melds: List[dict]):
    """返回 (standard_shanten, waits_count, waits_tile34_bools, tehai_count)。"""
    counts34 = _counts34_from_hand(hand)
    try:
        regular_shanten, waits_count, waits34, tehai_count = _oracle_calc_shanten_waits_from_counts(tuple(counts34))
        waits = list(waits34)
        shanten = regular_shanten
        if not melds:
            shanten_calc = Shanten()
            special_shanten, special_waits = _find_special_waits(tuple(counts34), shanten_calc=shanten_calc)
            if special_shanten < shanten:
                shanten = special_shanten
                waits = special_waits if special_shanten == 0 else [False] * 34
            elif special_shanten == shanten == 0:
                waits = [a or b for a, b in zip(waits, special_waits)]
        return shanten, sum(waits), waits, tehai_count
    except Exception:
        del melds
        tehai_count = sum(counts34)
        return 8, 0, [False] * 34, tehai_count


def _meld_tile34_counts(melds: List[dict]) -> List[int]:
    counts = [0] * 34
    for meld in melds:
        for p in meld.get("consumed", []) + ([meld.get("pai")] if meld.get("pai") else []):
            t34 = _to_34(p)
            if 0 <= t34 < 34:
                counts[t34] += 1
    return counts


def _hand_tile34_counts(hand: List[str]) -> List[int]:
    counts = [0] * 34
    for tile in hand:
        t34 = _to_34(tile)
        if 0 <= t34 < 34:
            counts[t34] += 1
    return counts


def _default_visible_counts(hand: List[str], melds: List[dict]) -> List[int]:
    counts = _hand_tile34_counts(hand)
    meld_counts = _meld_tile34_counts(melds)
    for i in range(34):
        counts[i] += meld_counts[i]
    return counts


def _counts_to_hand(counts: Sequence[int]) -> List[str]:
    hand: List[str] = []
    for t34, cnt in enumerate(counts):
        if cnt > 0:
            hand.extend([_TILE34_TO_STR[t34]] * cnt)
    return hand


def _tenpai_live_wait_count(
    counts13: tuple[int, ...],
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> int:
    cached = shanten_cache.get(counts13)
    if cached is None:
        cached = _calc_shanten_waits(_counts_to_hand(counts13), melds)
        shanten_cache[counts13] = cached
    shanten, _waits_count, waits_tiles, _tehai_count = cached
    if shanten != 0:
        return 0
    return sum(
        max(0, 4 - visible_counts_local[t34])
        for t34, flag in enumerate(waits_tiles)
        if flag
    )


def _best_tenpai_wait_live_after_draw(
    counts14: tuple[int, ...],
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> int:
    best_wait_live = 0
    seen_discards: Set[int] = set()
    for discard34, cnt in enumerate(counts14):
        if cnt <= 0 or discard34 in seen_discards:
            continue
        seen_discards.add(discard34)
        after_counts13 = list(counts14)
        after_counts13[discard34] -= 1
        after_counts13_t = tuple(after_counts13)
        after_shanten, _w_cnt, _w_tiles, _tehai = shanten_cache.get(after_counts13_t, (None, None, None, None))
        if after_shanten is None:
            after_shanten, _w_cnt, _w_tiles, _tehai = _calc_shanten_waits(_counts_to_hand(after_counts13_t), melds)
            shanten_cache[after_counts13_t] = (after_shanten, _w_cnt, _w_tiles, _tehai)
        if after_shanten != 0:
            continue
        wait_live = _tenpai_live_wait_count(after_counts13_t, melds, visible_counts_local, shanten_cache)
        if wait_live > best_wait_live:
            best_wait_live = wait_live
    return best_wait_live


def _is_good_shape_draw_for_one_shanten(
    counts13: tuple[int, ...],
    draw_tile34: int,
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> bool:
    """严格复刻一向听好型判定：

    摸入有效牌后，枚举所有去重打牌分支；只要存在一个分支能形成一般形听牌，
    且该听牌的 live waits > 4，就判该摸牌为好型进张。
    """

    counts14 = list(counts13)
    counts14[draw_tile34] += 1
    counts14_t = tuple(counts14)
    return _best_tenpai_wait_live_after_draw(
        counts14_t, melds, visible_counts_local, shanten_cache
    ) > 4


def _calc_normal_progress(
    hand: List[str],
    melds: List[dict],
    visible_counts: Optional[List[int]] = None,
) -> NormalProgressInfo:
    if visible_counts is None:
        visible_counts = _default_visible_counts(hand, melds)
    return _oracle_calc_normal_progress_from_counts(
        tuple(_hand_tile34_counts(hand)),
        tuple(visible_counts),
    )


@dataclass
class ReplaySample:
    state: Dict
    actor: int
    actor_name: str
    label_action: Dict
    legal_actions: List[Dict]
    value_target: float
    score_delta_target: float = 0.0
    win_target: float = 0.0
    dealin_target: float = 0.0
    ryukyoku_tenpai_target: float = 0.0


class IllegalLabelActionError(ValueError):
    """Raised when a replay label action is not contained in the reconstructed legal set."""


@dataclass
class ValueComputationContext:
    event_index: int
    event: MjaiEvent
    actor: int
    delta_shanten: float
    delta_waits: float
    delta_tehai: float
    events: List[MjaiEvent]


@dataclass
class PendingValueSample:
    sample: ReplaySample
    round_step_index: int


def _finalize_aux_targets(
    pending_samples: List[PendingValueSample],
    terminal_event: Optional[MjaiEvent],
    *,
    score_norm: float = 30000.0,
) -> None:
    if not pending_samples:
        return

    score_deltas = None
    hora_actor = None
    hora_target = None
    ryukyoku_tenpai_players: set[int] = set()
    if terminal_event is not None:
        score_deltas = terminal_event.get("deltas") or terminal_event.get("score_delta")
        if terminal_event.get("type") == "hora":
            hora_actor = terminal_event.get("actor")
            hora_target = terminal_event.get("target")
        elif terminal_event.get("type") == "ryukyoku":
            ryukyoku_tenpai_players = {
                int(pid) for pid in (terminal_event.get("tenpai_players") or []) if pid is not None
            }

    for pending in pending_samples:
        actor = pending.sample.actor
        if isinstance(score_deltas, list) and len(score_deltas) == 4:
            pending.sample.score_delta_target = float(score_deltas[actor]) / score_norm
        else:
            pending.sample.score_delta_target = 0.0
        pending.sample.win_target = 1.0 if hora_actor == actor else 0.0
        pending.sample.dealin_target = 1.0 if (hora_target == actor and hora_actor != actor) else 0.0
        pending.sample.ryukyoku_tenpai_target = 1.0 if actor in ryukyoku_tenpai_players else 0.0


class ValueTargetStrategy(Protocol):
    def initial_value(self, ctx: ValueComputationContext) -> float:
        ...

    def finalize_round(
        self,
        pending_samples: List[PendingValueSample],
        terminal_event: Optional[MjaiEvent],
    ) -> None:
        ...


class HeuristicValueStrategy:
    def __init__(self):
        self.w_shanten = 0.05
        self.w_waits = 0.008
        self.w_call_tehai = 0.003
        self.value_norm = 30000.0
        self.call_actions = {"chi", "pon", "daiminkan", "ankan", "kakan"}

    def initial_value(self, ctx: ValueComputationContext) -> float:
        et = ctx.event["type"]
        actor = ctx.actor
        delta_shanten = ctx.delta_shanten
        delta_waits = ctx.delta_waits
        delta_tehai = ctx.delta_tehai

        if et == "hora":
            sd = ctx.event.get("deltas") or ctx.event.get("score_delta")
            if isinstance(sd, list) and len(sd) == 4:
                return float(sd[actor]) / self.value_norm * 1.5
            return float(self.w_shanten * delta_shanten + self.w_waits * delta_waits)

        if et == "ryukyoku":
            sd = ctx.event.get("deltas")
            if isinstance(sd, list) and len(sd) == 4:
                base = float(sd[actor]) / self.value_norm
                if base != 0.0:
                    return base
            tenpai_players = {int(pid) for pid in (ctx.event.get("tenpai_players") or []) if pid is not None}
            if tenpai_players:
                return 0.05 if actor in tenpai_players else -0.05
            return 0.0

        value = float(self.w_shanten * delta_shanten + self.w_waits * delta_waits)
        if et in self.call_actions:
            value += float(self.w_call_tehai * delta_tehai)
            value = max(min(value, 10.0), -10.0)

        if et == "dahai":
            next_ev = _next_meaningful_event(ctx.events, ctx.event_index)
            if (
                next_ev is not None
                and next_ev.get("type") == "hora"
                and next_ev.get("actor") != actor
            ):
                sd = next_ev.get("deltas") or next_ev.get("score_delta")
                if isinstance(sd, list) and len(sd) == 4:
                    return float(sd[actor]) / self.value_norm
        return value

    def finalize_round(
        self,
        pending_samples: List[PendingValueSample],
        terminal_event: Optional[MjaiEvent],
    ) -> None:
        del pending_samples, terminal_event


class MCReturnValueStrategy:
    def __init__(self, gamma: float = 0.99, value_norm: float = 30000.0):
        self.gamma = gamma
        self.value_norm = value_norm

    def initial_value(self, ctx: ValueComputationContext) -> float:
        del ctx
        return 0.0

    def finalize_round(
        self,
        pending_samples: List[PendingValueSample],
        terminal_event: Optional[MjaiEvent],
    ) -> None:
        if not pending_samples or terminal_event is None:
            return
        sd = terminal_event.get("deltas") or terminal_event.get("score_delta")
        if not (isinstance(sd, list) and len(sd) == 4):
            return
        last_step = pending_samples[-1].round_step_index
        for pending in pending_samples:
            actor = pending.sample.actor
            steps_remaining = max(0, last_step - pending.round_step_index)
            pending.sample.value_target = float(sd[actor]) / self.value_norm * (
                self.gamma ** steps_remaining
            )


def _resolve_value_strategy(
    value_strategy: str | ValueTargetStrategy | None,
) -> ValueTargetStrategy:
    if value_strategy is None or value_strategy == "heuristic":
        return HeuristicValueStrategy()
    if value_strategy == "mc_return":
        return MCReturnValueStrategy()
    return value_strategy


ACTION_TYPES_FOR_LABEL = {
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "reach",
    "hora",
    "ryukyoku",
}


def read_mjai_jsonl(path: str | Path) -> List[MjaiEvent]:
    events: List[MjaiEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def extract_actor_names(events: Sequence[MjaiEvent]) -> List[str]:
    for e in events:
        if (
            e.get("type") == "start_game"
            and isinstance(e.get("names"), list)
            and len(e["names"]) == 4
        ):
            return [str(x) for x in e["names"]]
    return ["p0", "p1", "p2", "p3"]


def _next_meaningful_event(events: List[MjaiEvent], i: int) -> Optional[MjaiEvent]:
    """跳过 reach_accepted/dora 等元事件，返回 i 之后第一个有意义的事件。"""
    for j in range(i + 1, len(events)):
        if not is_replay_meta_event(events[j]):
            return events[j]
    return None


def _next_actor_dahai(
    events: List[MjaiEvent], i: int, actor: int
) -> Optional[MjaiEvent]:
    """在副露事件 i 后找到同 actor 的 dahai 事件（副露后必须打牌）。"""
    for j in range(i + 1, len(events)):
        ev = events[j]
        if is_replay_meta_event(ev):
            continue
        if ev.get("type") == "dahai" and ev.get("actor") == actor:
            return ev
        # 如果遇到非 skip 的其他事件则停止（副露后紧接着就是 dahai）
        break
    return None


def build_supervised_samples(
    events: List[MjaiEvent],
    actor_filter: Optional[Set[int]] = None,
    actor_name_filter: Optional[Set[str]] = None,
    value_strategy: str | ValueTargetStrategy | None = None,
    strict_legal_labels: bool = True,
) -> List[ReplaySample]:
    events = normalize_replay_events(events)
    state = GameState()
    samples: List[ReplaySample] = []
    strategy = _resolve_value_strategy(value_strategy)
    actor_names = extract_actor_names(events)
    pending_round_samples: List[PendingValueSample] = []
    round_step_index = 0
    call_actions = {"chi", "pon", "daiminkan", "ankan", "kakan"}
    multi_ron_base_state: Optional[GameState] = None

    for i, event in enumerate(events):
        et = event["type"]
        actor = event.get("actor")
        next_ev = _next_meaningful_event(events, i)

        if et == "start_kyoku":
            pending_round_samples.clear()
            round_step_index = 0
            multi_ron_base_state = None

        if et == "hora" and multi_ron_base_state is None and next_ev is not None and next_ev.get("type") == "hora":
            multi_ron_base_state = copy.deepcopy(state)

        collect_sample = (
            actor is not None
            and et in ACTION_TYPES_FOR_LABEL
            and state.in_game
            and (actor_filter is None or actor in actor_filter)
            # 立直后摸切（reached=True 时的 dahai）无决策价值，跳过
            and not (et == "dahai" and 0 <= actor < 4 and state.players[actor].reached)
        )
        if collect_sample and actor_name_filter is not None:
            actor_name = (
                actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
            )
            collect_sample = actor_name in actor_name_filter

        shanten_before: Optional[int] = None
        waits_before_cnt: Optional[int] = None
        tehai_before_cnt: Optional[int] = None
        if collect_sample:
            sample_state = multi_ron_base_state if (et == "hora" and multi_ron_base_state is not None) else state
            snap_before = sample_state.snapshot(actor)
            hand_before = snap_before.get("hand", [])
            melds_before = (snap_before.get("melds") or [[], [], [], []])[actor]
            shanten_before, waits_before_cnt, _waits_before_bools, tehai_before_cnt = (
                _calc_shanten_waits(hand_before, melds_before)
            )

        if collect_sample:
            # Apply event first so we can compute after-state
            apply_event(state, event)
            snap_after = state.snapshot(actor)
            hand_after = snap_after.get("hand", [])
            melds_after = (snap_after.get("melds") or [[], [], [], []])[actor]
            shanten_after, waits_after_cnt, _waits_after_bools, tehai_after_cnt = (
                _calc_shanten_waits(hand_after, melds_after)
            )
            # 副露+打牌联合 value_target：找到紧接着的 dahai，用副露+打牌后的状态计算 delta
            # snap[waits_count] 使用副露+打牌联合后的值（waits_after_cnt_vt），
            # 与 value_target 计算保持一致
            shanten_after_vt = shanten_after
            waits_after_cnt_vt = waits_after_cnt
            tehai_after_cnt_vt = tehai_after_cnt
            if et in call_actions and et not in ("ankan", "kakan"):
                next_dahai = _next_actor_dahai(events, i, actor)
                if next_dahai is not None:
                    discard_tile = _normalize_or_keep_aka(next_dahai["pai"])
                    hand_after_dahai = list(hand_after)
                    try:
                        hand_after_dahai.remove(discard_tile)
                    except ValueError:
                        norm = (
                            discard_tile[:-1]
                            if discard_tile.endswith("r")
                            else discard_tile
                        )
                        try:
                            hand_after_dahai.remove(norm)
                        except ValueError:
                            pass
                    shanten_after_vt, waits_after_cnt_vt, _, tehai_after_cnt_vt = (
                        _calc_shanten_waits(hand_after_dahai, melds_after)
                    )
        if collect_sample:
            delta_shanten = shanten_before - shanten_after_vt  # type: ignore[operator]
            delta_waits = waits_after_cnt_vt - waits_before_cnt  # type: ignore[operator]
            delta_tehai = tehai_after_cnt_vt - tehai_before_cnt  # type: ignore[operator]
            value_ctx = ValueComputationContext(
                event_index=i,
                event=event,
                actor=actor,
                delta_shanten=float(delta_shanten),
                delta_waits=float(delta_waits),
                delta_tehai=float(delta_tehai),
                events=events,
            )
            value_target_local = strategy.initial_value(value_ctx)

            # Keep local value proxy computed above.
            actor_name = (
                actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
            )
            snap = dict(snap_before)
            # Only collect supervised samples when the actor hand is visible.
            if snap["hand"]:
                # Attach shanten/waits features for the policy/value model.
                snap["shanten"] = shanten_before if shanten_before is not None else 0
                # tsumo_pai：摸牌事件时注入当前摸到的牌（raw，含赤宝牌），其余为 None
                snap["tsumo_pai"] = event.get("pai") if et == "tsumo" else None
                if et == "hora":
                    if event.get("is_haitei") is True:
                        snap["_hora_is_haitei"] = True
                    if event.get("is_houtei") is True:
                        snap["_hora_is_houtei"] = True
                    if event.get("is_rinshan") is True:
                        snap["_hora_is_rinshan"] = True
                    if event.get("is_chankan") is True:
                        snap["_hora_is_chankan"] = True
                # waits_count/waits_tiles: 与 snap_before（决策时刻）对应，两者保持同一时间点
                snap["waits_count"] = (
                    waits_before_cnt if waits_before_cnt is not None else 0
                )
                snap["waits_tiles"] = (
                    _waits_before_bools  # length-34 bool list, before action
                )

                legal_specs = enumerate_legal_action_specs(snap, actor)
                label = normalize_replay_label_for_legal_compare(event)
                legal_dicts = [spec.to_mjai() for spec in legal_specs]
                if strict_legal_labels and not replay_label_matches_legal(label, legal_dicts):
                    raise IllegalLabelActionError(
                        "replay label action not in reconstructed legal set: "
                        f"event_index={i} actor={actor} actor_name={actor_name} "
                        f"label={label} legal={legal_dicts} "
                        f"snap={{'bakaze': {snap.get('bakaze')!r}, 'kyoku': {snap.get('kyoku')!r}, "
                        f"'honba': {snap.get('honba')!r}, 'hand': {snap.get('hand')!r}, "
                        f"'last_discard': {snap.get('last_discard')!r}, 'melds': {(snap.get('melds') or [[], [], [], []])[actor]!r}}}"
                    )

                sample = ReplaySample(
                    state=snap,
                    actor=actor,
                    actor_name=actor_name,
                    label_action=label,
                    legal_actions=legal_dicts,
                    value_target=float(value_target_local),
                    score_delta_target=0.0,
                    win_target=0.0,
                    dealin_target=0.0,
                    ryukyoku_tenpai_target=0.0,
                )
                samples.append(sample)
                pending_round_samples.append(
                    PendingValueSample(
                        sample=sample,
                        round_step_index=round_step_index,
                    )
                )
                round_step_index += 1

        if not collect_sample:
            apply_event(state, event)

        # 生成 none/pass 样本：他家打牌后，有鸣牌机会但主动放弃的玩家
        if et == "dahai" and state.in_game:
            discarder = event.get("actor")
            if next_ev is not None:
                next_type = next_ev.get("type")
                next_actor = next_ev.get("actor")
                # 只有下一个事件是 tsumo（无人鸣/荣）才是主动 pass
                is_next_tsumo = next_type == "tsumo"
                for p in range(4):
                    if p == discarder:
                        continue
                    if actor_filter is not None and p not in actor_filter:
                        continue
                    if actor_name_filter is not None:
                        p_name = (
                            actor_names[p] if 0 <= p < len(actor_names) else f"p{p}"
                        )
                        if p_name not in actor_name_filter:
                            continue
                    snap_p = state.snapshot(p)
                    if not snap_p["hand"]:
                        continue
                    legal_p_specs = enumerate_legal_action_specs(snap_p, p)
                    non_none_p = [a for a in legal_p_specs if a.type != "none"]
                    if not non_none_p:
                        continue
                    if not is_next_tsumo:
                        continue
                    # 注入 shanten
                    hand_p = snap_p.get("hand", [])
                    melds_p = (snap_p.get("melds") or [[], [], [], []])[p]
                    shanten_p, waits_cnt_p, waits_tiles_p, _ = _calc_shanten_waits(
                        hand_p, melds_p
                    )
                    snap_p["shanten"] = shanten_p
                    snap_p["waits_count"] = waits_cnt_p
                    snap_p["waits_tiles"] = waits_tiles_p
                    sample = ReplaySample(
                        state=snap_p,
                        actor=p,
                        actor_name=actor_names[p]
                        if 0 <= p < len(actor_names)
                        else f"p{p}",
                        label_action={"type": "none", "actor": p},
                        legal_actions=[spec.to_mjai() for spec in legal_p_specs],
                        value_target=0.0,
                        score_delta_target=0.0,
                        win_target=0.0,
                        dealin_target=0.0,
                        ryukyoku_tenpai_target=0.0,
                    )
                    samples.append(sample)
                    pending_round_samples.append(
                        PendingValueSample(
                            sample=sample,
                            round_step_index=round_step_index,
                        )
                    )
                    round_step_index += 1

        if et in {"hora", "ryukyoku"}:
            strategy.finalize_round(pending_round_samples, event)
            _finalize_aux_targets(pending_round_samples, event)
        if et == "end_kyoku":
            strategy.finalize_round(pending_round_samples, None)
            _finalize_aux_targets(pending_round_samples, None)
            pending_round_samples.clear()
            round_step_index = 0
            multi_ron_base_state = None

        if et == "hora" and not (next_ev is not None and next_ev.get("type") == "hora"):
            multi_ron_base_state = None

    return samples


def replay_validate_label_legal(samples: List[ReplaySample]) -> List[str]:
    errors: List[str] = []
    for idx, sample in enumerate(samples):
        label = sample.label_action
        legal = sample.legal_actions
        found = replay_label_matches_legal(label, legal)
        if not found:
            errors.append(f"sample#{idx}: label {label} not in legal set")
    return errors


def _label_matches_legal(label: Dict, legal_actions: List[Dict]) -> bool:
    return replay_label_matches_legal(label, legal_actions)


def _label_to_legal_mjai(label: Dict, actor: int) -> Dict:
    return replay_label_to_legal_mjai(label, actor)
