from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from libriichi.dataset import GameplayLoader


MORTAL_DISCARD_ID_TO_TILE = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
)
CALL_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}


@dataclass(frozen=True)
class RiskGateConfig:
    enabled: bool
    base_weight: float
    risk_weight: float
    disable_after_fuuro_discard: bool
    disable_vs_riichi_discard: bool
    disable_after_fuuro_vs_riichi_discard: bool
    disable_dealer_or_leading: bool
    disable_start_rank_1: bool


@dataclass
class DecisionContext:
    action: dict[str, Any]
    action_ids: tuple[int, ...]
    weight: float
    reasons: tuple[str, ...]


def risk_gate_config_from_mapping(teacher_config: Mapping[str, Any]) -> RiskGateConfig:
    risk_gate = teacher_config.get("risk_gate", {})
    return RiskGateConfig(
        enabled=bool(risk_gate.get("enabled", False)),
        base_weight=float(risk_gate.get("base_weight", teacher_config.get("ce_weight", 0.0))),
        risk_weight=float(risk_gate.get("risk_weight", 0.0)),
        disable_after_fuuro_discard=bool(risk_gate.get("disable_after_fuuro_discard", True)),
        disable_vs_riichi_discard=bool(risk_gate.get("disable_vs_riichi_discard", True)),
        disable_after_fuuro_vs_riichi_discard=bool(
            risk_gate.get("disable_after_fuuro_vs_riichi_discard", True)
        ),
        disable_dealer_or_leading=bool(risk_gate.get("disable_dealer_or_leading", True)),
        disable_start_rank_1=bool(risk_gate.get("disable_start_rank_1", True)),
    )


def normalize_tile(tile: str) -> str:
    if tile in {"5mr", "5pr", "5sr"}:
        return tile[0] + tile[2]
    return tile


def discard_action_ids(tile: str) -> tuple[int, ...]:
    exact = tuple(i for i, mortal_tile in enumerate(MORTAL_DISCARD_ID_TO_TILE) if mortal_tile == tile)
    if exact:
        return exact
    normalized = normalize_tile(tile)
    return tuple(
        i for i, mortal_tile in enumerate(MORTAL_DISCARD_ID_TO_TILE) if normalize_tile(mortal_tile) == normalized
    )


def score_ranks(scores: Sequence[int]) -> list[int]:
    ordered = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(ordered, 1):
        ranks[seat] = rank
    return ranks


def score_bucket(scores: Sequence[int], seat: int) -> str:
    best_other = max(int(scores[other]) for other in range(4) if other != int(seat))
    diff = int(scores[int(seat)]) - best_other
    if diff >= 12000:
        return "ahead_big"
    if diff > 0:
        return "ahead_small"
    if diff >= -8000:
        return "near_even"
    if diff >= -18000:
        return "behind_small"
    return "behind_big"


def iter_events(path: str | Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def selected_action_id_from_meta(meta: Mapping[str, Any]) -> int | None:
    mask_bits = int(meta.get("mask_bits", 0) or 0)
    q_values = [float(value) for value in (meta.get("q_values") or [])]
    action_ids = [action_id for action_id in range(46) if mask_bits & (1 << action_id)]
    if not q_values or len(action_ids) != len(q_values):
        return None
    best = max(range(len(q_values)), key=lambda index: q_values[index])
    return int(action_ids[best])


def parse_decision_contexts_by_player(
    path: str | Path,
    *,
    player_names: Sequence[str],
    gate: RiskGateConfig,
) -> dict[int, list[DecisionContext]]:
    names: list[str] = [str(i) for i in range(4)]
    target_names = set(str(name) for name in player_names)
    target_seats: set[int] = set()
    contexts: dict[int, list[DecisionContext]] = {}
    oya = 0
    scores = [25000, 25000, 25000, 25000]
    start_ranks = [1, 2, 3, 4]
    riichi = [False, False, False, False]
    fuuro = [False, False, False, False]
    finished_target_seats: set[int] = set()

    for event in iter_events(path):
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            names = [str(name) for name in event.get("names", names)]
            target_seats = {seat for seat, name in enumerate(names) if name in target_names}
            contexts = {seat: [] for seat in target_seats}
            continue
        if event_type == "start_kyoku":
            oya = int(event.get("oya", 0) or 0)
            raw_scores = event.get("scores", scores)
            if isinstance(raw_scores, list) and len(raw_scores) >= 4:
                scores = [int(raw_scores[seat]) for seat in range(4)]
            start_ranks = score_ranks(scores)
            riichi = [False, False, False, False]
            fuuro = [False, False, False, False]
            continue

        actor = event.get("actor")
        if not isinstance(actor, int) or not 0 <= actor < 4:
            continue

        if actor in target_seats and actor not in finished_target_seats and isinstance(event.get("meta"), Mapping):
            action = dict(event)
            if event_type == "dahai":
                action_ids = discard_action_ids(str(action.get("pai", "")))
                if not action_ids:
                    continue
            else:
                action_id = selected_action_id_from_meta(action["meta"])
                if action_id is None:
                    continue
                action_ids = (int(action_id),)
            reasons: list[str] = []
            if event_type == "dahai" and any(action_id < 37 for action_id in action_ids):
                opponent_riichi = any(riichi[other] for other in range(4) if other != actor)
                bucket = score_bucket(scores, actor)
                if gate.disable_after_fuuro_discard and fuuro[actor]:
                    reasons.append("after_fuuro_discard")
                if gate.disable_vs_riichi_discard and opponent_riichi:
                    reasons.append("vs_riichi_discard")
                if gate.disable_after_fuuro_vs_riichi_discard and fuuro[actor] and opponent_riichi:
                    reasons.append("after_fuuro_vs_riichi_discard")
                if gate.disable_dealer_or_leading and (actor == oya or bucket == "ahead_big"):
                    reasons.append("dealer_or_ahead_big")
                if gate.disable_start_rank_1 and start_ranks[actor] == 1:
                    reasons.append("start_rank_1")
            weight = gate.risk_weight if reasons else gate.base_weight
            contexts.setdefault(actor, []).append(
                DecisionContext(action=action, action_ids=tuple(int(action_id) for action_id in action_ids), weight=float(weight), reasons=tuple(reasons))
            )
            if any(action_id in {43, 44} for action_id in action_ids):
                finished_target_seats.add(actor)

        if event_type == "reach":
            riichi[actor] = True
        elif event_type in CALL_TYPES:
            fuuro[actor] = True

    return contexts


def build_teacher_ce_weights_for_game(
    *,
    actions: Sequence[int],
    file_path: str | Path,
    player_id: int,
    player_names: Sequence[str],
    gate: RiskGateConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    contexts = parse_decision_contexts_by_player(file_path, player_names=player_names, gate=gate).get(player_id, [])
    weights = np.full(len(actions), float(gate.base_weight), dtype=np.float32)
    context_index = 0
    mismatches: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    gated = 0
    discard_count = 0
    extra_loader_decisions = 0

    for sample_index, raw_action in enumerate(actions):
        action_id = int(raw_action)
        if context_index < len(contexts) and action_id in contexts[context_index].action_ids:
            context = contexts[context_index]
            context_index += 1
            weights[sample_index] = float(context.weight)
            if action_id < 37:
                discard_count += 1
            if context.reasons:
                gated += 1
                for reason in context.reasons:
                    reason_counts[reason] += 1
            continue
        if action_id != 45:
            extra_loader_decisions += 1

    if context_index != len(contexts):
        mismatches.append(
            {
                "loader_matched_decision_count": context_index,
                "raw_explicit_decision_count": len(contexts),
                "next_raw_action": contexts[context_index].action if context_index < len(contexts) else None,
                "next_expected_action_ids": list(contexts[context_index].action_ids)
                if context_index < len(contexts)
                else None,
                "reason": "unused_raw_explicit_decision",
            }
        )
    return weights, {
        "sample_count": len(actions),
        "discard_count": int(discard_count),
        "raw_dahai_count": sum(1 for context in contexts if context.action.get("type") == "dahai"),
        "raw_explicit_decision_count": len(contexts),
        "matched_explicit_decision_count": context_index,
        "extra_loader_decision_count": int(extra_loader_decisions),
        "gated_count": int(gated),
        "base_count": int(len(actions) - gated),
        "gated_discard_count": int(gated),
        "base_discard_count": int(discard_count - gated),
        "disabled_discard_count": int(gated) if float(gate.risk_weight) == 0.0 else 0,
        "active_count": int(np.count_nonzero(weights > 0)),
        "disabled_count": int(np.count_nonzero(weights == 0)),
        "weight_mean": float(weights.mean()) if len(weights) else 0.0,
        "reason_counts": dict(reason_counts),
        "mismatches": mismatches,
    }


class RiskGatedFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version: int,
        file_list: list[str],
        pts: Sequence[float],
        *,
        oracle: bool = False,
        file_batch_size: int = 20,
        reserve_ratio: float = 0,
        player_names: Sequence[str] | None = None,
        excludes: Sequence[str] | None = None,
        num_epochs: int = 1,
        enable_augmentation: bool = False,
        augmented_first: bool = False,
        gate: RiskGateConfig,
    ) -> None:
        super().__init__()
        if enable_augmentation:
            raise ValueError("risk-gated teacher CE does not support dataset augmentation")
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = list(player_names or [])
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.gate = gate
        self.iterator = None

    def build_iter(self):
        from config import config  # noqa: PLC0415
        from model import GRP  # noqa: PLC0415
        from reward_calculator import RewardCalculator  # noqa: PLC0415

        self.grp = GRP(**config["grp"]["network"])
        grp_state = torch.load(config["grp"]["state_file"], weights_only=True, map_location=torch.device("cpu"))
        self.grp.load_state_dict(grp_state["model"])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(False)

    def load_files(self, augmented: bool):
        random.shuffle(self.file_list)
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer = []
        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            batch_files = self.file_list[start_idx : start_idx + self.file_batch_size]
            self.populate_buffer(batch_files)
            buffer_size = len(self.buffer)
            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue
            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list: Sequence[str]) -> None:
        data = self.loader.load_gz_log_files(list(file_list))
        for file_path, file_data in zip(file_list, data, strict=True):
            for game in file_data:
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                grp = game.take_grp()
                player_id = int(game.take_player_id())
                game_size = len(obs)
                teacher_ce_weights, summary = build_teacher_ce_weights_for_game(
                    actions=actions,
                    file_path=file_path,
                    player_id=player_id,
                    player_names=self.player_names,
                    gate=self.gate,
                )
                if summary["mismatches"]:
                    raise RuntimeError(
                        f"risk-gated teacher CE alignment failed for {file_path} player_id={player_id}: "
                        f"{summary['mismatches'][:3]}"
                    )

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                        teacher_ce_weights[i],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator
