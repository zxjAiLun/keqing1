#!/usr/bin/env python3
"""Small evidence harness for keqingv4 special-action calibration.

Runs a few fixed synthetic special-action cases twice:
1. calibration disabled
2. calibration enabled

Outputs a JSON report showing chosen action and candidate scores before/after.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate keqingv4 special-action calibration on fixed synthetic cases")
    parser.add_argument("--output", required=True, help="Path to write JSON evidence report")
    return parser.parse_args()


@dataclass
class FakeForward:
    logits: np.ndarray
    value: float = 0.0
    score_delta: float = 0.0
    win_prob: float = 0.0
    dealin_prob: float = 0.0


def _policy_with_scores(action_to_idx, scores: dict[tuple[str, str | None], float]) -> np.ndarray:
    logits = np.full((45,), -1e9, dtype=np.float32)
    for (atype, pai), score in scores.items():
        action = {"type": atype}
        if pai is not None:
            action["pai"] = pai
        if atype == "dahai":
            action["tsumogiri"] = False
        if atype == "reach":
            action["actor"] = 0
        logits[action_to_idx(action)] = score
    return logits


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    import inference.scoring as inference_scoring
    from inference import DecisionContext, DefaultActionScorer, ModelAuxOutputs, ModelForwardResult
    from mahjong_env.action_space import action_to_idx

    class FakeV4Adapter:
        model_version = "keqingv4"

        def __init__(self, forwards: list[FakeForward], special_summary: np.ndarray):
            self._forwards = list(forwards)
            self._special_summary = special_summary

        def forward(self, snap: dict, actor: int):
            forward = self._forwards.pop(0)
            return ModelForwardResult(
                policy_logits=forward.logits,
                value=forward.value,
                aux=ModelAuxOutputs(
                    score_delta=forward.score_delta,
                    win_prob=forward.win_prob,
                    dealin_prob=forward.dealin_prob,
                ),
            )

        def resolve_runtime_v4_summaries(self, snap: dict, actor: int, legal_actions=None):
            return (
                np.zeros((34, 14), dtype=np.float32),
                np.zeros((8, 14), dtype=np.float32),
                self._special_summary,
            )

    def run_case(name: str, ctx: DecisionContext, forwards: list[FakeForward], special_summary: np.ndarray, patch_reach=None):
        report: dict[str, object] = {"case": name}
        original_reach = inference_scoring._reach_discard_candidates
        if patch_reach is not None:
            inference_scoring._reach_discard_candidates = patch_reach
        try:
            before_adapter = FakeV4Adapter(list(forwards), special_summary.copy())
            after_adapter = FakeV4Adapter(list(forwards), special_summary.copy())
            before = DefaultActionScorer(
                adapter=before_adapter,
                beam_k=1,
                beam_lambda=1.0,
                style_lambda=0.0,
                score_delta_lambda=0.0,
                win_prob_lambda=0.0,
                dealin_prob_lambda=0.0,
                special_calibration_scale=0.0,
            ).score(ctx)
            after = DefaultActionScorer(
                adapter=after_adapter,
                beam_k=1,
                beam_lambda=1.0,
                style_lambda=0.0,
                score_delta_lambda=0.0,
                win_prob_lambda=0.0,
                dealin_prob_lambda=0.0,
                special_calibration_scale=1.0,
            ).score(ctx)
        finally:
            inference_scoring._reach_discard_candidates = original_reach

        report["before"] = {
            "chosen": before.chosen,
            "candidates": [
                {
                    "action": c.action,
                    "final_score": c.final_score,
                    "meta": c.meta,
                }
                for c in before.candidates
            ],
        }
        report["after"] = {
            "chosen": after.chosen,
            "candidates": [
                {
                    "action": c.action,
                    "final_score": c.final_score,
                    "meta": c.meta,
                }
                for c in after.candidates
            ],
        }
        return report

    reports: list[dict[str, object]] = []

    reach_special = np.zeros((3, 14), dtype=np.float32)
    reach_special[0, 1] = 1.0
    reach_special[0, 2] = 0.35
    reach_special[0, 4] = 0.6
    reach_special[0, 6] = 0.2
    reach_special[0, 12] = 1.0
    reach_special[0, 13] = 1.0
    reports.append(
        run_case(
            "reach_decl",
            DecisionContext(
                actor=0,
                event={"type": "tsumo", "actor": 0, "pai": "4m"},
                runtime_snap={
                    "hand": ["4m"],
                    "discards": [[], [], [], []],
                    "reached": [False, False, False, False],
                    "last_tsumo": ["4m", None, None, None],
                    "last_tsumo_raw": ["4m", None, None, None],
                },
                model_snap={
                    "hand": [],
                    "tsumo_pai": "4m",
                    "reached": [False, False, False, False],
                    "last_tsumo": ["4m", None, None, None],
                    "last_tsumo_raw": ["4m", None, None, None],
                },
                legal_actions=[
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True},
                ],
            ),
            forwards=[
                FakeForward(_policy_with_scores(action_to_idx, {("reach", None): 0.20, ("dahai", "4m"): 0.35})),
                FakeForward(_policy_with_scores(action_to_idx, {}), value=0.3),
            ],
            special_summary=reach_special,
            patch_reach=lambda hand, last_tsumo, last_tsumo_raw: [("4m", False)],
        )
    )

    hora_special = np.zeros((3, 14), dtype=np.float32)
    hora_special[1, 2] = 1.0
    hora_special[1, 4] = 1.0
    hora_special[1, 12] = 1.0
    hora_special[1, 13] = 1.0
    reports.append(
        run_case(
            "hora_finish",
            DecisionContext(
                actor=0,
                event={"type": "tsumo", "actor": 0, "pai": "4p"},
                runtime_snap={"hand": ["4p"], "tsumo_pai": "4p"},
                model_snap={"hand": [], "tsumo_pai": "4p"},
                legal_actions=[
                    {"type": "hora", "actor": 0, "target": 0, "pai": "4p"},
                    {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
                ],
            ),
            forwards=[
                FakeForward(_policy_with_scores(action_to_idx, {("hora", "4p"): 0.25, ("dahai", "4p"): 0.38})),
            ],
            special_summary=hora_special,
        )
    )

    ryu_special = np.zeros((3, 14), dtype=np.float32)
    ryu_special[2, 2] = 1.0
    ryu_special[2, 3] = 1.0
    ryu_special[2, 4] = 1.4
    ryu_special[2, 12] = 1.0
    ryu_special[2, 13] = 1.0
    reports.append(
        run_case(
            "abortive_ryukyoku",
            DecisionContext(
                actor=0,
                event={"type": "tsumo", "actor": 0, "pai": "5m"},
                runtime_snap={"hand": ["1m", "9m"], "tsumo_pai": "5m"},
                model_snap={"hand": ["1m", "9m"], "tsumo_pai": "5m"},
                legal_actions=[
                    {"type": "ryukyoku"},
                    {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
                ],
            ),
            forwards=[
                FakeForward(_policy_with_scores(action_to_idx, {("ryukyoku", None): 0.10, ("dahai", "9m"): 0.45})),
            ],
            special_summary=ryu_special,
        )
    )

    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": reports}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote special calibration evidence -> {output_path}")


if __name__ == "__main__":
    main()
