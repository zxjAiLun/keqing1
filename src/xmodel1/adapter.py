"""Runtime/review adapter for Xmodel1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from evals.xmodel1.review_export import ReviewCandidate, tile34_to_action_label, topk_candidates_from_row
from xmodel1.candidate_quality import build_special_candidate_arrays
from xmodel1.features import (
    EVENT_HISTORY_FEATURE_DIM,
    EVENT_HISTORY_LEN,
    build_runtime_candidate_arrays,
    empty_event_history,
    resolve_runtime_event_history,
)
from xmodel1.model import Xmodel1Model
from xmodel1.schema import XMODEL1_MAX_CANDIDATES, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM


@dataclass(frozen=True)
class Xmodel1ScoredRow:
    """Review 行级评分结果。

    Stage 1 起:MC 形态的 ``score_delta`` / ``offense_quality`` 已删除,
    改由 ``win_prob`` / ``dealin_prob`` + ``pts_given_win`` /
    ``pts_given_dealin`` 组合出 composed_ev。Stage 3 runtime 直通后,
    review 会直接消费 composed_ev 与 per-candidate 分数。
    """

    top_k: tuple[ReviewCandidate, ...]
    chosen_action: str
    win_prob: float
    dealin_prob: float
    pts_given_win: float
    pts_given_dealin: float
    composed_ev: float


class Xmodel1Adapter:
    """Thin checkpoint-backed inference adapter for review-oriented scoring."""

    def __init__(self, model: Xmodel1Model, *, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> "Xmodel1Adapter":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        state_dict = ckpt["model"]
        state_tile_channels = int(
            cfg.get("state_tile_channels", state_dict["state_proj.0.weight"].shape[1])
        )
        state_scalar_dim = int(
            cfg.get("state_scalar_dim", state_dict["scalar_proj.0.weight"].shape[1])
        )
        candidate_flag_dim = int(cfg.get("candidate_flag_dim", 10))
        candidate_feature_dim = int(
            cfg.get(
                "candidate_feature_dim",
                state_dict["candidate_proj.0.weight"].shape[1] - 16 - candidate_flag_dim,
            )
        )
        model = Xmodel1Model(
            state_tile_channels=state_tile_channels,
            state_scalar_dim=state_scalar_dim,
            candidate_feature_dim=candidate_feature_dim,
            candidate_flag_dim=candidate_flag_dim,
            special_candidate_feature_dim=int(cfg.get("special_candidate_feature_dim", XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_res_blocks=int(cfg.get("num_res_blocks", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        model.load_state_dict(state_dict, strict=False)
        return cls(model, device=device)

    def score_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if self.device.type != "cuda" and value.dtype == torch.float16:
                    value = value.float()
            moved[key] = value
        with torch.no_grad():
            out = self.model(
                moved["state_tile_feat"],
                moved["state_scalar"],
                moved["candidate_feat"],
                moved["candidate_tile_id"],
                moved["candidate_flags"],
                moved["candidate_mask"],
                moved.get("special_candidate_feat"),
                moved.get("special_candidate_type_id"),
                moved.get("special_candidate_mask"),
                event_history=moved.get("event_history"),
            )
        return {
            "discard_logits": out.discard_logits.detach().cpu(),
            "special_logits": out.special_logits.detach().cpu(),
            "action_logits": out.action_logits.detach().cpu(),
            "win_logit": out.win_logit.detach().cpu(),
            "dealin_logit": out.dealin_logit.detach().cpu(),
            "pts_given_win": out.pts_given_win.detach().cpu(),
            "pts_given_dealin": out.pts_given_dealin.detach().cpu(),
            "opp_tenpai_logits": out.opp_tenpai_logits.detach().cpu(),
            "chosen_candidate_idx": batch["chosen_candidate_idx"].detach().cpu(),
            "candidate_tile_id": batch["candidate_tile_id"].detach().cpu(),
            "candidate_mask": batch["candidate_mask"].detach().cpu(),
            "candidate_quality_score": batch["candidate_quality_score"].detach().cpu(),
            "candidate_rank_bucket": batch["candidate_rank_bucket"].detach().cpu(),
            "candidate_hard_bad_flag": batch["candidate_hard_bad_flag"].detach().cpu(),
            "special_candidate_type_id": batch["special_candidate_type_id"].detach().cpu(),
            "special_candidate_mask": batch["special_candidate_mask"].detach().cpu(),
            "chosen_special_candidate_idx": batch["chosen_special_candidate_idx"].detach().cpu(),
            "event_history": batch["event_history"].detach().cpu(),
        }

    def build_runtime_batch(
        self,
        snap: dict,
        actor: int,
        *,
        state_scalar_dim: int | None = None,
    ) -> dict[str, torch.Tensor]:
        state_scalar_dim = int(state_scalar_dim or self.model.state_scalar_dim)
        candidate_feat, candidate_tile_id, candidate_mask, candidate_flags = build_runtime_candidate_arrays(
            snap,
            actor,
            max_candidates=XMODEL1_MAX_CANDIDATES,
            candidate_feature_dim=self.model.candidate_feature_dim,
            candidate_flag_dim=self.model.candidate_flag_dim,
        )
        (
            special_candidate_feat,
            special_candidate_type_id,
            special_candidate_mask,
            special_candidate_quality_score,
            special_candidate_rank_bucket,
            special_candidate_hard_bad_flag,
            chosen_special_candidate_idx,
        ) = build_special_candidate_arrays(
            snap,
            actor,
            snap.get("legal_actions", []),
            None,
            include_terminal_actions=True,
        )
        from xmodel1.features import encode

        state_tile_feat, state_scalar = encode(snap, actor, state_scalar_dim=state_scalar_dim)
        valid_indices = [i for i, flag in enumerate(candidate_mask) if flag > 0]
        chosen_idx = valid_indices[0] if valid_indices else 0
        event_history = resolve_runtime_event_history(snap) if "event_history" in snap else empty_event_history()
        assert event_history.shape == (EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM)
        return {
            "state_tile_feat": torch.from_numpy(state_tile_feat).unsqueeze(0),
            "state_scalar": torch.from_numpy(state_scalar).unsqueeze(0),
            "candidate_feat": torch.from_numpy(candidate_feat).unsqueeze(0),
            "candidate_tile_id": torch.from_numpy(candidate_tile_id).unsqueeze(0),
            "candidate_mask": torch.from_numpy(candidate_mask).unsqueeze(0),
            "candidate_flags": torch.from_numpy(candidate_flags).unsqueeze(0),
            "chosen_candidate_idx": torch.tensor([chosen_idx], dtype=torch.long),
            "candidate_quality_score": torch.zeros((1, XMODEL1_MAX_CANDIDATES), dtype=torch.float32),
            "candidate_rank_bucket": torch.zeros((1, XMODEL1_MAX_CANDIDATES), dtype=torch.long),
            "candidate_hard_bad_flag": torch.zeros((1, XMODEL1_MAX_CANDIDATES), dtype=torch.float32),
            "special_candidate_feat": torch.from_numpy(special_candidate_feat).unsqueeze(0),
            "special_candidate_type_id": torch.from_numpy(special_candidate_type_id).unsqueeze(0),
            "special_candidate_mask": torch.from_numpy(special_candidate_mask).unsqueeze(0),
            "special_candidate_quality_score": torch.from_numpy(special_candidate_quality_score).unsqueeze(0),
            "special_candidate_rank_bucket": torch.from_numpy(special_candidate_rank_bucket).unsqueeze(0),
            "special_candidate_hard_bad_flag": torch.from_numpy(special_candidate_hard_bad_flag).unsqueeze(0),
            "chosen_special_candidate_idx": torch.tensor([int(chosen_special_candidate_idx)], dtype=torch.long),
            "event_history": torch.from_numpy(event_history).unsqueeze(0).long(),
        }

    def scored_row_to_review(self, scored: dict[str, torch.Tensor], row_index: int, *, k: int = 3) -> Xmodel1ScoredRow:
        logits = scored["discard_logits"][row_index].tolist()
        tile_ids = scored["candidate_tile_id"][row_index].tolist()
        mask = scored["candidate_mask"][row_index].tolist()
        quality = scored["candidate_quality_score"][row_index].tolist()
        rank = scored["candidate_rank_bucket"][row_index].tolist()
        hard_bad = scored["candidate_hard_bad_flag"][row_index].tolist()
        top_k = topk_candidates_from_row(
            scores=logits,
            candidate_tile_ids=tile_ids,
            candidate_mask=mask,
            quality_scores=quality,
            rank_buckets=rank,
            hard_bad_flags=hard_bad,
            k=k,
        )
        chosen_idx = int(scored["chosen_candidate_idx"][row_index].item())
        chosen_action = tile34_to_action_label(int(tile_ids[chosen_idx]))
        win_prob = float(torch.sigmoid(scored["win_logit"][row_index]).item())
        dealin_prob = float(torch.sigmoid(scored["dealin_logit"][row_index]).item())
        pts_win = float(scored["pts_given_win"][row_index].item())
        pts_dealin = float(scored["pts_given_dealin"][row_index].item())
        composed_ev = win_prob * pts_win - dealin_prob * pts_dealin
        return Xmodel1ScoredRow(
            top_k=top_k,
            chosen_action=chosen_action,
            win_prob=win_prob,
            dealin_prob=dealin_prob,
            pts_given_win=pts_win,
            pts_given_dealin=pts_dealin,
            composed_ev=composed_ev,
        )


__all__ = ["Xmodel1Adapter", "Xmodel1ScoredRow"]
