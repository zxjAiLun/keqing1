"""Runtime/review adapter for Xmodel1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from evals.xmodel1.review_export import ReviewCandidate, tile34_to_action_label, topk_candidates_from_row
from xmodel1.features import build_runtime_candidate_arrays
from xmodel1.model import Xmodel1Model
from xmodel1.schema import XMODEL1_MAX_CANDIDATES


@dataclass(frozen=True)
class Xmodel1ScoredRow:
    top_k: tuple[ReviewCandidate, ...]
    chosen_action: str
    score_delta: float
    win_prob: float
    dealin_prob: float
    offense_quality: float


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
        candidate_feature_dim = int(
            cfg.get("candidate_feature_dim", state_dict["candidate_proj.0.weight"].shape[1] - 16 - 10)
        )
        candidate_flag_dim = int(cfg.get("candidate_flag_dim", 10))
        model = Xmodel1Model(
            state_tile_channels=state_tile_channels,
            state_scalar_dim=state_scalar_dim,
            candidate_feature_dim=candidate_feature_dim,
            candidate_flag_dim=candidate_flag_dim,
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
                moved["candidate_flags"],
                moved["candidate_mask"],
                moved["candidate_tile_id"],
            )
        return {
            "discard_logits": out.discard_logits.detach().cpu(),
            "score_delta": out.score_delta.detach().cpu(),
            "win_logit": out.win_logit.detach().cpu(),
            "dealin_logit": out.dealin_logit.detach().cpu(),
            "offense_quality": out.offense_quality.detach().cpu(),
            "chosen_candidate_idx": batch["chosen_candidate_idx"].detach().cpu(),
            "candidate_tile_id": batch["candidate_tile_id"].detach().cpu(),
            "candidate_mask": batch["candidate_mask"].detach().cpu(),
            "candidate_quality_score": batch["candidate_quality_score"].detach().cpu(),
            "candidate_rank_bucket": batch["candidate_rank_bucket"].detach().cpu(),
            "candidate_hard_bad_flag": batch["candidate_hard_bad_flag"].detach().cpu(),
        }

    def build_runtime_batch(
        self,
        snap: dict,
        actor: int,
        *,
        state_scalar_dim: int = 56,
    ) -> dict[str, torch.Tensor]:
        del state_scalar_dim
        candidate_arrays = build_runtime_candidate_arrays(
            snap,
            actor,
            max_candidates=XMODEL1_MAX_CANDIDATES,
            candidate_feature_dim=self.model.candidate_feature_dim,
            candidate_flag_dim=self.model.candidate_flag_dim,
        )
        from xmodel1.features import encode

        state_tile_feat, state_scalar = encode(snap, actor)
        candidate_feat = candidate_arrays["candidate_feat"]
        candidate_tile_id = candidate_arrays["candidate_tile_id"]
        candidate_mask = candidate_arrays["candidate_mask"]
        candidate_flags = candidate_arrays["candidate_flags"]
        valid_indices = [i for i, flag in enumerate(candidate_mask) if flag > 0]
        chosen_idx = valid_indices[0] if valid_indices else 0
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
        return Xmodel1ScoredRow(
            top_k=top_k,
            chosen_action=chosen_action,
            score_delta=float(scored["score_delta"][row_index].item()),
            win_prob=float(torch.sigmoid(scored["win_logit"][row_index]).item()),
            dealin_prob=float(torch.sigmoid(scored["dealin_logit"][row_index]).item()),
            offense_quality=float(scored["offense_quality"][row_index].item()),
        )


__all__ = ["Xmodel1Adapter", "Xmodel1ScoredRow"]
