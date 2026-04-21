"""Runtime/review adapter for Xmodel1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from xmodel1.checkpoint import (
    default_xmodel1_state_scalar_dim,
    infer_xmodel1_model_dims,
    load_xmodel1_checkpoint_state,
    resolve_xmodel1_state_scalar_dim,
    validate_xmodel1_checkpoint_metadata,
)
from xmodel1.features import (
    build_runtime_candidate_arrays,
    empty_history_summary,
    resolve_runtime_history_summary,
    resolve_runtime_tensor_payload,
)
from xmodel1.model import Xmodel1Model
from xmodel1.review_export import (
    ReviewCandidate,
    action_idx_to_action_label,
    tile34_to_action_label,
    topk_candidates_from_row,
    topk_response_candidates_from_row,
)
from xmodel1.schema import (
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_SAMPLE_TYPE_DISCARD,
)


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
        validate_xmodel1_checkpoint_metadata(
            ckpt,
            checkpoint_label=f"Xmodel1 checkpoint {checkpoint_path}",
            allow_legacy_inference=True,
        )
        cfg = ckpt.get("cfg", {})
        state_dict = ckpt["model"]
        inferred_dims = infer_xmodel1_model_dims(
            state_dict,
            cfg=cfg if isinstance(cfg, dict) else None,
        )
        state_tile_channels = int(
            cfg.get("state_tile_channels", inferred_dims["state_tile_channels"])
        )
        state_scalar_dim = resolve_xmodel1_state_scalar_dim(cfg, state_dict)
        candidate_flag_dim = int(cfg.get("candidate_flag_dim", inferred_dims["candidate_flag_dim"]))
        candidate_feature_dim = int(
            cfg.get(
                "candidate_feature_dim",
                inferred_dims["candidate_feature_dim"],
            )
        )
        model = Xmodel1Model(
            state_tile_channels=state_tile_channels,
            state_scalar_dim=state_scalar_dim,
            candidate_feature_dim=candidate_feature_dim,
            candidate_flag_dim=candidate_flag_dim,
            hidden_dim=int(cfg.get("hidden_dim", inferred_dims["hidden_dim"])),
            num_res_blocks=int(cfg.get("num_res_blocks", inferred_dims["num_res_blocks"])),
            dropout=float(cfg.get("dropout", inferred_dims["dropout"])),
        )
        load_xmodel1_checkpoint_state(
            model,
            state_dict,
            checkpoint_label=f"Xmodel1 checkpoint {checkpoint_path}",
        )
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
                moved.get("response_action_idx"),
                moved.get("response_action_mask"),
                moved.get("response_post_candidate_feat"),
                moved.get("response_post_candidate_tile_id"),
                moved.get("response_post_candidate_mask"),
                moved.get("response_post_candidate_flags"),
                history_summary=moved.get("history_summary"),
            )
        return {
            "discard_logits": out.discard_logits.detach().cpu(),
            "response_logits": out.response_logits.detach().cpu(),
            "response_post_logits": out.response_post_logits.detach().cpu(),
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
            "candidate_hard_bad_flag": batch["candidate_hard_bad_flag"].detach().cpu(),
            "sample_type": batch["sample_type"].detach().cpu(),
            "response_action_idx": batch["response_action_idx"].detach().cpu(),
            "response_action_mask": batch["response_action_mask"].detach().cpu(),
            "response_post_candidate_quality_score": batch["response_post_candidate_quality_score"].detach().cpu(),
            "response_post_candidate_hard_bad_flag": batch["response_post_candidate_hard_bad_flag"].detach().cpu(),
            "chosen_response_action_idx": batch["chosen_response_action_idx"].detach().cpu(),
            "response_teacher_discard_idx": batch["response_teacher_discard_idx"].detach().cpu(),
            "history_summary": batch["history_summary"].detach().cpu(),
            "replay_id": list(batch.get("replay_id", [])),
            "sample_id": list(batch.get("sample_id", [])),
        }

    def build_runtime_batch(
        self,
        snap: dict,
        actor: int,
        *,
        state_scalar_dim: int | None = None,
    ) -> dict[str, torch.Tensor]:
        state_scalar_dim = int(
            state_scalar_dim
            or getattr(self.model, "state_scalar_dim", default_xmodel1_state_scalar_dim())
        )
        candidate_feat, candidate_tile_id, candidate_mask, candidate_flags = build_runtime_candidate_arrays(
            snap,
            actor,
            snap.get("legal_actions", []),
            max_candidates=XMODEL1_MAX_CANDIDATES,
            candidate_feature_dim=self.model.candidate_feature_dim,
            candidate_flag_dim=self.model.candidate_flag_dim,
        )
        if all(key in snap for key in ("bakaze", "scores", "melds", "discards", "reached")):
            runtime_payload = resolve_runtime_tensor_payload(
                snap,
                actor,
                snap.get("legal_actions", []),
            )
            response_action_idx = runtime_payload["response_action_idx"]
            response_action_mask = runtime_payload["response_action_mask"]
            response_post_candidate_feat = runtime_payload["response_post_candidate_feat"]
            response_post_candidate_tile_id = runtime_payload["response_post_candidate_tile_id"]
            response_post_candidate_mask = runtime_payload["response_post_candidate_mask"]
            response_post_candidate_flags = runtime_payload["response_post_candidate_flags"]
            response_post_candidate_quality_score = runtime_payload["response_post_candidate_quality_score"]
            response_post_candidate_hard_bad_flag = runtime_payload["response_post_candidate_hard_bad_flag"]
            response_teacher_discard_idx = runtime_payload["response_teacher_discard_idx"]
        else:
            response_action_idx = np.full(
                (XMODEL1_MAX_RESPONSE_CANDIDATES,),
                -1,
                dtype=np.int16,
            )
            response_action_mask = np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES,), dtype=np.uint8)
            response_post_candidate_feat = np.zeros(
                (
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    XMODEL1_MAX_CANDIDATES,
                    self.model.candidate_feature_dim,
                ),
                dtype=np.float32,
            )
            response_post_candidate_tile_id = np.full(
                (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
                -1,
                dtype=np.int16,
            )
            response_post_candidate_mask = np.zeros(
                (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
                dtype=np.uint8,
            )
            response_post_candidate_flags = np.zeros(
                (
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    XMODEL1_MAX_CANDIDATES,
                    self.model.candidate_flag_dim,
                ),
                dtype=np.uint8,
            )
            response_post_candidate_quality_score = np.zeros(
                (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
                dtype=np.float32,
            )
            response_post_candidate_hard_bad_flag = np.zeros(
                (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
                dtype=np.uint8,
            )
            response_teacher_discard_idx = np.full(
                (XMODEL1_MAX_RESPONSE_CANDIDATES,),
                -1,
                dtype=np.int16,
            )
        from xmodel1.features import encode

        state_tile_feat, state_scalar = encode(snap, actor, state_scalar_dim=state_scalar_dim)
        valid_indices = [i for i, flag in enumerate(candidate_mask) if flag > 0]
        chosen_idx = valid_indices[0] if valid_indices else 0
        history_summary = (
            resolve_runtime_history_summary(snap)
            if "history_summary" in snap
            else empty_history_summary()
        )
        return {
            "state_tile_feat": torch.from_numpy(state_tile_feat).unsqueeze(0),
            "state_scalar": torch.from_numpy(state_scalar).unsqueeze(0),
            "candidate_feat": torch.from_numpy(candidate_feat).unsqueeze(0),
            "candidate_tile_id": torch.from_numpy(candidate_tile_id).unsqueeze(0),
            "candidate_mask": torch.from_numpy(candidate_mask).unsqueeze(0),
            "candidate_flags": torch.from_numpy(candidate_flags).unsqueeze(0),
            "chosen_candidate_idx": torch.tensor([chosen_idx], dtype=torch.long),
            "candidate_quality_score": torch.zeros((1, XMODEL1_MAX_CANDIDATES), dtype=torch.float32),
            "candidate_hard_bad_flag": torch.zeros((1, XMODEL1_MAX_CANDIDATES), dtype=torch.float32),
            "response_action_idx": torch.from_numpy(response_action_idx).unsqueeze(0),
            "response_action_mask": torch.from_numpy(response_action_mask).unsqueeze(0),
            "chosen_response_action_idx": torch.tensor([-1], dtype=torch.long),
            "response_post_candidate_feat": torch.from_numpy(response_post_candidate_feat).unsqueeze(0),
            "response_post_candidate_tile_id": torch.from_numpy(response_post_candidate_tile_id).unsqueeze(0),
            "response_post_candidate_mask": torch.from_numpy(response_post_candidate_mask).unsqueeze(0),
            "response_post_candidate_flags": torch.from_numpy(response_post_candidate_flags).unsqueeze(0),
            "response_post_candidate_quality_score": torch.from_numpy(response_post_candidate_quality_score).unsqueeze(0),
            "response_post_candidate_hard_bad_flag": torch.from_numpy(response_post_candidate_hard_bad_flag).unsqueeze(0),
            "response_teacher_discard_idx": torch.from_numpy(response_teacher_discard_idx).unsqueeze(0),
            "history_summary": torch.from_numpy(history_summary).unsqueeze(0).float(),
        }

    def scored_row_to_review(self, scored: dict[str, torch.Tensor], row_index: int, *, k: int = 3) -> Xmodel1ScoredRow:
        sample_type = int(scored["sample_type"][row_index].item()) if "sample_type" in scored else XMODEL1_SAMPLE_TYPE_DISCARD
        if sample_type == XMODEL1_SAMPLE_TYPE_DISCARD:
            logits = scored["discard_logits"][row_index].tolist()
            tile_ids = scored["candidate_tile_id"][row_index].tolist()
            mask = scored["candidate_mask"][row_index].tolist()
            quality = scored["candidate_quality_score"][row_index].tolist()
            hard_bad = scored["candidate_hard_bad_flag"][row_index].tolist()
            top_k = topk_candidates_from_row(
                scores=logits,
                candidate_tile_ids=tile_ids,
                candidate_mask=mask,
                quality_scores=quality,
                hard_bad_flags=hard_bad,
                k=k,
            )
            chosen_idx = int(scored["chosen_candidate_idx"][row_index].item())
            chosen_action = (
                tile34_to_action_label(int(tile_ids[chosen_idx]))
                if 0 <= chosen_idx < len(tile_ids)
                else "padding"
            )
        else:
            response_logits = scored["response_logits"][row_index].tolist()
            response_action_idx = scored["response_action_idx"][row_index].tolist()
            response_mask = scored["response_action_mask"][row_index].tolist()
            response_quality = []
            response_hard_bad = []
            for slot, active in enumerate(response_mask):
                if int(active) <= 0:
                    response_quality.append(0.0)
                    response_hard_bad.append(0)
                    continue
                teacher_idx = int(scored["response_teacher_discard_idx"][row_index, slot].item())
                if teacher_idx >= 0:
                    response_quality.append(
                        float(scored["response_post_candidate_quality_score"][row_index, slot, teacher_idx].item())
                    )
                    response_hard_bad.append(
                        int(scored["response_post_candidate_hard_bad_flag"][row_index, slot, teacher_idx].item())
                    )
                else:
                    response_quality.append(0.0)
                    response_hard_bad.append(0)
            top_k = topk_response_candidates_from_row(
                scores=response_logits,
                response_action_idx=response_action_idx,
                response_mask=response_mask,
                quality_scores=response_quality,
                hard_bad_flags=response_hard_bad,
                k=k,
            )
            chosen_idx = int(scored["chosen_response_action_idx"][row_index].item())
            chosen_action = (
                action_idx_to_action_label(int(response_action_idx[chosen_idx]))
                if 0 <= chosen_idx < len(response_action_idx)
                else "padding"
            )
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
