from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from inference.contracts import ModelAuxOutputs, ModelForwardResult, Xmodel1RuntimeOutputs
from inference.pt_map import placement_utility_from_outputs
from mahjong_env.action_space import (
    ANKAN_IDX,
    CHI_HIGH_IDX,
    CHI_LOW_IDX,
    CHI_MID_IDX,
    DAIMINKAN_IDX,
    HORA_IDX,
    KAKAN_IDX,
    NONE_IDX,
    PON_IDX,
    REACH_IDX,
    RYUKYOKU_IDX,
    action_to_idx,
)
from mahjong_env.legal_actions import enumerate_legal_actions
from keqingv4.checkpoint import (
    infer_keqingv4_input_dims,
    load_keqingv4_checkpoint_state,
    validate_keqingv4_checkpoint_metadata,
)
from xmodel1.checkpoint import (
    default_xmodel1_state_scalar_dim,
    infer_xmodel1_model_dims,
    load_xmodel1_checkpoint_state,
    resolve_xmodel1_state_scalar_dim,
    validate_xmodel1_checkpoint_metadata,
)
from training.cache_schema import (
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_SUMMARY_DIM,
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.schema import (
    XMODEL1_SPECIAL_TYPE_ANKAN,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_DAIMINKAN,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_KAKAN,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
    XMODEL1_SPECIAL_TYPE_RYUKYOKU,
)


class KeqingModelAdapter:
    def __init__(
        self,
        *,
        model_version: str,
        model: torch.nn.Module,
        encode_fn,
        device: torch.device,
    ):
        self.model_version = model_version
        self.model = model
        self._encode = encode_fn
        self.device = device
        self._runtime_candidate_builder = None
        self._runtime_special_candidate_builder = None
        self._runtime_v4_summary_builder = None
        self._runtime_v4_summary_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._placement_rank_bonus: tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0)
        self._placement_rank_bonus_norm: float = 90.0
        self._placement_rank_score_scale: float = 0.0

    @staticmethod
    def _resolve_placement_runtime_cfg(cfg: object) -> tuple[tuple[float, float, float, float], float, float]:
        placement_cfg = cfg.get("placement", {}) if isinstance(cfg, dict) else {}
        rank_bonus_raw = placement_cfg.get("rank_bonus", (90.0, 45.0, 0.0, -135.0))
        if not isinstance(rank_bonus_raw, (list, tuple)) or len(rank_bonus_raw) != 4:
            rank_bonus = (90.0, 45.0, 0.0, -135.0)
        else:
            rank_bonus = tuple(float(value) for value in rank_bonus_raw)
        rank_bonus_norm = float(placement_cfg.get("rank_bonus_norm", 90.0))
        rank_score_scale = float(placement_cfg.get("rank_score_scale", 0.0))
        return rank_bonus, rank_bonus_norm, rank_score_scale

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        *,
        device: torch.device,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        model_version: Optional[str] = None,
    ) -> "KeqingModelAdapter":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        inferred_version = model_version or ckpt.get("model_version") or cfg.get("model_name")
        if inferred_version is None and any(key.startswith("candidate_proj.") for key in state_dict):
            inferred_version = "xmodel1"
        if inferred_version is None and any(key.startswith("call_state_proj.") for key in state_dict):
            inferred_version = "keqingv4"
        if inferred_version == "xmodel1":
            validate_xmodel1_checkpoint_metadata(
                ckpt,
                checkpoint_label=f"xmodel1 checkpoint {model_path}",
                allow_legacy_inference=True,
            )
            inferred_dims = infer_xmodel1_model_dims(
                state_dict,
                cfg=cfg if isinstance(cfg, dict) else None,
            )
            features_mod = importlib.import_module("xmodel1.features")
            model_mod = importlib.import_module("xmodel1.model")
            model = model_mod.Xmodel1Model(
                state_tile_channels=int(cfg.get("state_tile_channels", inferred_dims["state_tile_channels"])),
                state_scalar_dim=int(resolve_xmodel1_state_scalar_dim(cfg, state_dict)),
                candidate_feature_dim=int(cfg.get("candidate_feature_dim", inferred_dims["candidate_feature_dim"])),
                candidate_flag_dim=int(cfg.get("candidate_flag_dim", inferred_dims["candidate_flag_dim"])),
                hidden_dim=int(cfg.get("hidden_dim", inferred_dims["hidden_dim"])),
                num_res_blocks=int(cfg.get("num_res_blocks", inferred_dims["num_res_blocks"])),
                dropout=float(cfg.get("dropout", inferred_dims["dropout"])),
            )
            load_xmodel1_checkpoint_state(
                model,
                state_dict,
                checkpoint_label=f"xmodel1 checkpoint {model_path}",
            )
            model.to(device)
            model.eval()
            inst = cls(
                model_version="xmodel1",
                model=model,
                encode_fn=features_mod.encode,
                device=device,
            )
            (
                inst._placement_rank_bonus,
                inst._placement_rank_bonus_norm,
                inst._placement_rank_score_scale,
            ) = cls._resolve_placement_runtime_cfg(cfg)
            inst._runtime_candidate_builder = lambda snap, actor, legal_actions=None: features_mod.build_runtime_candidate_arrays(
                snap,
                actor,
                legal_actions,
                max_candidates=XMODEL1_MAX_CANDIDATES,
                candidate_feature_dim=int(cfg.get("candidate_feature_dim", inferred_dims["candidate_feature_dim"])),
                candidate_flag_dim=int(cfg.get("candidate_flag_dim", inferred_dims["candidate_flag_dim"])),
            )
            inst._runtime_special_candidate_builder = (
                lambda snap, actor, legal_actions=None: features_mod.build_runtime_special_candidate_arrays(
                    snap,
                    actor,
                    legal_actions,
                    max_candidates=XMODEL1_MAX_SPECIAL_CANDIDATES,
                    feature_dim=XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
                )
            )
            return inst
        if inferred_version == "keqingv4":
            validate_keqingv4_checkpoint_metadata(
                ckpt,
                checkpoint_label=f"keqingv4 checkpoint {model_path}",
            )
            inferred_inputs = infer_keqingv4_input_dims(state_dict)
            features_mod = importlib.import_module("training.state_features")
            model_mod = importlib.import_module("keqingv4.model")
            preprocess_mod = importlib.import_module("keqingv4.preprocess_features")
            core_mod = importlib.import_module("keqing_core")
            model = model_mod.KeqingV4Model(
                hidden_dim=int(ckpt["hidden_dim"]),
                num_res_blocks=int(ckpt["num_res_blocks"]),
                c_tile=int(inferred_inputs["c_tile"]),
                n_scalar=int(inferred_inputs["n_scalar"]),
                action_embed_dim=int(ckpt["action_embed_dim"]),
                context_dim=int(ckpt["context_dim"]),
                summary_dim=int(ckpt["summary_dim"]),
                dropout=float(ckpt["dropout"]),
            )
            load_keqingv4_checkpoint_state(
                model,
                state_dict,
                checkpoint_label=f"keqingv4 checkpoint {model_path}",
            )
            model.to(device)
            model.eval()
            inst = cls(
                model_version="keqingv4",
                model=model,
                encode_fn=features_mod.encode,
                device=device,
            )
            (
                inst._placement_rank_bonus,
                inst._placement_rank_bonus_norm,
                inst._placement_rank_score_scale,
            ) = cls._resolve_placement_runtime_cfg(cfg)
            def _build_runtime_v4_summaries(snapshot: dict, actor: int, legal_actions: list[dict]):
                try:
                    return core_mod.build_keqingv4_typed_summaries(snapshot, actor, legal_actions)
                except RuntimeError as exc:
                    if not core_mod.is_missing_rust_capability_error(exc):
                        raise
                    # Keep the Python path as an emergency mirror only when the
                    # Rust typed-summary bridge is unavailable.
                    fallback = preprocess_mod.build_typed_action_summaries(snapshot, actor, legal_actions)
                    if not isinstance(fallback, tuple) or len(fallback) < 3:
                        raise RuntimeError("keqingv4 python summary fallback contract drifted")
                    return fallback[:3]
            inst._runtime_v4_summary_builder = _build_runtime_v4_summaries
            return inst

        raise ValueError(
            f"Unsupported checkpoint model_version={inferred_version!r}; "
            "only xmodel1 and keqingv4 are supported."
        )

    def encode(self, snap: dict, actor: int) -> tuple[np.ndarray, np.ndarray]:
        if self.model_version == "xmodel1":
            return self._encode(
                snap,
                actor,
                state_scalar_dim=getattr(
                    self.model,
                    "state_scalar_dim",
                    default_xmodel1_state_scalar_dim(),
                ),
            )
        return self._encode(snap, actor)

    @staticmethod
    def _resolve_runtime_legal_actions(snap: dict, actor: int) -> list[dict]:
        if "legal_actions" in snap and snap["legal_actions"] is not None:
            return [
                action.to_mjai() if hasattr(action, "to_mjai") else dict(action)
                for action in snap["legal_actions"]
            ]
        return [
            action.to_mjai() if hasattr(action, "to_mjai") else dict(action)
            for action in enumerate_legal_actions(snap, actor)
        ]

    @staticmethod
    def _build_v4_summary_cache_key(snap: dict, actor: int, legal_actions: list[dict]) -> str:
        payload = {
            "actor": actor,
            "bakaze": snap.get("bakaze"),
            "kyoku": snap.get("kyoku"),
            "honba": snap.get("honba"),
            "kyotaku": snap.get("kyotaku"),
            "oya": snap.get("oya"),
            "scores": snap.get("scores"),
            "hand": snap.get("hand"),
            "tsumo_pai": snap.get("tsumo_pai"),
            "melds": snap.get("melds"),
            "discards": snap.get("discards"),
            "dora_markers": snap.get("dora_markers"),
            "reached": snap.get("reached"),
            "last_tsumo": snap.get("last_tsumo"),
            "last_tsumo_raw": snap.get("last_tsumo_raw"),
            "legal_actions": legal_actions,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _validate_v4_summary_shape(
        name: str,
        value,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"keqingv4 {name} contract drift: expected shape {expected_shape}, got {arr.shape}"
            )
        return arr

    @staticmethod
    def _xmodel1_special_type_to_action_idx(special_type: int) -> int | None:
        mapping = {
            XMODEL1_SPECIAL_TYPE_REACH: REACH_IDX,
            XMODEL1_SPECIAL_TYPE_CHI_LOW: CHI_LOW_IDX,
            XMODEL1_SPECIAL_TYPE_CHI_MID: CHI_MID_IDX,
            XMODEL1_SPECIAL_TYPE_CHI_HIGH: CHI_HIGH_IDX,
            XMODEL1_SPECIAL_TYPE_PON: PON_IDX,
            XMODEL1_SPECIAL_TYPE_DAIMINKAN: DAIMINKAN_IDX,
            XMODEL1_SPECIAL_TYPE_ANKAN: ANKAN_IDX,
            XMODEL1_SPECIAL_TYPE_KAKAN: KAKAN_IDX,
            XMODEL1_SPECIAL_TYPE_HORA: HORA_IDX,
            XMODEL1_SPECIAL_TYPE_RYUKYOKU: RYUKYOKU_IDX,
            XMODEL1_SPECIAL_TYPE_NONE: NONE_IDX,
        }
        return mapping.get(int(special_type))

    @staticmethod
    def _xmodel1_special_candidate_bonus(features: np.ndarray) -> float:
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.size < 19:
            return 0.0
        risk_mean = float(np.mean(feat[13:16]))
        return (
            0.6 * float(feat[8])
            + 0.35 * float(feat[9])
            - 0.4 * float(feat[10])
            - 0.35 * risk_mean
            + 0.15 * float(feat[11])
            + 0.15 * float(feat[12])
            + 0.2 * float(feat[16])
            + 0.1 * float(feat[17])
            + 0.05 * float(feat[18])
        )

    @classmethod
    def _xmodel1_special_action_bonus_map(
        cls,
        special_feat: np.ndarray,
        special_type_id: np.ndarray,
        special_mask: np.ndarray,
        legal_actions: list[dict],
    ) -> dict[int, float]:
        legal_action_ids = {action_to_idx(action) for action in legal_actions}
        bonuses: dict[int, float] = {}
        for feat_row, type_value, mask_value in zip(special_feat, special_type_id, special_mask):
            if int(mask_value) <= 0:
                continue
            action_idx = cls._xmodel1_special_type_to_action_idx(int(type_value))
            if action_idx is None or action_idx not in legal_action_ids:
                continue
            bonus = cls._xmodel1_special_candidate_bonus(feat_row)
            previous = bonuses.get(action_idx)
            bonuses[action_idx] = bonus if previous is None else max(previous, bonus)
        return bonuses

    def _resolve_v4_runtime_summaries(
        self,
        snap: dict,
        actor: int,
        legal_actions: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            "v4_discard_summary" in snap
            and "v4_call_summary" in snap
            and "v4_special_summary" in snap
        ):
            return (
                self._validate_v4_summary_shape(
                    "discard summary",
                    snap["v4_discard_summary"],
                    (34, KEQINGV4_SUMMARY_DIM),
                ),
                self._validate_v4_summary_shape(
                    "call summary",
                    snap["v4_call_summary"],
                    (8, KEQINGV4_SUMMARY_DIM),
                ),
                self._validate_v4_summary_shape(
                    "special summary",
                    snap["v4_special_summary"],
                    (3, KEQINGV4_SUMMARY_DIM),
                ),
            )

        cache_key = self._build_v4_summary_cache_key(snap, actor, legal_actions)
        cached = self._runtime_v4_summary_cache.get(cache_key)
        if cached is not None:
            return cached

        summary_snap = dict(snap)
        # event_history is part of the keqingv4 model input contract, but the
        # typed-summary builders still operate on the plain replay snapshot.
        summary_snap.pop("event_history", None)
        discard_summary, call_summary, special_summary = self._runtime_v4_summary_builder(
            summary_snap,
            actor,
            legal_actions,
        )
        summaries = (
            self._validate_v4_summary_shape(
                "discard summary",
                discard_summary,
                (34, KEQINGV4_SUMMARY_DIM),
            ),
            self._validate_v4_summary_shape(
                "call summary",
                call_summary,
                (8, KEQINGV4_SUMMARY_DIM),
            ),
            self._validate_v4_summary_shape(
                "special summary",
                special_summary,
                (3, KEQINGV4_SUMMARY_DIM),
            ),
        )
        self._runtime_v4_summary_cache[cache_key] = summaries
        return summaries  # type: ignore[return-value]

    @staticmethod
    def _resolve_v4_event_history(snap: dict) -> np.ndarray:
        if "event_history" not in snap:
            raise RuntimeError("keqingv4 runtime contract drift: snapshot is missing event_history")
        arr = np.asarray(snap["event_history"], dtype=np.int16)
        expected = (KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM)
        if arr.shape != expected:
            raise RuntimeError(
                f"keqingv4 runtime event_history contract drift: expected shape {expected}, got {arr.shape}"
            )
        return arr

    def resolve_runtime_v4_summaries(
        self,
        snap: dict,
        actor: int,
        legal_actions: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_version != "keqingv4":
            raise RuntimeError("runtime v4 summaries are only available for keqingv4")
        resolved_legal = legal_actions if legal_actions is not None else self._resolve_runtime_legal_actions(snap, actor)
        return self._resolve_v4_runtime_summaries(snap, actor, resolved_legal)

    def forward(self, snap: dict, actor: int) -> ModelForwardResult:
        tile_feat, scalar = self.encode(snap, actor)
        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        xmodel1_payload = None
        with torch.no_grad():
            if self.model_version == "xmodel1":
                features_mod = importlib.import_module("xmodel1.features")
                legal_actions = self._resolve_runtime_legal_actions(snap, actor)
                candidate_feat, candidate_tile_id, candidate_mask, candidate_flags = self._runtime_candidate_builder(
                    snap,
                    actor,
                    legal_actions,
                )
                history_summary = features_mod.resolve_runtime_history_summary(snap)
                candidate_feat_t = torch.from_numpy(candidate_feat).unsqueeze(0).to(self.device)
                candidate_tile_id_t = torch.from_numpy(candidate_tile_id).unsqueeze(0).to(self.device)
                candidate_mask_t = torch.from_numpy(candidate_mask).unsqueeze(0).to(self.device)
                candidate_flags_t = torch.from_numpy(candidate_flags).unsqueeze(0).to(self.device)
                history_summary_t = torch.from_numpy(history_summary).unsqueeze(0).to(self.device).float()
                runtime_payload = features_mod.resolve_runtime_tensor_payload(
                    snap,
                    actor,
                    legal_actions,
                    max_candidates=XMODEL1_MAX_CANDIDATES,
                    candidate_feature_dim=int(getattr(self.model, "candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
                    candidate_flag_dim=int(getattr(self.model, "candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
                )
                response_action_idx_t = torch.from_numpy(runtime_payload["response_action_idx"]).unsqueeze(0).to(self.device).long()
                response_action_mask_t = torch.from_numpy(runtime_payload["response_action_mask"]).unsqueeze(0).to(self.device)
                response_post_candidate_feat_t = torch.from_numpy(runtime_payload["response_post_candidate_feat"]).unsqueeze(0).to(self.device)
                response_post_candidate_tile_id_t = torch.from_numpy(runtime_payload["response_post_candidate_tile_id"]).unsqueeze(0).to(self.device).long()
                response_post_candidate_mask_t = torch.from_numpy(runtime_payload["response_post_candidate_mask"]).unsqueeze(0).to(self.device)
                response_post_candidate_flags_t = torch.from_numpy(runtime_payload["response_post_candidate_flags"]).unsqueeze(0).to(self.device)
                out = self.model(
                    tile_t.float() if self.device.type != "cuda" else tile_t,
                    scalar_t.float() if self.device.type != "cuda" else scalar_t,
                    candidate_feat_t.float() if self.device.type != "cuda" else candidate_feat_t,
                    candidate_tile_id_t,
                    candidate_flags_t.float(),
                    candidate_mask_t.float(),
                    response_action_idx=response_action_idx_t,
                    response_action_mask=(
                        response_action_mask_t.float() if self.device.type != "cuda" else response_action_mask_t
                    ),
                    response_post_candidate_feat=(
                        response_post_candidate_feat_t.float() if self.device.type != "cuda" else response_post_candidate_feat_t
                    ),
                    response_post_candidate_tile_id=response_post_candidate_tile_id_t,
                    response_post_candidate_mask=(
                        response_post_candidate_mask_t.float() if self.device.type != "cuda" else response_post_candidate_mask_t
                    ),
                    response_post_candidate_flags=response_post_candidate_flags_t.float(),
                    history_summary=history_summary_t,
                )
                policy_logits_t = out.action_logits.clone()
                if self._runtime_special_candidate_builder is not None:
                    special_feat, special_type_id, special_mask = self._runtime_special_candidate_builder(
                        snap,
                        actor,
                        legal_actions,
                    )
                    for action_idx, bonus in self._xmodel1_special_action_bonus_map(
                        special_feat,
                        special_type_id,
                        special_mask,
                        legal_actions,
                    ).items():
                        policy_logits_t[:, action_idx] = (
                            policy_logits_t[:, action_idx]
                            + policy_logits_t.new_tensor(float(bonus))
                        )
                # Stage 1:删除 MC 形态的 global_value,runtime value 由分解 EV
                # 组合出;Stage 3 runtime 直通后会进一步替换为 per-candidate rerank。
                win_prob_t = torch.sigmoid(out.win_logit.float())
                dealin_prob_t = torch.sigmoid(out.dealin_logit.float())
                value_t = (
                    win_prob_t * out.pts_given_win.float()
                    - dealin_prob_t * out.pts_given_dealin.float()
                )
                xmodel1_payload = Xmodel1RuntimeOutputs(
                    discard_logits=out.discard_logits.squeeze(0).detach().cpu().numpy(),
                    candidate_tile_id=candidate_tile_id_t.squeeze(0).detach().cpu().numpy(),
                    candidate_mask=candidate_mask_t.squeeze(0).detach().cpu().numpy(),
                    response_logits=out.response_logits.squeeze(0).detach().cpu().numpy(),
                    response_action_idx=response_action_idx_t.squeeze(0).detach().cpu().numpy(),
                    response_action_mask=response_action_mask_t.squeeze(0).detach().cpu().numpy(),
                    response_post_candidate_feat=response_post_candidate_feat_t.squeeze(0).detach().cpu().numpy(),
                    response_post_candidate_mask=response_post_candidate_mask_t.squeeze(0).detach().cpu().numpy(),
                    response_teacher_discard_idx=runtime_payload.get(
                        "response_teacher_discard_idx",
                        np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16),
                    ),
                    win_prob=float(win_prob_t.squeeze().detach().cpu().item()),
                    dealin_prob=float(dealin_prob_t.squeeze().detach().cpu().item()),
                    pts_given_win=float(out.pts_given_win.squeeze().detach().cpu().item()),
                    pts_given_dealin=float(out.pts_given_dealin.squeeze().detach().cpu().item()),
                    opp_tenpai_probs=torch.sigmoid(out.opp_tenpai_logits.float()).squeeze(0).detach().cpu().numpy(),
                )
            elif self.model_version == "keqingv4":
                legal_actions = self._resolve_runtime_legal_actions(snap, actor)
                event_history = self._resolve_v4_event_history(snap)
                discard_summary, call_summary, special_summary = self._resolve_v4_runtime_summaries(
                    snap,
                    actor,
                    legal_actions,
                )
                event_history_t = torch.from_numpy(event_history).unsqueeze(0).to(self.device).long()
                discard_summary_t = torch.from_numpy(discard_summary).unsqueeze(0).to(self.device)
                call_summary_t = torch.from_numpy(call_summary).unsqueeze(0).to(self.device)
                special_summary_t = torch.from_numpy(special_summary).unsqueeze(0).to(self.device)
                policy_logits_t, value_t = self.model(
                    tile_t.float() if self.device.type != "cuda" else tile_t,
                    scalar_t.float() if self.device.type != "cuda" else scalar_t,
                    event_history=event_history_t,
                    discard_summary=discard_summary_t.float(),
                    call_summary=call_summary_t.float(),
                    special_summary=special_summary_t.float(),
                )
            else:
                policy_logits_t, value_t = self.model(tile_t, scalar_t)
        aux = self._get_aux_outputs()
        return ModelForwardResult(
            policy_logits=policy_logits_t.squeeze(0).detach().cpu().numpy(),
            value=float(value_t.squeeze().detach().cpu().item()),
            aux=aux,
            xmodel1=xmodel1_payload,
        )

    def forward_many(self, snaps: list[dict], actor: int) -> list[ModelForwardResult]:
        if not snaps:
            return []
        return [self.forward(snap, actor) for snap in snaps]

    def _get_aux_outputs(self) -> ModelAuxOutputs:
        if not hasattr(self.model, "get_last_aux_outputs"):
            return ModelAuxOutputs()
        try:
            aux = self.model.get_last_aux_outputs()
        except Exception:
            return ModelAuxOutputs()
        score_like = aux.get("score_delta")
        if score_like is None:
            score_like = aux.get("composed_ev")
        if score_like is None:
            score_like = aux.get("global_value")
        score_delta = 0.0 if score_like is None else float(score_like.squeeze().detach().cpu().item())
        win_prob = 0.0
        if "win_prob" in aux:
            win_prob = float(torch.sigmoid(aux["win_prob"].squeeze()).detach().cpu().item())
        dealin_prob = 0.0
        if "dealin_prob" in aux:
            dealin_prob = float(torch.sigmoid(aux["dealin_prob"].squeeze()).detach().cpu().item())
        rank_probs = (0.0, 0.0, 0.0, 0.0)
        if "rank_logits" in aux:
            rank_probs_t = torch.softmax(aux["rank_logits"].float(), dim=-1).reshape(-1, 4)[0]
            rank_probs = tuple(float(value) for value in rank_probs_t.detach().cpu().tolist())
        final_score_delta = 0.0
        if "final_score_delta" in aux:
            final_score_delta = float(aux["final_score_delta"].squeeze().detach().cpu().item())
        rank_pt_value = 0.0
        if "rank_logits" in aux:
            rank_pt_value = placement_utility_from_outputs(
                rank_probs,
                final_score_delta=final_score_delta,
                rank_bonus=self._placement_rank_bonus,
                rank_bonus_norm=self._placement_rank_bonus_norm,
                rank_score_scale=self._placement_rank_score_scale,
            )
        return ModelAuxOutputs(
            score_delta=score_delta,
            win_prob=win_prob,
            dealin_prob=dealin_prob,
            rank_probs=rank_probs,
            final_score_delta=final_score_delta,
            rank_pt_value=rank_pt_value,
        )
