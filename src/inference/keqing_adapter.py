from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from inference.contracts import ModelAuxOutputs, ModelForwardResult, Xmodel1RuntimeOutputs
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
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
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
                special_candidate_feature_dim=int(
                    cfg.get(
                        "special_candidate_feature_dim",
                        inferred_dims["special_candidate_feature_dim"],
                    )
                ),
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
            inst._runtime_candidate_builder = lambda snap, actor, legal_actions=None: features_mod.build_runtime_candidate_arrays(
                snap,
                actor,
                legal_actions,
                max_candidates=XMODEL1_MAX_CANDIDATES,
                candidate_feature_dim=int(cfg.get("candidate_feature_dim", inferred_dims["candidate_feature_dim"])),
                candidate_flag_dim=int(cfg.get("candidate_flag_dim", inferred_dims["candidate_flag_dim"])),
            )
            # Stage 3 E:runtime 补上 special candidate 通道,与 preprocess
            # 同源;此前 runtime 这块未 wire,forward 走 zero-default 导致
            # reach/pon/kan/none 的 flat logit 只由 misc_action_head 投影。
            inst._runtime_special_candidate_builder = lambda snap, actor, legal_actions=None: features_mod.build_runtime_special_candidate_arrays(
                snap,
                actor,
                legal_actions,
                max_candidates=int(cfg.get("max_special_candidates", XMODEL1_MAX_SPECIAL_CANDIDATES)),
                feature_dim=int(
                    cfg.get(
                        "special_candidate_feature_dim",
                        inferred_dims["special_candidate_feature_dim"],
                    )
                ),
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
                if self._runtime_special_candidate_builder is not None:
                    special_feat, special_type_id, special_mask = self._runtime_special_candidate_builder(
                        snap,
                        actor,
                        legal_actions,
                    )
                    special_feat_t = torch.from_numpy(special_feat).unsqueeze(0).to(self.device)
                    special_type_id_t = torch.from_numpy(special_type_id).unsqueeze(0).to(self.device).long()
                    special_mask_t = torch.from_numpy(special_mask).unsqueeze(0).to(self.device)
                else:
                    special_feat_t = None
                    special_type_id_t = None
                    special_mask_t = None
                out = self.model(
                    tile_t.float() if self.device.type != "cuda" else tile_t,
                    scalar_t.float() if self.device.type != "cuda" else scalar_t,
                    candidate_feat_t.float() if self.device.type != "cuda" else candidate_feat_t,
                    candidate_tile_id_t,
                    candidate_flags_t.float(),
                    candidate_mask_t.float(),
                    special_candidate_feat=(
                        special_feat_t.float() if special_feat_t is not None and self.device.type != "cuda" else special_feat_t
                    ),
                    special_candidate_type_id=special_type_id_t,
                    special_candidate_mask=(
                        special_mask_t.float() if special_mask_t is not None and self.device.type != "cuda" else special_mask_t
                    ),
                    history_summary=history_summary_t,
                )
                policy_logits_t = out.action_logits
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
                    special_logits=out.special_logits.squeeze(0).detach().cpu().numpy(),
                    special_type_id=(
                        special_type_id_t.squeeze(0).detach().cpu().numpy()
                        if special_type_id_t is not None
                        else np.full((XMODEL1_MAX_SPECIAL_CANDIDATES,), -1, dtype=np.int16)
                    ),
                    special_mask=(
                        special_mask_t.squeeze(0).detach().cpu().numpy()
                        if special_mask_t is not None
                        else np.zeros((XMODEL1_MAX_SPECIAL_CANDIDATES,), dtype=np.uint8)
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
        if score_like is None:
            return ModelAuxOutputs()
        return ModelAuxOutputs(
            score_delta=float(score_like.squeeze().detach().cpu().item()),
            win_prob=float(torch.sigmoid(aux["win_prob"].squeeze()).detach().cpu().item()),
            dealin_prob=float(torch.sigmoid(aux["dealin_prob"].squeeze()).detach().cpu().item()),
        )
