from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from inference.contracts import ModelAuxOutputs, ModelForwardResult
from mahjong_env.legal_actions import enumerate_legal_actions
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
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
        if inferred_version == "xmodel1":
            features_mod = importlib.import_module("xmodel1.features")
            model_mod = importlib.import_module("xmodel1.model")
            model = model_mod.Xmodel1Model(
                state_tile_channels=int(cfg.get("state_tile_channels", 57)),
                state_scalar_dim=int(cfg.get("state_scalar_dim", 64)),
                candidate_feature_dim=int(cfg.get("candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
                candidate_flag_dim=int(cfg.get("candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
                hidden_dim=int(cfg.get("hidden_dim", hidden_dim)),
                num_res_blocks=int(cfg.get("num_res_blocks", num_res_blocks)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
            model.load_state_dict(state_dict, strict=False)
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
                candidate_feature_dim=int(cfg.get("candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
                candidate_flag_dim=int(cfg.get("candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
            )
            return inst

        c_tile = state_dict["input_proj.0.weight"].shape[1]
        n_scalar = state_dict["scalar_proj.0.weight"].shape[1]
        inferred_version = inferred_version or (
            "keqingv3" if (c_tile == 57 and n_scalar == 56) else "keqingv1"
        )
        if inferred_version == "keqingv3":
            features_mod = importlib.import_module("keqingv3.features")
            model_mod = importlib.import_module("keqingv3.model")
        else:
            features_mod = importlib.import_module("keqingv1.features")
            model_mod = importlib.import_module("keqingv1.model")

        model_cls = model_mod.MahjongModel
        model = model_cls(
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            c_tile=c_tile,
            n_scalar=n_scalar,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return cls(
            model_version=inferred_version,
            model=model,
            encode_fn=features_mod.encode,
            device=device,
        )

    def encode(self, snap: dict, actor: int) -> tuple[np.ndarray, np.ndarray]:
        if self.model_version == "xmodel1":
            return self._encode(snap, actor, state_scalar_dim=getattr(self.model, "state_scalar_dim", 64))
        return self._encode(snap, actor)

    def forward(self, snap: dict, actor: int) -> ModelForwardResult:
        tile_feat, scalar = self.encode(snap, actor)
        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_version == "xmodel1":
                legal_actions = [a.to_mjai() for a in enumerate_legal_actions(snap, actor)]
                candidate_feat, candidate_tile_id, candidate_mask, candidate_flags = self._runtime_candidate_builder(
                    snap,
                    actor,
                    legal_actions,
                )
                candidate_feat_t = torch.from_numpy(candidate_feat).unsqueeze(0).to(self.device)
                candidate_tile_id_t = torch.from_numpy(candidate_tile_id).unsqueeze(0).to(self.device)
                candidate_mask_t = torch.from_numpy(candidate_mask).unsqueeze(0).to(self.device)
                candidate_flags_t = torch.from_numpy(candidate_flags).unsqueeze(0).to(self.device)
                out = self.model(
                    tile_t.float() if self.device.type != "cuda" else tile_t,
                    scalar_t.float() if self.device.type != "cuda" else scalar_t,
                    candidate_feat_t.float() if self.device.type != "cuda" else candidate_feat_t,
                    candidate_tile_id_t,
                    candidate_flags_t.float(),
                    candidate_mask_t.float(),
                )
                policy_logits_t = out.action_logits
                value_t = out.global_value
            else:
                policy_logits_t, value_t = self.model(tile_t, scalar_t)
        aux = self._get_aux_outputs()
        return ModelForwardResult(
            policy_logits=policy_logits_t.squeeze(0).detach().cpu().numpy(),
            value=float(value_t.squeeze().detach().cpu().item()),
            aux=aux,
        )

    def _get_aux_outputs(self) -> ModelAuxOutputs:
        if not hasattr(self.model, "get_last_aux_outputs"):
            return ModelAuxOutputs()
        try:
            aux = self.model.get_last_aux_outputs()
        except Exception:
            return ModelAuxOutputs()
        return ModelAuxOutputs(
            score_delta=float(aux["score_delta"].squeeze().detach().cpu().item()),
            win_prob=float(torch.sigmoid(aux["win_prob"].squeeze()).detach().cpu().item()),
            dealin_prob=float(torch.sigmoid(aux["dealin_prob"].squeeze()).detach().cpu().item()),
        )
