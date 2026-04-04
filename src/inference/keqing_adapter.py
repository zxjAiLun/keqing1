from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from inference.contracts import ModelAuxOutputs, ModelForwardResult


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
        c_tile = state_dict["input_proj.0.weight"].shape[1]
        n_scalar = state_dict["scalar_proj.0.weight"].shape[1]
        inferred_version = model_version or (
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
        return self._encode(snap, actor)

    def forward(self, snap: dict, actor: int) -> ModelForwardResult:
        tile_feat, scalar = self.encode(snap, actor)
        tile_t = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        with torch.no_grad():
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
