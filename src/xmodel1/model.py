"""Xmodel1 discard-first model.

This implementation focuses on the first concrete milestone:
shared state encoder + discard candidate scorer + auxiliary heads.
It is generic over state/candidate dimensions so Rust-exported caches can be
wired in without changing the model code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from keqingv1.action_space import ACTION_SPACE


@dataclass
class Xmodel1Output:
    discard_logits: torch.Tensor
    action_logits: torch.Tensor
    global_value: torch.Tensor
    score_delta: torch.Tensor
    win_logit: torch.Tensor
    dealin_logit: torch.Tensor
    offense_quality: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Xmodel1Model(nn.Module):
    """Candidate-centric small model for Xmodel1 discard training."""

    def __init__(
        self,
        state_tile_channels: int,
        state_scalar_dim: int,
        candidate_feature_dim: int,
        candidate_flag_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_tile_channels = state_tile_channels
        self.state_scalar_dim = state_scalar_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.candidate_flag_dim = candidate_flag_dim
        self.hidden_dim = hidden_dim
        self.action_space = ACTION_SPACE

        self.state_proj = nn.Sequential(
            nn.Conv1d(state_tile_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-3),
            nn.Mish(),
        )
        self.state_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(state_scalar_dim, hidden_dim // 4),
            nn.Mish(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.tile_embed = nn.Embedding(35, 16, padding_idx=34)
        candidate_in_dim = candidate_feature_dim + candidate_flag_dim + 16
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_in_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Mish(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.misc_action_head = nn.Linear(hidden_dim, self.action_space - 34)

        self.global_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.score_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.win_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dealin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.offense_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()
        self._last_aux_outputs: dict[str, torch.Tensor] | None = None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_state(self, state_tile_feat: torch.Tensor, state_scalar: torch.Tensor) -> torch.Tensor:
        if state_tile_feat.dtype == torch.float16 and state_tile_feat.device.type != "cuda":
            state_tile_feat = state_tile_feat.float()
        if state_scalar.dtype == torch.float16 and state_scalar.device.type != "cuda":
            state_scalar = state_scalar.float()
        x = self.state_proj(state_tile_feat)
        x = self.state_blocks(x)
        x = x.mean(dim=-1)
        s = self.scalar_proj(state_scalar)
        return self.state_head(torch.cat([x, s], dim=-1))

    def forward(
        self,
        state_tile_feat: torch.Tensor,
        state_scalar: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_tile_id: torch.Tensor | None = None,
        candidate_flags: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> Xmodel1Output:
        # Legacy 5-arg path:
        #   model(state, scalar, candidate_feat, candidate_flags, candidate_mask)
        if candidate_mask is None and candidate_flags is not None and candidate_tile_id is not None:
            candidate_mask = candidate_flags
            candidate_flags = candidate_tile_id
            candidate_tile_id = None

        if candidate_flags is None or candidate_mask is None:
            raise TypeError("candidate_flags and candidate_mask are required")

        # Compatibility bridge:
        # - canonical order: (candidate_feat, candidate_tile_id, candidate_flags, candidate_mask)
        # - legacy/test order: (candidate_feat, candidate_flags, candidate_mask, candidate_tile_id)
        if (
            candidate_tile_id is not None
            and candidate_tile_id.ndim == 3
            and candidate_tile_id.shape[-1] == self.candidate_flag_dim
            and candidate_flags.ndim == 2
            and candidate_mask.ndim == 2
        ):
            candidate_tile_id, candidate_flags, candidate_mask = candidate_mask, candidate_tile_id, candidate_flags

        state_embed = self.encode_state(state_tile_feat, state_scalar)
        if candidate_tile_id is None:
            candidate_tile_id = torch.full(
                candidate_mask.shape,
                34,
                dtype=torch.long,
                device=candidate_mask.device,
            )
        tile_ids = candidate_tile_id.long().clamp(min=0, max=34)
        tile_embed = self.tile_embed(tile_ids)
        candidate_input = torch.cat([candidate_feat, candidate_flags.float(), tile_embed], dim=-1)
        candidate_embed = self.candidate_proj(candidate_input)
        state_expand = state_embed.unsqueeze(1).expand(
            candidate_embed.shape[0],
            candidate_embed.shape[1],
            state_embed.shape[-1],
        )
        interaction = state_expand[..., : candidate_embed.shape[-1]] * candidate_embed
        score_input = torch.cat([state_expand, candidate_embed, interaction], dim=-1)
        discard_logits = self.score_head(score_input).squeeze(-1)
        discard_logits = discard_logits.masked_fill(candidate_mask <= 0, -1e4)
        misc_logits = self.misc_action_head(state_embed)
        action_logits = self.to_action_logits(discard_logits, candidate_tile_id, candidate_mask, misc_logits)
        global_value = self.global_value_head(state_embed)
        score_delta = self.score_delta_head(state_embed)
        win_logit = self.win_head(state_embed)
        dealin_logit = self.dealin_head(state_embed)
        offense_quality = self.offense_quality_head(state_embed)
        self._last_aux_outputs = {
            "score_delta": score_delta,
            "win_prob": win_logit,
            "dealin_prob": dealin_logit,
            "offense_quality": offense_quality,
        }

        return Xmodel1Output(
            discard_logits=discard_logits,
            action_logits=action_logits,
            global_value=global_value,
            score_delta=score_delta,
            win_logit=win_logit,
            dealin_logit=dealin_logit,
            offense_quality=offense_quality,
        )

    def to_action_logits(
        self,
        discard_logits: torch.Tensor,
        candidate_tile_id: torch.Tensor,
        candidate_mask: torch.Tensor,
        misc_logits: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.full(
            (discard_logits.shape[0], self.action_space),
            fill_value=-1e4,
            dtype=discard_logits.dtype,
            device=discard_logits.device,
        )
        for b in range(discard_logits.shape[0]):
            for k in range(discard_logits.shape[1]):
                if candidate_mask[b, k] <= 0:
                    continue
                tile_id = int(candidate_tile_id[b, k].item())
                if 0 <= tile_id < 34:
                    out[b, tile_id] = torch.maximum(out[b, tile_id], discard_logits[b, k])
        out[:, 34:] = misc_logits
        return out

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError("Aux outputs are not available before a forward pass")
        return self._last_aux_outputs


__all__ = ["Xmodel1Model", "Xmodel1Output"]
