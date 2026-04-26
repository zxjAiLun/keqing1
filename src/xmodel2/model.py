"""xmodel2 experimental model: policy + decomposed EV + placement auxiliary v1."""

from __future__ import annotations

import torch
import torch.nn as nn

from mahjong_env.action_space import ACTION_SPACE
from training.state_features import C_TILE, N_SCALAR


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


class _Head(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        mid = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.Mish(),
            nn.Linear(mid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Xmodel2Model(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        c_tile: int = C_TILE,
        n_scalar: int = N_SCALAR,
        action_space: int = ACTION_SPACE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_space = action_space

        self.input_proj = nn.Sequential(
            nn.Conv1d(c_tile, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-3),
            nn.Mish(),
        )
        self.res_tower = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.scalar_proj = nn.Sequential(
            nn.Linear(n_scalar, hidden_dim // 4),
            nn.Mish(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.Mish(),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.offense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.defense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.placement_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.policy_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.policy_head = nn.Linear(hidden_dim, action_space)

        self.win_head = _Head(hidden_dim, 1)
        self.dealin_head = _Head(hidden_dim, 1)
        self.pts_given_win_head = _Head(hidden_dim, 1)
        self.pts_given_dealin_head = _Head(hidden_dim, 1)
        self.opp_tenpai_head = _Head(hidden_dim, 3)
        self.rank_logits_head = _Head(hidden_dim, 4)
        self.final_score_delta_head = _Head(hidden_dim, 1)

        self._last_aux_outputs: dict[str, torch.Tensor] | None = None
        self._last_policy_logits: torch.Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_state(
        self,
        tile_feat: torch.Tensor,
        scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tile_feat.dtype == torch.float16 and tile_feat.device.type != "cuda":
            tile_feat = tile_feat.float()
        if scalar.dtype == torch.float16 and scalar.device.type != "cuda":
            scalar = scalar.float()
        x = self.input_proj(tile_feat)
        x = self.res_tower(x)
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        scalar_latent = self.scalar_proj(scalar)
        shared = self.shared(torch.cat([avg_pool, max_pool, scalar_latent], dim=-1))
        return shared, {
            "offense": self.offense_stream(shared),
            "defense": self.defense_stream(shared),
            "placement": self.placement_stream(shared),
        }

    def forward(
        self,
        tile_feat: torch.Tensor,
        scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared, streams = self.encode_state(tile_feat, scalar)
        policy_repr = self.policy_proj(
            torch.cat([shared, streams["offense"], streams["defense"]], dim=-1)
        )
        policy_logits = self.policy_head(policy_repr)
        self._last_policy_logits = policy_logits

        win_logit = self.win_head(streams["offense"])
        dealin_logit = self.dealin_head(streams["defense"])
        pts_given_win = self.pts_given_win_head(streams["offense"])
        pts_given_dealin = self.pts_given_dealin_head(streams["defense"])
        opp_tenpai_logits = self.opp_tenpai_head(streams["defense"])
        rank_logits = self.rank_logits_head(streams["placement"])
        final_score_delta = self.final_score_delta_head(streams["placement"])

        win_prob = torch.sigmoid(win_logit.float())
        dealin_prob = torch.sigmoid(dealin_logit.float())
        opp_tenpai_risk = torch.sigmoid(opp_tenpai_logits.float()).mean(dim=-1, keepdim=True)
        composed_ev = win_prob * pts_given_win.float() - dealin_prob * pts_given_dealin.float() - 0.25 * opp_tenpai_risk
        composed_ev = torch.tanh(composed_ev).to(policy_logits.dtype)

        self._last_aux_outputs = {
            "global_value": composed_ev,
            "composed_ev": composed_ev,
            "win_logit": win_logit,
            "dealin_logit": dealin_logit,
            "pts_given_win": pts_given_win,
            "pts_given_dealin": pts_given_dealin,
            "opp_tenpai_logits": opp_tenpai_logits,
            "rank_logits": rank_logits,
            "final_score_delta": final_score_delta,
        }
        return policy_logits, composed_ev

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError("Aux outputs are not available before a forward pass")
        return self._last_aux_outputs

    def get_last_policy_logits(self) -> torch.Tensor:
        if self._last_policy_logits is None:
            raise RuntimeError("Policy logits are not available before a forward pass")
        return self._last_policy_logits


__all__ = ["Xmodel2Model"]
