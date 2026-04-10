"""keqingv31 model.

Architecture goals:
- stronger action comparison than a flat policy head
- explicit offense / defense / global latent streams
- keep a unified Mahjong action-space interface
"""

from __future__ import annotations

import torch
import torch.nn as nn

from keqingv1.action_space import ACTION_SPACE
from keqingv3.features import C_TILE, N_SCALAR


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
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


class _ValueHead(nn.Module):
    def __init__(self, hidden_dim: int, out_act: nn.Module | None = None):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        ]
        if out_act is not None:
            layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KeqingV31Model(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 320,
        num_res_blocks: int = 6,
        c_tile: int = C_TILE,
        n_scalar: int = N_SCALAR,
        action_space: int = ACTION_SPACE,
        action_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.action_embed_dim = action_embed_dim

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

        self.global_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.offense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.defense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())

        self.stream_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.action_embed = nn.Embedding(action_space, action_embed_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(action_embed_dim, hidden_dim // 2),
            nn.Mish(),
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.global_value_head = _ValueHead(hidden_dim, nn.Tanh())
        self.offense_value_head = _ValueHead(hidden_dim, nn.Tanh())
        self.defense_risk_head = _ValueHead(hidden_dim, None)
        self.score_delta_head = _ValueHead(hidden_dim, nn.Tanh())
        self.win_head = _ValueHead(hidden_dim, None)
        self.dealin_head = _ValueHead(hidden_dim, None)

        self._last_aux_outputs: dict[str, torch.Tensor] | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def encode_state(self, tile_feat: torch.Tensor, scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(tile_feat)
        x = self.res_tower(x)
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        s = self.scalar_proj(scalar)
        shared = self.shared(torch.cat([avg_pool, max_pool, s], dim=-1))
        g = self.global_stream(shared)
        o = self.offense_stream(shared)
        d = self.defense_stream(shared)
        return shared, g, o, d

    def forward(self, tile_feat: torch.Tensor, scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared, global_latent, offense_latent, defense_latent = self.encode_state(tile_feat, scalar)
        gate_logits = self.stream_gate(shared)
        gate = torch.softmax(gate_logits, dim=-1)
        mixed = (
            gate[:, 0:1] * global_latent
            + gate[:, 1:2] * offense_latent
            + gate[:, 2:3] * defense_latent
        )

        action_ids = torch.arange(self.action_space, device=tile_feat.device)
        action_emb = self.action_proj(self.action_embed(action_ids))
        action_emb = action_emb.unsqueeze(0).expand(tile_feat.shape[0], -1, -1)
        state_expand = mixed.unsqueeze(1).expand(-1, self.action_space, -1)
        interaction = state_expand * torch.cat(
            [action_emb, action_emb], dim=-1
        )[:, :, : self.hidden_dim]
        policy_logits = self.action_scorer(torch.cat([state_expand, action_emb, interaction], dim=-1)).squeeze(-1)

        global_value = self.global_value_head(global_latent)
        self._last_aux_outputs = {
            "global_value": global_value,
            "offense_value": self.offense_value_head(offense_latent),
            "defense_risk": self.defense_risk_head(defense_latent),
            "score_delta": self.score_delta_head(global_latent),
            "win_prob": self.win_head(offense_latent),
            "dealin_prob": self.dealin_head(defense_latent),
            "stream_gate": gate,
        }
        return policy_logits, global_value

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError('Aux outputs are not available before a forward pass')
        return self._last_aux_outputs
