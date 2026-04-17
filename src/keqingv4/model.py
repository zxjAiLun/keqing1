"""keqingv4 v2 architecture, phase 1.

This phase keeps the typed policy shell and current cache contract, while
aligning the model with the v2 design direction:

- remove the 5-stream + field-gate mixture
- keep only offense / defense streams
- remove additive summary bias injection
- predict decomposed EV components instead of a dedicated global-value head

The second returned tensor remains the weak MC-regularization target used by the
shared trainer, but it is now *composed* from decomposed heads rather than
coming from a standalone global-value branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from keqingv1.action_space import (
    ACTION_SPACE,
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
)
from keqingv3.features import C_TILE, N_SCALAR
from training.cache_schema import (
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)

_DISCARD_TILE_IDS = tuple(range(34))
_CALL_ACTION_IDS = (
    CHI_LOW_IDX,
    CHI_MID_IDX,
    CHI_HIGH_IDX,
    PON_IDX,
    DAIMINKAN_IDX,
    ANKAN_IDX,
    KAKAN_IDX,
    NONE_IDX,
)
_SPECIAL_ACTION_IDS = (REACH_IDX, HORA_IDX, RYUKYOKU_IDX)
_EVENT_TYPE_PAD = 0
_EVENT_NO_ACTOR = 4
_EVENT_NO_TILE = -1


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


class _TypedDecoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + embed_dim + embed_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_repr: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        state_expand = state_repr.unsqueeze(1).expand(-1, action_embed.shape[1], -1)
        interaction = state_expand[..., : action_embed.shape[-1]] * action_embed
        return self.net(torch.cat([state_expand, action_embed, interaction], dim=-1)).squeeze(-1)


class _SummaryPool(nn.Module):
    def __init__(self, summary_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.gate = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim != 3:
            raise ValueError(f"expected summary tensor [batch, slots, dim], got {tuple(summary.shape)}")
        latent = self.proj(summary)
        logits = self.gate(summary).squeeze(-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (latent * weights).sum(dim=1)


class KeqingV4Model(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 320,
        num_res_blocks: int = 6,
        c_tile: int = C_TILE,
        n_scalar: int = N_SCALAR,
        action_space: int = ACTION_SPACE,
        action_embed_dim: int = 64,
        context_dim: int = 32,
        summary_dim: int = KEQINGV4_SUMMARY_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.action_embed_dim = action_embed_dim
        self.context_dim = context_dim
        self.summary_dim = summary_dim
        self.event_history_len = KEQINGV4_EVENT_HISTORY_LEN

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
        history_dim = max(32, hidden_dim // 4)
        self.history_actor_embed = nn.Embedding(5, 8)
        self.history_event_type_embed = nn.Embedding(16, 8)
        self.history_tile_embed = nn.Embedding(35, 16, padding_idx=34)
        self.history_turn_embed = nn.Embedding(25, 8)
        self.history_tedashi_embed = nn.Embedding(2, 4)
        self.history_proj = nn.Sequential(
            nn.Linear(8 + 8 + 16 + 8 + 4, history_dim),
            nn.Mish(),
            nn.Linear(history_dim, history_dim),
            nn.Mish(),
        )
        history_encoder_layer = nn.TransformerEncoderLayer(
            d_model=history_dim,
            nhead=4,
            dim_feedforward=max(history_dim * 2, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(history_encoder_layer, num_layers=1)
        self.history_encoder.use_nested_tensor = False
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4 + history_dim, context_dim),
            nn.Mish(),
            nn.Linear(context_dim, context_dim),
            nn.Mish(),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4 + history_dim + context_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.offense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())
        self.defense_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())

        self.discard_summary_pool = _SummaryPool(summary_dim, hidden_dim)
        self.call_summary_pool = _SummaryPool(summary_dim, hidden_dim)
        self.special_summary_pool = _SummaryPool(summary_dim, hidden_dim)

        self.discard_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Mish(),
        )
        self.call_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Mish(),
        )
        self.special_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Mish(),
        )

        self.discard_embed = nn.Sequential(
            nn.Embedding(len(_DISCARD_TILE_IDS), action_embed_dim),
            nn.LayerNorm(action_embed_dim),
        )
        self.call_embed = nn.Sequential(
            nn.Embedding(len(_CALL_ACTION_IDS), action_embed_dim),
            nn.LayerNorm(action_embed_dim),
        )
        self.special_embed = nn.Sequential(
            nn.Embedding(len(_SPECIAL_ACTION_IDS), action_embed_dim),
            nn.LayerNorm(action_embed_dim),
        )

        self.discard_decoder = _TypedDecoder(hidden_dim, action_embed_dim, hidden_dim, dropout)
        self.call_decoder = _TypedDecoder(hidden_dim, action_embed_dim, hidden_dim, dropout)
        self.special_decoder = _TypedDecoder(hidden_dim, action_embed_dim, hidden_dim, dropout)

        self.win_head = _Head(hidden_dim, 1)
        self.pts_given_win_head = _Head(hidden_dim, 1)
        self.dealin_head = _Head(hidden_dim, 1)
        self.pts_given_dealin_head = _Head(hidden_dim, 1)
        self.opp_tenpai_head = _Head(hidden_dim, 3)

        self._discard_action_ids = torch.tensor(_DISCARD_TILE_IDS, dtype=torch.long)
        self._call_action_ids = torch.tensor(_CALL_ACTION_IDS, dtype=torch.long)
        self._special_action_ids = torch.tensor(_SPECIAL_ACTION_IDS, dtype=torch.long)
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode_history(self, event_history: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if event_history is None:
            return torch.zeros((batch_size, self.history_proj[0].out_features), device=device)
        event_history = event_history.long()
        actor_ids = event_history[..., 0].clamp(min=0, max=4)
        event_type_ids = event_history[..., 1].clamp(min=0, max=15)
        tile_ids = event_history[..., 2].clamp(min=_EVENT_NO_TILE, max=33)
        tile_ids = torch.where(tile_ids < 0, torch.full_like(tile_ids, 34), tile_ids)
        turn_ids = event_history[..., 3].clamp(min=0, max=24)
        tedashi_ids = event_history[..., 4].clamp(min=0, max=1)
        history_input = torch.cat(
            [
                self.history_actor_embed(actor_ids),
                self.history_event_type_embed(event_type_ids),
                self.history_tile_embed(tile_ids),
                self.history_turn_embed(turn_ids),
                self.history_tedashi_embed(tedashi_ids),
            ],
            dim=-1,
        )
        history_embed = self.history_proj(history_input)
        pad_mask = event_type_ids == _EVENT_TYPE_PAD
        encoded = self.history_encoder(history_embed, src_key_padding_mask=pad_mask)
        valid_mask = (~pad_mask).float()
        pooled = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = pooled / denom
        pooled = torch.where(valid_mask.sum(dim=1, keepdim=True) > 0, pooled, torch.zeros_like(pooled))
        return pooled

    def encode_state(
        self,
        tile_feat: torch.Tensor,
        scalar: torch.Tensor,
        event_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_proj(tile_feat)
        x = self.res_tower(x)
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        scalar_latent = self.scalar_proj(scalar)
        history_latent = self._encode_history(event_history, batch_size=tile_feat.shape[0], device=tile_feat.device)
        pooled = torch.cat([avg_pool, max_pool, scalar_latent, history_latent], dim=-1)
        context = self.context_proj(pooled)
        shared = self.shared(torch.cat([pooled, context], dim=-1))
        return shared, {
            "offense": self.offense_stream(shared),
            "defense": self.defense_stream(shared),
        }

    def _typed_state_repr(
        self,
        shared: torch.Tensor,
        fields: dict[str, torch.Tensor],
        discard_summary: torch.Tensor | None = None,
        call_summary: torch.Tensor | None = None,
        special_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(shared)
        discard_summary_latent = self.discard_summary_pool(discard_summary) if discard_summary is not None else zeros
        call_summary_latent = self.call_summary_pool(call_summary) if call_summary is not None else zeros
        special_summary_latent = self.special_summary_pool(special_summary) if special_summary is not None else zeros

        discard_repr = self.discard_state_proj(
            torch.cat([shared, fields["offense"], discard_summary_latent], dim=-1)
        )
        call_repr = self.call_state_proj(
            torch.cat([shared, fields["offense"], fields["defense"], call_summary_latent], dim=-1)
        )
        special_repr = self.special_state_proj(
            torch.cat([shared, fields["offense"], fields["defense"], special_summary_latent], dim=-1)
        )
        return discard_repr, call_repr, special_repr

    def forward(
        self,
        tile_feat: torch.Tensor,
        scalar: torch.Tensor,
        event_history: torch.Tensor | None = None,
        discard_summary: torch.Tensor | None = None,
        call_summary: torch.Tensor | None = None,
        special_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared, fields = self.encode_state(tile_feat, scalar, event_history=event_history)
        discard_repr, call_repr, special_repr = self._typed_state_repr(
            shared,
            fields,
            discard_summary=discard_summary,
            call_summary=call_summary,
            special_summary=special_summary,
        )

        batch_size = tile_feat.shape[0]
        device = tile_feat.device
        logits = torch.zeros((batch_size, self.action_space), device=device, dtype=shared.dtype)

        discard_ids = self._discard_action_ids.to(device)
        discard_embed = self.discard_embed(discard_ids).unsqueeze(0).expand(batch_size, -1, -1)
        logits[:, discard_ids] = self.discard_decoder(discard_repr, discard_embed).to(logits.dtype)

        call_ids = self._call_action_ids.to(device)
        call_embed = self.call_embed(torch.arange(len(_CALL_ACTION_IDS), device=device)).unsqueeze(0).expand(batch_size, -1, -1)
        logits[:, call_ids] = self.call_decoder(call_repr, call_embed).to(logits.dtype)

        special_ids = self._special_action_ids.to(device)
        special_embed = self.special_embed(torch.arange(len(_SPECIAL_ACTION_IDS), device=device)).unsqueeze(0).expand(batch_size, -1, -1)
        logits[:, special_ids] = self.special_decoder(special_repr, special_embed).to(logits.dtype)
        self._last_policy_logits = logits

        win_logit = self.win_head(fields["offense"])
        pts_given_win = self.pts_given_win_head(fields["offense"])
        dealin_logit = self.dealin_head(fields["defense"])
        pts_given_dealin = self.pts_given_dealin_head(fields["defense"])
        opp_tenpai_logits = self.opp_tenpai_head(fields["defense"])

        win_prob = torch.sigmoid(win_logit.float())
        dealin_prob = torch.sigmoid(dealin_logit.float())
        opp_tenpai_risk = torch.sigmoid(opp_tenpai_logits.float()).mean(dim=-1, keepdim=True)
        composed_ev = win_prob * pts_given_win.float() - dealin_prob * pts_given_dealin.float() - 0.25 * opp_tenpai_risk
        composed_ev = torch.tanh(composed_ev).to(shared.dtype)

        self._last_aux_outputs = {
            "global_value": composed_ev,
            "composed_ev": composed_ev,
            "win_prob": win_logit,
            "dealin_prob": dealin_logit,
            "pts_given_win": pts_given_win,
            "pts_given_dealin": pts_given_dealin,
            "opp_tenpai_logits": opp_tenpai_logits,
        }
        return logits, composed_ev

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError("Aux outputs are not available before a forward pass")
        return self._last_aux_outputs

    def get_last_policy_logits(self) -> torch.Tensor:
        if self._last_policy_logits is None:
            raise RuntimeError("Policy logits are not available before a forward pass")
        return self._last_policy_logits


__all__ = ["KeqingV4Model"]
