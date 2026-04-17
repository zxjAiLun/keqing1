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
from xmodel1.cached_dataset import (
    EVENT_HISTORY_LEN,
    EVENT_TYPE_PAD,
)
from xmodel1.schema import (
    XMODEL1_CHI_SPECIAL_TYPES,
    XMODEL1_KAN_SPECIAL_TYPES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
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


@dataclass
class Xmodel1Output:
    """xmodel1 前向输出。

    Stage 1 起:
    - 删除 MC 形态的 ``global_value`` / ``score_delta`` / ``offense_quality``,
      改走分解 EV: ``win_logit`` / ``dealin_logit`` / ``pts_given_win`` /
      ``pts_given_dealin`` 四头组合。
    - 新增 ``opp_tenpai_logits`` 三家听牌 head,Stage 1 不启用监督,
      先作为结构占位;preprocess 补齐标签后在 Stage 3 接入 loss。
    """

    discard_logits: torch.Tensor
    special_logits: torch.Tensor
    action_logits: torch.Tensor
    win_logit: torch.Tensor
    dealin_logit: torch.Tensor
    pts_given_win: torch.Tensor
    pts_given_dealin: torch.Tensor
    opp_tenpai_logits: torch.Tensor


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
        special_candidate_feature_dim: int = XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_tile_channels = state_tile_channels
        self.state_scalar_dim = state_scalar_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.candidate_flag_dim = candidate_flag_dim
        self.special_candidate_feature_dim = special_candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.action_space = ACTION_SPACE
        self.event_history_len = EVENT_HISTORY_LEN

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
        history_embed_dim = 128
        self.history_actor_embed = nn.Embedding(5, 8)
        self.history_event_type_embed = nn.Embedding(16, 8)
        self.history_tedashi_embed = nn.Embedding(2, 4)
        self.history_turn_embed = nn.Embedding(25, 8)
        self.history_proj = nn.Sequential(
            nn.Linear(8 + 8 + 16 + 8 + 4, history_embed_dim),
            nn.Mish(),
            nn.Linear(history_embed_dim, history_embed_dim),
            nn.Mish(),
        )
        history_encoder_layer = nn.TransformerEncoderLayer(
            d_model=history_embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(history_encoder_layer, num_layers=2)
        # Validation/runtime batches may contain fully padded history windows.
        # PyTorch's nested-tensor fast path can choke on all-padding sequences,
        # so keep the small encoder on the regular dense path.
        self.history_encoder.use_nested_tensor = False
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + history_embed_dim, hidden_dim),
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
        self.special_type_embed = nn.Embedding(13, 8, padding_idx=0)
        special_in_dim = special_candidate_feature_dim + 8
        self.special_proj = nn.Sequential(
            nn.Linear(special_in_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Mish(),
        )
        self.special_score_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        # 分解 EV 的两个条件回归头,label 由 trainer 从
        # (score_delta_target, win_target, dealin_target) 派生,使用 mask 屏蔽
        # 非对应样本。Stage 1 不改 preprocess。
        self.pts_given_win_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pts_given_dealin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # 三家对手听牌 head。Stage 1 不接入 loss,仅作为结构占位;
        # Stage 3 preprocess 补齐标签后启用 BCE 监督。
        self.opp_tenpai_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 3),
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

    def _encode_history(self, event_history: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if event_history is None:
            return torch.zeros((batch_size, self.history_proj[0].out_features), device=device)
        if event_history.dtype not in (torch.int16, torch.int32, torch.int64, torch.long):
            event_history = event_history.long()
        else:
            event_history = event_history.long()
        actor_ids = event_history[..., 0].clamp(min=0, max=4)
        event_type_ids = event_history[..., 1].clamp(min=0, max=15)
        tile_ids = event_history[..., 2].clamp(min=-1, max=33)
        tile_ids = torch.where(tile_ids < 0, torch.full_like(tile_ids, 34), tile_ids)
        turn_ids = event_history[..., 3].clamp(min=0, max=24)
        tedashi_ids = event_history[..., 4].clamp(min=0, max=1)
        history_input = torch.cat(
            [
                self.history_actor_embed(actor_ids),
                self.history_event_type_embed(event_type_ids),
                self.tile_embed(tile_ids),
                self.history_turn_embed(turn_ids),
                self.history_tedashi_embed(tedashi_ids),
            ],
            dim=-1,
        )
        history_embed = self.history_proj(history_input)
        pad_mask = event_type_ids == EVENT_TYPE_PAD
        encoded = self.history_encoder(history_embed, src_key_padding_mask=pad_mask)
        valid_mask = (~pad_mask).float()
        pooled = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = pooled / denom
        pooled = torch.where(
            (valid_mask.sum(dim=1, keepdim=True) > 0),
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled

    def encode_state(
        self,
        state_tile_feat: torch.Tensor,
        state_scalar: torch.Tensor,
        event_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_tile_feat.dtype == torch.float16 and state_tile_feat.device.type != "cuda":
            state_tile_feat = state_tile_feat.float()
        if state_scalar.dtype == torch.float16 and state_scalar.device.type != "cuda":
            state_scalar = state_scalar.float()
        x = self.state_proj(state_tile_feat)
        x = self.state_blocks(x)
        x = x.mean(dim=-1)
        s = self.scalar_proj(state_scalar)
        h = self._encode_history(event_history, batch_size=state_tile_feat.shape[0], device=state_tile_feat.device)
        return self.state_head(torch.cat([x, s, h], dim=-1))

    def forward(
        self,
        state_tile_feat: torch.Tensor,
        state_scalar: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_tile_id: torch.Tensor | None = None,
        candidate_flags: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        special_candidate_feat: torch.Tensor | None = None,
        special_candidate_type_id: torch.Tensor | None = None,
        special_candidate_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
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

        state_embed = self.encode_state(state_tile_feat, state_scalar, event_history=event_history)
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
        if (
            special_candidate_feat is None
            or special_candidate_type_id is None
            or special_candidate_mask is None
        ):
            batch_size = state_embed.shape[0]
            device = state_embed.device
            special_candidate_feat = torch.zeros(
                (batch_size, XMODEL1_MAX_SPECIAL_CANDIDATES, self.special_candidate_feature_dim),
                device=device,
                dtype=state_embed.dtype,
            )
            special_candidate_type_id = torch.full(
                (batch_size, XMODEL1_MAX_SPECIAL_CANDIDATES),
                -1,
                device=device,
                dtype=torch.long,
            )
            special_candidate_mask = torch.zeros((batch_size, XMODEL1_MAX_SPECIAL_CANDIDATES), device=device, dtype=torch.uint8)
        special_type_ids = special_candidate_type_id.long().clamp(min=-1, max=11) + 1
        special_type_embed = self.special_type_embed(special_type_ids)
        special_input = torch.cat([special_candidate_feat, special_type_embed], dim=-1)
        special_embed = self.special_proj(special_input)
        special_state_expand = state_embed.unsqueeze(1).expand(
            special_embed.shape[0],
            special_embed.shape[1],
            state_embed.shape[-1],
        )
        special_interaction = special_state_expand[..., : special_embed.shape[-1]] * special_embed
        special_score_input = torch.cat([special_state_expand, special_embed, special_interaction], dim=-1)
        special_logits = self.special_score_head(special_score_input).squeeze(-1)
        special_logits = special_logits.masked_fill(special_candidate_mask <= 0, -1e4)
        action_logits = self.to_action_logits(
            discard_logits,
            candidate_tile_id,
            candidate_mask,
            special_logits,
            special_candidate_type_id,
            special_candidate_mask,
        )
        win_logit = self.win_head(state_embed)
        dealin_logit = self.dealin_head(state_embed)
        pts_given_win = self.pts_given_win_head(state_embed)
        pts_given_dealin = self.pts_given_dealin_head(state_embed)
        opp_tenpai_logits = self.opp_tenpai_head(state_embed)
        composed_ev = torch.sigmoid(win_logit) * pts_given_win - torch.sigmoid(dealin_logit) * pts_given_dealin
        self._last_aux_outputs = {
            "score_delta": composed_ev,
            "composed_ev": composed_ev,
            "win_prob": win_logit,
            "dealin_prob": dealin_logit,
            "pts_given_win": pts_given_win,
            "pts_given_dealin": pts_given_dealin,
            "opp_tenpai_logits": opp_tenpai_logits,
        }

        return Xmodel1Output(
            discard_logits=discard_logits,
            special_logits=special_logits,
            action_logits=action_logits,
            win_logit=win_logit,
            dealin_logit=dealin_logit,
            pts_given_win=pts_given_win,
            pts_given_dealin=pts_given_dealin,
            opp_tenpai_logits=opp_tenpai_logits,
        )

    def to_action_logits(
        self,
        discard_logits: torch.Tensor,
        candidate_tile_id: torch.Tensor,
        candidate_mask: torch.Tensor,
        special_logits: torch.Tensor | None = None,
        special_candidate_type_id: torch.Tensor | None = None,
        special_candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = (candidate_mask > 0) & (candidate_tile_id >= 0) & (candidate_tile_id < 34)
        discard_action_logits = []
        for tile_id in range(34):
            tile_mask = valid_mask & (candidate_tile_id == tile_id)
            tile_scores = discard_logits.masked_fill(~tile_mask, -1e4)
            discard_action_logits.append(tile_scores.max(dim=1).values)
        non_discard_logits = discard_logits.new_full((discard_logits.shape[0], self.action_space - 34), -1e4)
        action_logits = torch.cat([torch.stack(discard_action_logits, dim=1), non_discard_logits], dim=1)
        if (
            special_logits is not None
            and special_candidate_type_id is not None
            and special_candidate_mask is not None
        ):
            valid_special = special_candidate_mask > 0
            for special_type, action_idx in (
                (XMODEL1_SPECIAL_TYPE_REACH, 34),
                (XMODEL1_SPECIAL_TYPE_CHI_LOW, 35),
                (XMODEL1_SPECIAL_TYPE_CHI_MID, 36),
                (XMODEL1_SPECIAL_TYPE_CHI_HIGH, 37),
                (XMODEL1_SPECIAL_TYPE_PON, 38),
                (XMODEL1_SPECIAL_TYPE_DAIMINKAN, 39),
                (XMODEL1_SPECIAL_TYPE_ANKAN, 40),
                (XMODEL1_SPECIAL_TYPE_KAKAN, 41),
                (XMODEL1_SPECIAL_TYPE_HORA, 42),
                (XMODEL1_SPECIAL_TYPE_RYUKYOKU, 43),
                (XMODEL1_SPECIAL_TYPE_NONE, 44),
            ):
                slot_mask = valid_special & (special_candidate_type_id == special_type)
                score = special_logits.masked_fill(~slot_mask, -1e4).max(dim=1).values
                action_logits[:, action_idx] = score
        return action_logits

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError("Aux outputs are not available before a forward pass")
        return self._last_aux_outputs


__all__ = ["Xmodel1Model", "Xmodel1Output"]
