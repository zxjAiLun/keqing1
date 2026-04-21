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

from mahjong_env.action_space import ACTION_SPACE
from training.cache_schema import XMODEL1_HISTORY_SUMMARY_DIM
from xmodel1.schema import XMODEL1_MAX_RESPONSE_CANDIDATES


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
    response_logits: torch.Tensor
    response_post_logits: torch.Tensor
    action_logits: torch.Tensor
    win_logit: torch.Tensor
    dealin_logit: torch.Tensor
    pts_given_win: torch.Tensor
    pts_given_dealin: torch.Tensor
    opp_tenpai_logits: torch.Tensor
    rank_logits: torch.Tensor
    final_score_delta: torch.Tensor


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
        history_hidden_dim = 64
        self.history_summary_proj = nn.Sequential(
            nn.Linear(XMODEL1_HISTORY_SUMMARY_DIM, history_hidden_dim),
            nn.Mish(),
            nn.Linear(history_hidden_dim, history_hidden_dim),
            nn.Mish(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + history_hidden_dim, hidden_dim),
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
        self.response_action_embed = nn.Embedding(46, 16, padding_idx=0)
        response_in_dim = 16 + hidden_dim // 2 + 4
        self.response_proj = nn.Sequential(
            nn.Linear(response_in_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Mish(),
        )
        self.response_score_head = nn.Sequential(
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
        self.placement_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.rank_logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.final_score_delta_head = nn.Sequential(
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

    def _encode_history_summary(
        self,
        history_summary: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if history_summary is None:
            return torch.zeros((batch_size, self.history_summary_proj[0].out_features), device=device)
        if history_summary.dtype == torch.float16 and device.type != "cuda":
            history_summary = history_summary.float()
        else:
            history_summary = history_summary.float()
        return self.history_summary_proj(history_summary)

    def encode_state(
        self,
        state_tile_feat: torch.Tensor,
        state_scalar: torch.Tensor,
        history_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_tile_feat.dtype == torch.float16 and state_tile_feat.device.type != "cuda":
            state_tile_feat = state_tile_feat.float()
        if state_scalar.dtype == torch.float16 and state_scalar.device.type != "cuda":
            state_scalar = state_scalar.float()
        x = self.state_proj(state_tile_feat)
        x = self.state_blocks(x)
        x = x.mean(dim=-1)
        s = self.scalar_proj(state_scalar)
        h = self._encode_history_summary(
            history_summary,
            batch_size=state_tile_feat.shape[0],
            device=state_tile_feat.device,
        )
        return self.state_head(torch.cat([x, s, h], dim=-1))

    def forward(
        self,
        state_tile_feat: torch.Tensor,
        state_scalar: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_tile_id: torch.Tensor | None = None,
        candidate_flags: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        response_action_idx: torch.Tensor | None = None,
        response_action_mask: torch.Tensor | None = None,
        response_post_candidate_feat: torch.Tensor | None = None,
        response_post_candidate_tile_id: torch.Tensor | None = None,
        response_post_candidate_mask: torch.Tensor | None = None,
        response_post_candidate_flags: torch.Tensor | None = None,
        history_summary: torch.Tensor | None = None,
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

        state_embed = self.encode_state(
            state_tile_feat,
            state_scalar,
            history_summary=history_summary,
        )
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
        batch_size = state_embed.shape[0]
        device = state_embed.device
        if response_action_idx is None or response_action_mask is None:
            response_action_idx = torch.full(
                (batch_size, XMODEL1_MAX_RESPONSE_CANDIDATES),
                -1,
                device=device,
                dtype=torch.long,
            )
            response_action_mask = torch.zeros(
                (batch_size, XMODEL1_MAX_RESPONSE_CANDIDATES),
                device=device,
                dtype=torch.uint8,
            )
        if response_post_candidate_feat is None:
            response_post_candidate_feat = torch.zeros(
                (
                    batch_size,
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    candidate_feat.shape[1],
                    self.candidate_feature_dim,
                ),
                device=device,
                dtype=state_embed.dtype,
            )
        if response_post_candidate_tile_id is None:
            response_post_candidate_tile_id = torch.full(
                (
                    batch_size,
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    candidate_feat.shape[1],
                ),
                34,
                device=device,
                dtype=torch.long,
            )
        if response_post_candidate_mask is None:
            response_post_candidate_mask = torch.zeros(
                (
                    batch_size,
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    candidate_feat.shape[1],
                ),
                device=device,
                dtype=torch.uint8,
            )
        if response_post_candidate_flags is None:
            response_post_candidate_flags = torch.zeros(
                (
                    batch_size,
                    XMODEL1_MAX_RESPONSE_CANDIDATES,
                    candidate_feat.shape[1],
                    self.candidate_flag_dim,
                ),
                device=device,
                dtype=torch.uint8,
            )
        if response_post_candidate_feat.dtype == torch.float16 and response_post_candidate_feat.device.type != "cuda":
            response_post_candidate_feat = response_post_candidate_feat.float()
        response_action_ids = response_action_idx.long().clamp(min=-1, max=44) + 1
        response_action_embed = self.response_action_embed(response_action_ids)
        response_post_tile_embed = self.tile_embed(
            response_post_candidate_tile_id.long().clamp(min=0, max=34)
        )
        response_post_input = torch.cat(
            [
                response_post_candidate_feat,
                response_post_candidate_flags.float(),
                response_post_tile_embed,
            ],
            dim=-1,
        )
        flat_post_input = response_post_input.reshape(
            batch_size * XMODEL1_MAX_RESPONSE_CANDIDATES,
            response_post_input.shape[2],
            response_post_input.shape[3],
        )
        flat_post_embed = self.candidate_proj(flat_post_input).reshape(
            batch_size,
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            response_post_input.shape[2],
            -1,
        )
        flat_state_expand = state_embed.unsqueeze(1).unsqueeze(2).expand(
            batch_size,
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            response_post_input.shape[2],
            state_embed.shape[-1],
        )
        flat_post_interaction = flat_state_expand[..., : flat_post_embed.shape[-1]] * flat_post_embed
        flat_post_score_input = torch.cat(
            [flat_state_expand, flat_post_embed, flat_post_interaction],
            dim=-1,
        ).reshape(
            batch_size * XMODEL1_MAX_RESPONSE_CANDIDATES,
            response_post_input.shape[2],
            state_embed.shape[-1] + flat_post_embed.shape[-1] * 2,
        )
        response_post_logits = self.score_head(flat_post_score_input).squeeze(-1).reshape(
            batch_size,
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            response_post_input.shape[2],
        )
        response_post_logits = response_post_logits.masked_fill(
            response_post_candidate_mask <= 0,
            -1e4,
        )
        post_valid = response_post_candidate_mask > 0
        post_valid_f = post_valid.float()
        post_count = post_valid_f.sum(dim=-1, keepdim=True)
        pooled_post_embed = (
            flat_post_embed * post_valid_f.unsqueeze(-1)
        ).sum(dim=-2) / post_count.clamp_min(1.0)
        pooled_post_embed = torch.where(
            post_count > 0,
            pooled_post_embed,
            torch.zeros_like(pooled_post_embed),
        )
        safe_post_logits = response_post_logits.masked_fill(~post_valid, -1e4)
        best_post_logit = safe_post_logits.max(dim=-1).values
        best_post_logit = torch.where(
            post_valid.any(dim=-1),
            best_post_logit,
            torch.zeros_like(best_post_logit),
        )
        mean_post_logit = (
            response_post_logits.masked_fill(~post_valid, 0.0).sum(dim=-1)
            / post_count.squeeze(-1).clamp_min(1.0)
        )
        mean_post_logit = torch.where(
            post_valid.any(dim=-1),
            mean_post_logit,
            torch.zeros_like(mean_post_logit),
        )
        post_count_norm = post_count.squeeze(-1) / float(response_post_input.shape[2])
        has_post = post_valid.any(dim=-1).float()
        response_input = torch.cat(
            [
                response_action_embed,
                pooled_post_embed,
                best_post_logit.unsqueeze(-1),
                mean_post_logit.unsqueeze(-1),
                post_count_norm.unsqueeze(-1),
                has_post.unsqueeze(-1),
            ],
            dim=-1,
        )
        response_embed = self.response_proj(response_input)
        response_state_expand = state_embed.unsqueeze(1).expand(
            response_embed.shape[0],
            response_embed.shape[1],
            state_embed.shape[-1],
        )
        response_interaction = response_state_expand[..., : response_embed.shape[-1]] * response_embed
        response_score_input = torch.cat(
            [response_state_expand, response_embed, response_interaction],
            dim=-1,
        )
        response_logits = self.response_score_head(response_score_input).squeeze(-1)
        response_logits = response_logits.masked_fill(response_action_mask <= 0, -1e4)
        action_logits = self.to_action_logits(
            discard_logits,
            candidate_tile_id,
            candidate_mask,
            response_logits,
            response_action_idx,
            response_action_mask,
        )
        win_logit = self.win_head(state_embed)
        dealin_logit = self.dealin_head(state_embed)
        pts_given_win = self.pts_given_win_head(state_embed)
        pts_given_dealin = self.pts_given_dealin_head(state_embed)
        opp_tenpai_logits = self.opp_tenpai_head(state_embed)
        placement_embed = self.placement_stream(state_embed)
        rank_logits = self.rank_logits_head(placement_embed)
        final_score_delta = self.final_score_delta_head(placement_embed)
        composed_ev = torch.sigmoid(win_logit) * pts_given_win - torch.sigmoid(dealin_logit) * pts_given_dealin
        self._last_aux_outputs = {
            "score_delta": composed_ev,
            "composed_ev": composed_ev,
            "win_prob": win_logit,
            "dealin_prob": dealin_logit,
            "pts_given_win": pts_given_win,
            "pts_given_dealin": pts_given_dealin,
            "opp_tenpai_logits": opp_tenpai_logits,
            "rank_logits": rank_logits,
            "final_score_delta": final_score_delta,
        }

        return Xmodel1Output(
            discard_logits=discard_logits,
            response_logits=response_logits,
            response_post_logits=response_post_logits,
            action_logits=action_logits,
            win_logit=win_logit,
            dealin_logit=dealin_logit,
            pts_given_win=pts_given_win,
            pts_given_dealin=pts_given_dealin,
            opp_tenpai_logits=opp_tenpai_logits,
            rank_logits=rank_logits,
            final_score_delta=final_score_delta,
        )

    def to_action_logits(
        self,
        discard_logits: torch.Tensor,
        candidate_tile_id: torch.Tensor,
        candidate_mask: torch.Tensor,
        response_logits: torch.Tensor | None = None,
        response_action_idx: torch.Tensor | None = None,
        response_action_mask: torch.Tensor | None = None,
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
            response_logits is not None
            and response_action_idx is not None
            and response_action_mask is not None
        ):
            valid_response = response_action_mask > 0
            for action_idx in range(34, self.action_space):
                slot_mask = valid_response & (response_action_idx == action_idx)
                score = response_logits.masked_fill(~slot_mask, -1e4).max(dim=1).values
                action_logits[:, action_idx] = score
        return action_logits

    def get_last_aux_outputs(self) -> dict[str, torch.Tensor]:
        if self._last_aux_outputs is None:
            raise RuntimeError("Aux outputs are not available before a forward pass")
        return self._last_aux_outputs


__all__ = ["Xmodel1Model", "Xmodel1Output"]
