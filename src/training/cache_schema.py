from __future__ import annotations

BASE_CACHE_FIELDS = ("tile_feat", "scalar", "mask", "action_idx", "value")
MELD_RANK_EXTRA_FIELDS = ("snap_json",)
V3_AUX_EXTRA_FIELDS = ("score_delta_target", "win_target", "dealin_target")

# Xmodel1 discard cache schema (candidate-centric)
XMODEL1_SCHEMA_NAME = "xmodel1_discard_v1"
XMODEL1_SCHEMA_VERSION = 1
XMODEL1_MAX_CANDIDATES = 14
XMODEL1_CANDIDATE_FEATURE_DIM = 21
XMODEL1_CANDIDATE_FLAG_DIM = 10

XMODEL1_BASE_FIELDS = (
    "state_tile_feat",
    "state_scalar",
    "candidate_feat",
    "candidate_tile_id",
    "candidate_mask",
    "candidate_flags",
    "chosen_candidate_idx",
)

XMODEL1_TEACHER_FIELDS = (
    "candidate_quality_score",
    "candidate_rank_bucket",
    "candidate_hard_bad_flag",
)

XMODEL1_AUX_TARGET_FIELDS = (
    "global_value_target",
    "score_delta_target",
    "win_target",
    "dealin_target",
    "offense_quality_target",
)

XMODEL1_METADATA_FIELDS = (
    "sample_type",
    "actor",
    "event_index",
    "kyoku",
    "honba",
    "is_open_hand",
)

__all__ = [
    "BASE_CACHE_FIELDS",
    "MELD_RANK_EXTRA_FIELDS",
    "V3_AUX_EXTRA_FIELDS",
    "XMODEL1_SCHEMA_NAME",
    "XMODEL1_SCHEMA_VERSION",
    "XMODEL1_MAX_CANDIDATES",
    "XMODEL1_CANDIDATE_FEATURE_DIM",
    "XMODEL1_CANDIDATE_FLAG_DIM",
    "XMODEL1_BASE_FIELDS",
    "XMODEL1_TEACHER_FIELDS",
    "XMODEL1_AUX_TARGET_FIELDS",
    "XMODEL1_METADATA_FIELDS",
]
