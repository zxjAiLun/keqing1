from __future__ import annotations

BASE_CACHE_FIELDS = ("tile_feat", "scalar", "mask", "action_idx", "value")
MELD_RANK_EXTRA_FIELDS = ("snap_json",)
V3_AUX_EXTRA_FIELDS = ("score_delta_target", "win_target", "dealin_target")

__all__ = ["BASE_CACHE_FIELDS", "MELD_RANK_EXTRA_FIELDS", "V3_AUX_EXTRA_FIELDS"]
