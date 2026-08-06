# -*- coding: utf-8 -*-
"""participants stats（R10-C skeleton）。

本轮只定义数据完整度层级与聚合入口；真实聚合在 R10-C 基于统一账本实现。
"""
from __future__ import annotations

from typing import Any


def compute_account_stats(account_id: str, registry, ledger) -> dict[str, Any]:
    """R10-C 占位：返回空结构，标明 implement 状态与可计算场次数。"""
    matches = ledger.list_matches(account_id=account_id).matches
    active = [m for m in matches if m.status == "active"]
    by_level = {"result_only": 0, "hand_summary": 0, "full_replay": 0}
    for m in active:
        by_level[m.data_completeness] = by_level.get(m.data_completeness, 0) + 1
    return {
        "implemented": False,
        "account_id": account_id,
        "total_active_matches": len(active),
        "data_completeness": by_level,
    }
