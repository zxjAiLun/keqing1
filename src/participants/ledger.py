# -*- coding: utf-8 -*-
"""participants ledger：统一对局账本 + 手动录入 + revision 审计。

- ``matches.jsonl``：每行一局（当前态），修改时整行原子重写。
- ``match_revisions.jsonl``：追加式审计日志，每次 create/revise/void 记全量 before/after。
- 排名恒由 ``mahjong_env.final_rank.final_ranks`` 服务端计算并落盘。
- 校验失败默认拒绝（422）；``force + reason`` 允许强制保存，原因与校验结果入 revision。
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

from mahjong_env.final_rank import final_ranks

from .paths import data_root, now_iso, atomic_write_text, data_lock
from .schemas import (
    MATCH_REVISION_SCHEMA,
    Match,
    MatchCreate,
    MatchListResponse,
    MatchRevise,
    MatchSeat,
    MatchVoid,
    RevisionSummary,
    ValidationIssue,
)

_write_lock = threading.RLock()


def _matches_path() -> Path:
    return data_root() / "matches.jsonl"


def _revisions_path() -> Path:
    return data_root() / "match_revisions.jsonl"


def _generate_match_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return f"m_{stamp}_{uuid.uuid4().hex[:6]}"


def _generate_revision_id(match_id: str, revision: int) -> str:
    return f"rev_{match_id}_{revision}"


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def _read_match_rows() -> list[dict]:
    path = _matches_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_revision_rows() -> list[dict]:
    path = _revisions_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_matches(
    *,
    source: str | None = None,
    status: str | None = None,
    account_id: str | None = None,
    from_at: str | None = None,
    to_at: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> MatchListResponse:
    rows = _read_match_rows()
    matches: list[Match] = []
    for raw in rows:
        match = Match.model_validate(raw)
        if source and match.source != source:
            continue
        if status and match.status != status:
            continue
        if account_id and not any(seat.account_id == account_id for seat in match.seats):
            continue
        if from_at and match.occurred_at < from_at:
            continue
        if to_at and match.occurred_at > to_at:
            continue
        matches.append(match)
    matches.sort(key=lambda m: (m.occurred_at, m.match_id), reverse=True)
    total = len(matches)
    if offset:
        matches = matches[offset:]
    if limit is not None:
        matches = matches[:limit]
    return MatchListResponse(matches=matches, total=total)


def get_match(match_id: str) -> Match | None:
    for raw in _read_match_rows():
        if raw.get("match_id") == match_id:
            return Match.model_validate(raw)
    return None


def list_revisions(match_id: str) -> list[dict]:
    return [row for row in _read_revision_rows() if row.get("match_id") == match_id]


def list_revision_summaries(match_id: str) -> list[RevisionSummary]:
    return [RevisionSummary.model_validate(row) for row in list_revisions(match_id)]


def match_references_account(account_id: str) -> bool:
    for raw in _read_match_rows():
        match = Match.model_validate(raw)
        if match.status == "active" and any(seat.account_id == account_id for seat in match.seats):
            return True
    return False


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate_match(
    seats: Sequence[MatchSeat],
    final_scores: Sequence[int],
    *,
    starting_points: int,
    initial_oya: int,
    force: bool,
    reason: str | None,
    registry,
) -> tuple[list[int], list[ValidationIssue]]:
    """返回 (ranks, issues)。排名恒计算；校验不通过时由调用方决定是否强制保存。"""
    issues: list[ValidationIssue] = []

    if len(seats) != 4:
        issues.append(ValidationIssue(code="seat_count", message=f"必须恰好 4 个座位，得到 {len(seats)}"))
    seat_nums = [seat.seat for seat in seats]
    if len(set(seat_nums)) != len(seat_nums):
        issues.append(ValidationIssue(code="seat_duplicate", message="座位编号重复"))
    account_ids = [seat.account_id for seat in seats]
    if len(set(account_ids)) != len(account_ids):
        issues.append(ValidationIssue(code="account_duplicate", message="同一对局不能重复出现同一账号"))

    for seat in seats:
        account = registry.get_account(seat.account_id)
        if account is None:
            issues.append(ValidationIssue(code="account_unknown", message=f"账号不存在: {seat.account_id}"))
            continue
        if not account.enabled and not force:
            issues.append(
                ValidationIssue(code="account_disabled", message=f"账号已停用: {seat.account_id}（可强制保存）")
            )
        if seat.controller_type is None:
            seat.controller_type = account.default_controller
        if seat.model_identity_id and not registry.identity_belongs_to_account(seat.model_identity_id, seat.account_id):
            issues.append(
                ValidationIssue(
                    code="model_identity_mismatch",
                    message=f"账号 {seat.account_id} 未绑定模型身份 {seat.model_identity_id}",
                )
            )
        if not registry.artifact_belongs_to_identity(seat.model_identity_id, seat.model_artifact_id):
            issues.append(
                ValidationIssue(
                    code="model_artifact_mismatch",
                    message=f"模型产物 {seat.model_artifact_id} 不属于身份 {seat.model_identity_id}",
                )
            )

    if len(final_scores) != 4:
        issues.append(ValidationIssue(code="score_count", message=f"必须恰好 4 个最终分数，得到 {len(final_scores)}"))
    if not all(isinstance(s, int) for s in final_scores):
        issues.append(ValidationIssue(code="score_type", message="最终分数必须为整数"))

    if len(final_scores) == 4 and all(isinstance(s, int) for s in final_scores):
        expected = 4 * starting_points
        if sum(final_scores) != expected:
            issues.append(
                ValidationIssue(
                    code="score_total_mismatch",
                    message=f"总分 {sum(final_scores)} ≠ 期望 {expected}",
                )
            )

    if initial_oya < 0 or initial_oya > 3:
        issues.append(ValidationIssue(code="initial_oya", message=f"initial_oya 必须在 [0,3]，得到 {initial_oya}"))

    if force and not (reason or "").strip():
        issues.append(ValidationIssue(code="force_reason_required", message="强制保存必须填写原因"))

    try:
        ranks = list(final_ranks(list(final_scores), initial_oya=initial_oya))
    except ValueError:
        ranks = [-1, -1, -1, -1]
    return ranks, issues


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def _append_revision(row: dict) -> None:
    with open(_revisions_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_match(match: Match) -> None:
    rows = _read_match_rows()
    out: list[dict] = []
    replaced = False
    for raw in rows:
        if raw.get("match_id") == match.match_id:
            out.append(match.model_dump(by_alias=True))
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        out.append(match.model_dump(by_alias=True))
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out)
    atomic_write_text(_matches_path(), text)


def create_match(payload: MatchCreate, registry) -> Match:
    if not payload.occurred_at:
        raise ValueError("occurred_at 不能为空")
    ranks, issues = validate_match(
        payload.seats,
        payload.final_scores,
        starting_points=payload.starting_points,
        initial_oya=payload.initial_oya,
        force=payload.force,
        reason=payload.reason,
        registry=registry,
    )
    blocking = [issue for issue in issues if issue.code != "force_reason_required"]
    if blocking and not (payload.force and (payload.reason or "").strip()):
        raise ValidationError(issues, "score_total_mismatch" in {i.code for i in issues})

    now = now_iso()
    match_id = _generate_match_id()
    match = Match(
        match_id=match_id,
        occurred_at=payload.occurred_at,
        game_length=payload.game_length,
        rule_set=payload.rule_set,
        starting_points=payload.starting_points,
        initial_oya=payload.initial_oya,
        source=payload.source,
        source_ref=payload.source_ref,
        note=payload.note,
        data_completeness=payload.data_completeness,
        replay_id=payload.replay_id,
        seats=payload.seats,
        final_scores=payload.final_scores,
        ranks=ranks,
        revision=1,
        latest_revision_id=_generate_revision_id(match_id, 1),
        created_at=now,
        updated_at=now,
        created_by="migration" if payload.source == "imported" else "manual",
    )

    with _write_lock, data_lock():
        _rewrite_match(match)
        _append_revision(
            {
                "schema": MATCH_REVISION_SCHEMA,
                "revision_id": match.latest_revision_id,
                "match_id": match.match_id,
                "revision": 1,
                "action": "create",
                "created_at": now,
                "by": "migration" if payload.source == "imported" else "manual-entry",
                "force": payload.force,
                "reason": payload.reason,
                "validation": {"passed": not blocking, "issues": [i.model_dump() for i in issues]},
                "before": None,
                "after": match.model_dump(by_alias=True),
            }
        )
    return match


def revise_match(match_id: str, payload: MatchRevise, registry) -> Match:
    with _write_lock, data_lock():
        current = get_match(match_id)
        if current is None:
            raise KeyError(f"match not found: {match_id}")
        if current.status == "void":
            raise ValueError("已作废的对局不能再修订")

        next_match = current.model_copy(deep=True)
        dump = payload.model_dump(exclude_unset=True, exclude={"force", "reason"})
        for key, value in dump.items():
            if value is not None:
                setattr(next_match, key, value)

        ranks, issues = validate_match(
            next_match.seats,
            next_match.final_scores,
            starting_points=next_match.starting_points,
            initial_oya=next_match.initial_oya,
            force=payload.force,
            reason=payload.reason,
            registry=registry,
        )
        next_match.ranks = ranks
        blocking = [issue for issue in issues if issue.code != "force_reason_required"]
        if blocking and not (payload.force and (payload.reason or "").strip()):
            raise ValidationError(issues, "score_total_mismatch" in {i.code for i in issues})

        next_match.revision = current.revision + 1
        next_match.latest_revision_id = _generate_revision_id(match_id, next_match.revision)
        next_match.updated_at = now_iso()
        _rewrite_match(next_match)
        _append_revision(
            {
                "schema": MATCH_REVISION_SCHEMA,
                "revision_id": next_match.latest_revision_id,
                "match_id": match_id,
                "revision": next_match.revision,
                "action": "revise",
                "created_at": next_match.updated_at,
                "by": "manual-entry",
                "force": payload.force,
                "reason": payload.reason,
                "validation": {"passed": not blocking, "issues": [i.model_dump() for i in issues]},
                "before": current.model_dump(by_alias=True),
                "after": next_match.model_dump(by_alias=True),
            }
        )
        return next_match


def void_match(match_id: str, payload: MatchVoid) -> Match:
    with _write_lock, data_lock():
        current = get_match(match_id)
        if current is None:
            raise KeyError(f"match not found: {match_id}")
        if current.status == "void":
            raise ValueError("对局已作废")
        now = now_iso()
        updated = current.model_copy(
            update={
                "status": "void",
                "void_reason": payload.reason,
                "revision": current.revision + 1,
                "latest_revision_id": _generate_revision_id(match_id, current.revision + 1),
                "updated_at": now,
            }
        )
        _rewrite_match(updated)
        _append_revision(
            {
                "schema": MATCH_REVISION_SCHEMA,
                "revision_id": updated.latest_revision_id,
                "match_id": match_id,
                "revision": updated.revision,
                "action": "void",
                "created_at": now,
                "by": "manual-entry",
                "force": False,
                "reason": payload.reason,
                "validation": {"passed": True, "issues": []},
                "before": current.model_dump(by_alias=True),
                "after": updated.model_dump(by_alias=True),
            }
        )
        return updated


class ValidationError(ValueError):
    """校验失败（对应 HTTP 422），携带 issues 与是否总分不匹配标记。"""

    def __init__(self, issues: list[ValidationIssue], score_mismatch: bool = False):
        super().__init__("; ".join(issue.message for issue in issues))
        self.issues = issues
        self.score_mismatch = score_mismatch
