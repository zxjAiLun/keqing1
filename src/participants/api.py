# -*- coding: utf-8 -*-
"""participants API：账号/模型/对局账本 HTTP 端点。

前缀 ``/api/participants``。错误映射：
- ``ledger.ValidationError`` → 422（携带 issues 与 score_mismatch）
- ``KeyError`` → 404
- ``ValueError``（ID 冲突等）→ 409
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import ledger, registry, stats
from .schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    Match,
    MatchCreate,
    MatchListResponse,
    MatchResponse,
    MatchRevise,
    MatchVoid,
    ModelArtifactCreate,
    ModelIdentity,
    ModelIdentityCreate,
    ModelIdentityUpdate,
)
from .ledger import ValidationError

router = APIRouter(prefix="/api/participants")


def _error(status: int, message: str):
    return HTTPException(status_code=status, detail={"error": message})


# ---------------------------------------------------------------------------
# 账号
# ---------------------------------------------------------------------------

@router.get("/accounts", response_model=dict)
def api_list_accounts(
    enabled: bool | None = None,
    q: str | None = None,
) -> dict:
    return {"schema": "keqing.participant.accounts.v1", "accounts": [a.model_dump() for a in registry.list_accounts(enabled=enabled, q=q)]}


@router.post("/accounts", response_model=Account)
def api_create_account(payload: AccountCreate) -> Account:
    try:
        return registry.create_account(payload)
    except ValueError as exc:
        raise _error(409, str(exc)) from exc


@router.get("/accounts/{account_id}", response_model=dict)
def api_get_account(account_id: str) -> dict:
    account = registry.get_account(account_id)
    if account is None:
        raise _error(404, f"account not found: {account_id}")
    identities = [m.model_dump() for m in registry.list_models() if m.account_id == account_id]
    return {"account": account.model_dump(), "identities": identities}


@router.patch("/accounts/{account_id}", response_model=Account)
def api_update_account(account_id: str, payload: AccountUpdate) -> Account:
    try:
        return registry.update_account(account_id, payload)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc


@router.delete("/accounts/{account_id}", response_model=dict)
def api_delete_account(account_id: str) -> dict:
    try:
        referenced = ledger.match_references_account(account_id)
        return registry.delete_account(account_id, referenced=referenced)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc


@router.get("/accounts/{account_id}/stats", response_model=dict)
def api_account_stats(account_id: str) -> dict:
    if registry.get_account(account_id) is None:
        raise _error(404, f"account not found: {account_id}")
    return stats.compute_account_stats(account_id, registry, ledger)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

@router.get("/models", response_model=dict)
def api_list_models() -> dict:
    return {"schema": "keqing.participant.models.v1", "identities": [m.model_dump() for m in registry.list_models()]}


@router.post("/models", response_model=ModelIdentity)
def api_create_model(payload: ModelIdentityCreate) -> ModelIdentity:
    try:
        return registry.create_model_identity(payload)
    except ValueError as exc:
        raise _error(409, str(exc)) from exc


@router.post("/models/{model_identity_id}/artifacts", response_model=dict)
def api_add_artifact(model_identity_id: str, payload: ModelArtifactCreate) -> dict:
    try:
        artifact = registry.add_model_artifact(model_identity_id, payload)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc
    except ValueError as exc:
        raise _error(409, str(exc)) from exc
    return artifact.model_dump()


@router.patch("/models/{model_identity_id}", response_model=ModelIdentity)
def api_update_model(model_identity_id: str, payload: ModelIdentityUpdate) -> ModelIdentity:
    try:
        return registry.update_model_identity(model_identity_id, payload)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc


# ---------------------------------------------------------------------------
# 对局账本
# ---------------------------------------------------------------------------

@router.get("/matches", response_model=MatchListResponse)
def api_list_matches(
    source: str | None = None,
    status: str | None = None,
    account_id: str | None = None,
    from_at: str | None = None,
    to_at: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> MatchListResponse:
    return ledger.list_matches(
        source=source,
        status=status,
        account_id=account_id,
        from_at=from_at,
        to_at=to_at,
        limit=limit,
        offset=offset,
    )


@router.post("/matches", response_model=MatchResponse)
def api_create_match(payload: MatchCreate) -> MatchResponse:
    try:
        match = ledger.create_match(payload, registry)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "校验失败", "issues": [i.model_dump() for i in exc.issues], "score_mismatch": exc.score_mismatch},
        ) from exc
    except ValueError as exc:
        raise _error(409, str(exc)) from exc
    return MatchResponse(match=match, revisions=ledger.list_revision_summaries(match.match_id))


@router.get("/matches/{match_id}", response_model=MatchResponse)
def api_get_match(match_id: str) -> MatchResponse:
    match = ledger.get_match(match_id)
    if match is None:
        raise _error(404, f"match not found: {match_id}")
    return MatchResponse(match=match, revisions=ledger.list_revision_summaries(match_id))


@router.get("/matches/{match_id}/revisions", response_model=dict)
def api_get_revisions(match_id: str) -> dict:
    if ledger.get_match(match_id) is None:
        raise _error(404, f"match not found: {match_id}")
    return {"match_id": match_id, "revisions": ledger.list_revisions(match_id)}


@router.post("/matches/{match_id}/revise", response_model=MatchResponse)
def api_revise_match(match_id: str, payload: MatchRevise) -> MatchResponse:
    try:
        match = ledger.revise_match(match_id, payload, registry)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "校验失败", "issues": [i.model_dump() for i in exc.issues], "score_mismatch": exc.score_mismatch},
        ) from exc
    except ValueError as exc:
        raise _error(409, str(exc)) from exc
    return MatchResponse(match=match, revisions=ledger.list_revision_summaries(match_id))


@router.post("/matches/{match_id}/void", response_model=MatchResponse)
def api_void_match(match_id: str, payload: MatchVoid) -> MatchResponse:
    try:
        match = ledger.void_match(match_id, payload)
    except KeyError as exc:
        raise _error(404, str(exc)) from exc
    except ValueError as exc:
        raise _error(409, str(exc)) from exc
    return MatchResponse(match=match, revisions=ledger.list_revision_summaries(match_id))
