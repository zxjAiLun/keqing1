# -*- coding: utf-8 -*-
"""Pydantic 模型 — participants（账号/模型/对局账本）API 请求与响应类型。"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

AccountType = Literal["human", "managed_bot", "external_bot"]
ControllerType = Literal["human_ui", "local_model", "external_agent", "manual_only"]
SourceType = Literal["native", "imported", "manual"]
DataCompleteness = Literal["result_only", "hand_summary", "full_replay"]
GameLength = Literal["tonpu", "hanchan"]
SeatNo = Literal[0, 1, 2, 3]

ACCOUNTS_SCHEMA = "keqing.participant.accounts.v1"
MODELS_SCHEMA = "keqing.participant.models.v1"
MATCH_SCHEMA = "keqing.participant.match.v1"
MATCH_REVISION_SCHEMA = "keqing.participant.match_revision.v1"
MIGRATION_STATE_SCHEMA = "keqing.participant.migration_state.v1"


class Account(BaseModel):
    account_id: str
    display_name: str
    account_type: AccountType
    enabled: bool = True
    default_controller: ControllerType
    avatar: str | None = None
    note: str | None = None
    migrated_from_replay: bool = False
    created_at: str
    updated_at: str


class AccountCreate(BaseModel):
    account_id: str | None = None  # 缺省自动生成 slug
    display_name: str = Field(min_length=1)
    account_type: AccountType
    default_controller: ControllerType | None = None  # 缺省按 account_type 推导
    avatar: str | None = None
    note: str | None = None


class AccountUpdate(BaseModel):
    display_name: str | None = None
    enabled: bool | None = None
    default_controller: ControllerType | None = None
    avatar: str | None = None
    note: str | None = None


class ModelArtifact(BaseModel):
    model_artifact_id: str
    label: str
    model_identity_id: str
    artifact_path: str | None = None
    hash: str | None = None
    is_current: bool = True
    created_at: str
    retired_at: str | None = None


class ModelIdentity(BaseModel):
    model_identity_id: str
    label: str
    kind: Literal["local_model", "external_agent", "none"]
    account_id: str | None = None
    is_current: bool = True
    created_at: str
    retired_at: str | None = None
    note: str | None = None
    artifacts: list[ModelArtifact] = Field(default_factory=list)


class ModelIdentityCreate(BaseModel):
    model_identity_id: str | None = None
    label: str = Field(min_length=1)
    kind: Literal["local_model", "external_agent", "none"]
    account_id: str | None = None
    artifact_path: str | None = None
    note: str | None = None


class ModelIdentityUpdate(BaseModel):
    label: str | None = None
    kind: Literal["local_model", "external_agent", "none"] | None = None
    account_id: str | None = None
    is_current: bool | None = None
    note: str | None = None


class ModelArtifactCreate(BaseModel):
    model_artifact_id: str | None = None
    label: str = Field(min_length=1)
    artifact_path: str | None = None


class MatchSeat(BaseModel):
    seat: SeatNo
    account_id: str
    controller_type: ControllerType | None = None  # 缺省取 account.default_controller
    model_identity_id: str | None = None
    model_artifact_id: str | None = None


class MatchCreate(BaseModel):
    occurred_at: str
    game_length: GameLength
    rule_set: str = "standard-4p"
    starting_points: int = 25000
    initial_oya: int = 0
    source: SourceType = "manual"
    source_ref: str | None = None
    note: str | None = None
    data_completeness: DataCompleteness = "result_only"
    replay_id: str | None = None
    seats: list[MatchSeat]
    final_scores: list[int]
    force: bool = False
    reason: str | None = None


class MatchRevise(BaseModel):
    occurred_at: str | None = None
    game_length: GameLength | None = None
    rule_set: str | None = None
    starting_points: int | None = None
    initial_oya: int | None = None
    note: str | None = None
    data_completeness: DataCompleteness | None = None
    seats: list[MatchSeat] | None = None
    final_scores: list[int] | None = None
    force: bool = False
    reason: str | None = None


class MatchVoid(BaseModel):
    reason: str = Field(min_length=1)


class Match(BaseModel):  # matches.jsonl 行（当前态）
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default=MATCH_SCHEMA, alias="schema")
    match_id: str
    occurred_at: str
    game_length: GameLength
    rule_set: str
    starting_points: int
    initial_oya: int
    source: SourceType
    source_ref: str | None = None
    note: str | None = None
    data_completeness: DataCompleteness
    replay_id: str | None = None
    seats: list[MatchSeat]
    final_scores: list[int]
    ranks: list[int]
    status: Literal["active", "void"] = "active"
    void_reason: str | None = None
    revision: int
    latest_revision_id: str
    created_at: str
    updated_at: str
    created_by: Literal["manual", "migration", "system"] = "manual"


class ValidationIssue(BaseModel):
    code: str
    message: str


class RevisionSummary(BaseModel):
    revision_id: str
    match_id: str
    revision: int
    action: Literal["create", "revise", "void"]
    created_at: str
    by: str
    force: bool
    reason: str | None = None
    validation: dict


class MatchResponse(BaseModel):
    match: Match
    revisions: list[RevisionSummary]


class MatchListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default=MATCH_SCHEMA, alias="schema")
    matches: list[Match]
    total: int
