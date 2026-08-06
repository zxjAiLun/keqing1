# -*- coding: utf-8 -*-
"""participants API 端点直接调用测试（仿 test_ladder_server.py）。"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from participants import api
from participants.schemas import (
    AccountCreate,
    MatchCreate,
    MatchRevise,
    MatchSeat,
    MatchVoid,
)


@pytest.fixture(autouse=True)
def participants_root(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    return root


@pytest.fixture
def four_account_ids():
    ids = ["nick@01", "friend@01", "70k@01", "mortal41b"]
    for aid in ids:
        api.api_create_account(AccountCreate(account_id=aid, display_name=aid, account_type="human"))
    return ids


def _match_create(account_ids, scores):
    return MatchCreate(
        occurred_at="2026-08-06T12:00:00+08:00",
        game_length="hanchan",
        seats=[MatchSeat(seat=i, account_id=account_ids[i]) for i in range(4)],
        final_scores=scores,
    )


def test_account_crud_endpoints():
    created = api.api_create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    assert created.account_id == "nick@01"
    listed = api.api_list_accounts()
    assert [a["account_id"] for a in listed["accounts"]] == ["nick@01"]
    got = api.api_get_account("nick@01")
    assert got["account"]["display_name"] == "Nick"
    from participants.schemas import AccountUpdate

    updated = api.api_update_account("nick@01", AccountUpdate(display_name="Nick v2"))
    assert updated.display_name == "Nick v2"
    # 重名冲突 → 409
    with pytest.raises(HTTPException) as exc:
        api.api_create_account(AccountCreate(account_id="nick@01", display_name="Nick2", account_type="human"))
    assert exc.value.status_code == 409
    # 不存在 → 404
    with pytest.raises(HTTPException) as exc2:
        api.api_get_account("ghost@01")
    assert exc2.value.status_code == 404
    api.api_delete_account("nick@01")
    with pytest.raises(HTTPException) as exc3:
        api.api_get_account("nick@01")
    assert exc3.value.status_code == 404


def test_match_flow_endpoints(four_account_ids):
    resp = api.api_create_match(_match_create(four_account_ids, [25000] * 4))
    match = resp.match
    assert match.revision == 1
    # 详情 + revisions
    detail = api.api_get_match(match.match_id)
    assert detail.match.match_id == match.match_id
    revs = api.api_get_revisions(match.match_id)
    assert len(revs["revisions"]) == 1
    # 修订 → revision 2
    revised = api.api_revise_match(match.match_id, MatchRevise(final_scores=[30000, 25000, 25000, 20000]))
    assert revised.match.revision == 2
    # 作废
    voided = api.api_void_match(match.match_id, MatchVoid(reason="中途结束"))
    assert voided.match.status == "void"
    # 列表 status 过滤
    listed = api.api_list_matches(status="void")
    assert listed.total == 1
    active = api.api_list_matches(status="active")
    assert active.total == 0


def test_create_match_validation_422(four_account_ids):
    with pytest.raises(HTTPException) as exc:
        api.api_create_match(_match_create(four_account_ids, [42300, 28100, 19400, 10201]))
    assert exc.value.status_code == 422
    assert exc.value.detail.get("score_mismatch") is True
    # force + reason 通过
    payload = _match_create(four_account_ids, [42300, 28100, 19400, 10201])
    payload.force = True
    payload.reason = "罚符"
    resp = api.api_create_match(payload)
    assert resp.match.final_scores == [42300, 28100, 19400, 10201]


def test_missing_match_404(four_account_ids):
    with pytest.raises(HTTPException) as exc:
        api.api_get_match("m_missing_000000")
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc2:
        api.api_revise_match("m_missing_000000", MatchRevise())
    assert exc2.value.status_code == 404


def test_account_stats_stub(four_account_ids):
    resp = api.api_account_stats("nick@01")
    assert resp["implemented"] is False
    assert resp["account_id"] == "nick@01"
    with pytest.raises(HTTPException) as exc:
        api.api_account_stats("ghost@01")
    assert exc.value.status_code == 404


def test_model_endpoints(four_account_ids):
    from participants.schemas import ModelArtifactCreate, ModelIdentityCreate

    identity = api.api_create_model(ModelIdentityCreate(model_identity_id="model-70k", label="70k", kind="local_model", account_id="70k@01", artifact_path="ckpt.pth"))
    assert identity.model_identity_id == "model-70k"
    listed = api.api_list_models()
    assert len(listed["identities"]) == 1
    artifact = api.api_add_artifact("model-70k", ModelArtifactCreate(label="70k-fixed.pth", artifact_path="ckpt2.pth"))
    assert artifact["is_current"] is True
    with pytest.raises(HTTPException) as exc:
        api.api_add_artifact("model-ghost", ModelArtifactCreate(label="x", artifact_path="y"))
    assert exc.value.status_code == 404


def test_list_matches_pagination_validation(four_account_ids):
    from fastapi import HTTPException as HE

    with pytest.raises(HE) as exc:
        api.api_list_matches(limit=0)
    assert exc.value.status_code == 422
    with pytest.raises(HE) as exc2:
        api.api_list_matches(offset=-1)
    assert exc2.value.status_code == 422
    # 合法分页
    resp = api.api_list_matches(limit=10, offset=0)
    assert resp.total == 0


def test_alias_api_endpoints(four_account_ids):
    from participants.schemas import ExternalAliasCreate

    created = api.api_create_alias(
        ExternalAliasCreate(provider="tenhou", external_id="keqing1", account_id="nick@01", scope="global")
    )
    assert created["account_id"] == "nick@01"
    listed = api.api_list_aliases(provider="tenhou")
    assert len(listed["aliases"]) == 1
    from fastapi import HTTPException as HE

    with pytest.raises(HE) as exc:
        api.api_create_alias(
            ExternalAliasCreate(provider="tenhou", external_id="x", account_id="ghost@01", scope="global")
        )
    assert exc.value.status_code == 422


def test_intake_preview_endpoint(four_account_ids, monkeypatch):
    from participants import intake

    monkeypatch.setattr(
        intake, "download_tenhou6",
        lambda log_id: {
            "name": ["Nick", "NoName-1", "NoName-2", "FriendID"],
            "rule": {"aka": True},
            "log": [
                [[0, 0, 0], [25000, 25000, 25000, 25000], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ["和了", [5000, -5000, 0, 0], [0, 1]]],
            ],
        },
    )
    from participants.schemas import IntakePreviewRequest

    preview = api.api_intake_preview(IntakePreviewRequest(url="https://tenhou.net/3/?log=20260804gm-0009-2147-32af115e"))
    assert preview["raw_player_names"][0] == "Nick"
    assert preview["duplicate_match_id"] is None
    assert preview["game_length"] == "tonpu"


def test_intake_confirm_endpoint(four_account_ids, monkeypatch):
    from participants import intake
    from participants.schemas import IntakeConfirmRequest, SeatResolution

    monkeypatch.setattr(
        intake, "download_tenhou6",
        lambda log_id: {
            "name": ["Nick", "NoName-1", "NoName-2", "FriendID"],
            "rule": {"aka": True},
            "log": [
                [[0, 0, 0], [25000, 25000, 25000, 25000], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ["和了", [5000, -5000, 0, 0], [0, 1]]],
            ],
        },
    )
    resolutions = [
        SeatResolution(seat=0, action="assign", account_id="nick@01", alias_scope="global"),
        SeatResolution(seat=1, action="create", display_name="Bot A", account_type="managed_bot", alias_scope="session"),
        SeatResolution(seat=2, action="create", display_name="Bot B", account_type="managed_bot", alias_scope="session"),
        SeatResolution(seat=3, action="create", display_name="Friend", account_type="human", alias_scope="global"),
    ]
    resp = api.api_intake_confirm(
        IntakeConfirmRequest(log_id="20260804gm-0009-2147-32af115e", resolutions=resolutions, session_id="s9")
    )
    assert resp.match.provider == "tenhou"
    assert resp.match.data_completeness == "full_replay"
    # 重复确认 → 409
    from fastapi import HTTPException as HE

    with pytest.raises(HE) as exc:
        api.api_intake_confirm(
            IntakeConfirmRequest(log_id="20260804gm-0009-2147-32af115e", resolutions=resolutions, session_id="s9")
        )
    assert exc.value.status_code == 409
    # replay artifact 可查，match_id 一致（P2-2）
    replay = api.api_match_replay_artifact(resp.match.match_id)
    assert replay["replay_id"] == "20260804gm-0009-2147-32af115e"
    assert replay["has_events"] is True
    assert replay["match_id"] == resp.match.match_id
