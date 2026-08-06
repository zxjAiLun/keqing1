# -*- coding: utf-8 -*-
"""R10-E：start 端点 roster 模式——移除 quantity=3 门槛 + 持久化 roster + session 别名。"""
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gateway.api.playwithyou as pw  # noqa: E402


class FakeProc:
    pid = 9999
    stdout = io.StringIO("")

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass

    def communicate(self):
        return ("", "")


@pytest.fixture
def pw_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "LAUNCHER", tmp_path / "launch_tenhou_bots.py")
    (tmp_path / "launch_tenhou_bots.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(pw, "_find_owned_pids", lambda: [])
    monkeypatch.setattr(
        pw, "_resolve_spec",
        lambda network, custom_paths, slot: "70k" if network == "mortal" else None,
    )
    fake_subprocess = mock.Mock()
    fake_subprocess.Popen.return_value = FakeProc()
    monkeypatch.setattr(pw, "subprocess", fake_subprocess)
    monkeypatch.setattr(pw, "_ladder_data_root", lambda: tmp_path / "ladder")
    monkeypatch.setattr(pw, "_ladder_config_dir", lambda: tmp_path / "configs")
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(tmp_path / "participants"))
    pw.SESSIONS.clear()
    pw._HISTORY.clear()
    # P1-2：roster 校验要求参与者账号存在且启用
    from participants import registry
    from participants.schemas import AccountCreate, ModelIdentityCreate

    for account_id, account_type in (
        ("nick@01", "human"),
        ("70k@01", "managed_bot"),
        ("70k@02", "managed_bot"),
        ("mortal@01", "external_bot"),
    ):
        registry.create_account(
            AccountCreate(account_id=account_id, display_name=account_id, account_type=account_type)
        )
    registry.create_model_identity(
        ModelIdentityCreate(model_identity_id="70k", label="70k", kind="local_model", artifact_path="ckpt.pth")
    )
    return tmp_path


def _roster_request():
    from gateway.api.playwithyou import ParticipantBindingRequest, StartPlayWithYouRequest

    return StartPlayWithYouRequest(
        networks=["mortal", "mortal", "none", "none"],
        roster=[
            ParticipantBindingRequest(account_id="nick@01", controller_type="human_ui"),
            ParticipantBindingRequest(account_id="70k@01", controller_type="local_model", launcher_slot=0, expected_raw_name="NoName-1"),
            ParticipantBindingRequest(account_id="70k@02", controller_type="local_model", launcher_slot=1, expected_raw_name="NoName-2"),
            ParticipantBindingRequest(account_id="mortal@01", controller_type="external_agent"),
        ],
    )


def test_start_with_roster_quantity_two(pw_env):
    """R10-E：Nick human + 2 本地 bot + 外部 Mortal（quantity=2）可启动捕获。"""
    status = pw.start_playwithyou(_roster_request())
    assert status.running is True
    assert len(status.bots) == 2  # quantity=2 合法，不再要求恰好 1 人类 + 3 bot

    capture_dir = pw_env / "ladder" / "captures" / "playwithyou" / status.session_id
    binding = json.loads((capture_dir / "binding.json").read_text(encoding="utf-8"))
    assert binding["mode"] == "roster"
    assert len(binding["roster"]) == 4
    # P1-2：launcher 模型身份/产物已从 spec 冻结（70k → 70k identity + current artifact）
    launched = [e for e in binding["roster"] if e.get("launcher_slot") is not None]
    assert len(launched) == 2
    assert all(e.get("model_identity_id") == "70k" for e in launched)
    assert all(e.get("model_artifact_id") for e in launched)

    # session-scoped 别名：NoName-1 → 70k@01 / NoName-2 → 70k@02
    from participants import aliases

    session_aliases = [
        a for a in aliases.list_aliases()
        if a.scope == "session" and a.session_id == status.session_id
    ]
    assert len(session_aliases) == 2
    assert any(a.external_id == "NoName-1" and a.account_id == "70k@01" and a.model_identity_id == "70k" for a in session_aliases)
    assert any(a.external_id == "NoName-2" and a.account_id == "70k@02" and a.model_identity_id == "70k" for a in session_aliases)


def test_start_with_roster_rejects_invalid_length(pw_env):
    from fastapi import HTTPException
    from gateway.api.playwithyou import ParticipantBindingRequest, StartPlayWithYouRequest

    req = StartPlayWithYouRequest(
        networks=["mortal", "none", "none", "none"],
        roster=[
            ParticipantBindingRequest(account_id="nick@01", controller_type="human_ui"),
            ParticipantBindingRequest(account_id="70k@01", controller_type="local_model", launcher_slot=0),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        pw.start_playwithyou(req)
    assert exc.value.status_code == 400
    assert "4" in str(exc.value.detail)


def test_start_with_roster_requires_launcher_slot(pw_env):
    from fastapi import HTTPException
    from gateway.api.playwithyou import ParticipantBindingRequest, StartPlayWithYouRequest

    req = StartPlayWithYouRequest(
        networks=["none", "none", "none", "none"],
        roster=[
            ParticipantBindingRequest(account_id="nick@01", controller_type="human_ui"),
            ParticipantBindingRequest(account_id="70k@01", controller_type="local_model"),
            ParticipantBindingRequest(account_id="70k@02", controller_type="local_model"),
            ParticipantBindingRequest(account_id="mortal@01", controller_type="external_agent"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        pw.start_playwithyou(req)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# R10-E Repair：真实 launcher 接线（P1-1）
# ---------------------------------------------------------------------------

def test_launcher_build_configs_roster_wiring(tmp_path, monkeypatch):
    """P1-1：真实 roster binding.json → _build_configs → 2 configs + collector 绑定账号。"""
    import argparse
    import sys as _sys

    from scripts import launch_tenhou_bots as launcher

    # 模拟 spec 解析与设备选择，聚焦 binding 接线
    monkeypatch.setattr(launcher, "resolve_bot_spec", lambda spec, root: ("checkpoint", tmp_path / f"{spec}.pth"))
    monkeypatch.setattr(launcher, "_pick_device", lambda device: "cpu")
    monkeypatch.setattr(launcher, "normalize_tenhou_room", lambda room, **kw: "L2147_9")

    capture_dir = tmp_path / "capture"
    (capture_dir / "pending").mkdir(parents=True)
    binding = {
        "session_id": "s_roster",
        "season_id": "",
        "human_account_id": "",
        "bot_account_ids": [],
        "mode": "roster",
        "roster": [
            {"account_id": "nick@01", "controller_type": "human_ui", "launcher_slot": None, "expected_raw_name": "Nick"},
            {"account_id": "70k@01", "controller_type": "local_model", "launcher_slot": 0, "expected_raw_name": "NoName-1", "model_identity_id": "70k", "model_artifact_id": "a1"},
            {"account_id": "70k@02", "controller_type": "local_model", "launcher_slot": 1, "expected_raw_name": "NoName-2", "model_identity_id": "70k", "model_artifact_id": "a2"},
            {"account_id": "mortal@01", "controller_type": "external_agent", "launcher_slot": None},
        ],
        "frozen_at": 0.0,
    }
    (capture_dir / "binding.json").write_text(json.dumps(binding, ensure_ascii=False), encoding="utf-8")

    args = argparse.Namespace(
        bots=["70k", "70k"],
        name_prefix="NoName",
        room="2147",
        device="cuda",
        game_type="hanchan",
        gateway_host="127.0.0.1",
        gateway_port=12101,
        bot_verbose=False,
        think_delay=0.0,
        ladder_capture_dir=str(capture_dir),
    )
    configs, collector = launcher._build_configs(args)
    assert len(configs) == 2
    assert collector is not None
    assert collector.binding.mode == "roster"
    assert len(collector.binding.roster) == 4
    # launcher slot 顺序 → config 顺序（NoName-1 → 70k@01，NoName-2 → 70k@02）
    assert configs[0].ladder_account_id == "70k@01"
    assert configs[1].ladder_account_id == "70k@02"
    assert configs[0].capture_sink is collector
    assert configs[1].capture_sink is collector


def test_launcher_roster_slot_sorting(tmp_path, monkeypatch):
    """P1-2：roster 顺序与 launcher_slot 顺序不同时，按 slot 对齐（NoName-1 不串线）。"""
    import argparse

    from scripts import launch_tenhou_bots as launcher

    monkeypatch.setattr(launcher, "resolve_bot_spec", lambda spec, root: ("checkpoint", tmp_path / f"{spec}.pth"))
    monkeypatch.setattr(launcher, "_pick_device", lambda device: "cpu")
    monkeypatch.setattr(launcher, "normalize_tenhou_room", lambda room, **kw: "L2147_9")

    capture_dir = tmp_path / "capture"
    (capture_dir / "pending").mkdir(parents=True)
    binding = {
        "session_id": "s_roster",
        "season_id": "",
        "human_account_id": "",
        "bot_account_ids": [],
        "mode": "roster",
        "roster": [
            {"account_id": "nick@01", "controller_type": "human_ui", "launcher_slot": None},
            {"account_id": "70k@02", "controller_type": "local_model", "launcher_slot": 1, "expected_raw_name": "NoName-2"},
            {"account_id": "70k@01", "controller_type": "local_model", "launcher_slot": 0, "expected_raw_name": "NoName-1"},
            {"account_id": "mortal@01", "controller_type": "external_agent", "launcher_slot": None},
        ],
        "frozen_at": 0.0,
    }
    (capture_dir / "binding.json").write_text(json.dumps(binding, ensure_ascii=False), encoding="utf-8")

    args = argparse.Namespace(
        bots=["70k", "70k"],
        name_prefix="NoName",
        room="2147",
        device="cuda",
        game_type="hanchan",
        gateway_host="127.0.0.1",
        gateway_port=12101,
        bot_verbose=False,
        think_delay=0.0,
        ladder_capture_dir=str(capture_dir),
    )
    configs, collector = launcher._build_configs(args)
    # 按 launcher_slot 排序：slot0 → 70k@01（NoName-1），slot1 → 70k@02（NoName-2）
    assert configs[0].ladder_account_id == "70k@01"
    assert configs[1].ladder_account_id == "70k@02"


def test_roster_and_ladder_capture_mutually_exclusive(pw_env):
    """P2：roster 与旧正式天梯绑定互斥 → 400。"""
    from fastapi import HTTPException
    from gateway.api.playwithyou import LadderCaptureRequest

    req = _roster_request()
    req.ladder_capture = LadderCaptureRequest(
        enabled=True, season_id="official-ladder-v1", human_account_id="nick@01",
        bot_account_ids=["70k@01", "70k@02", "70k@03"], mode="confirm",
    )
    with pytest.raises(HTTPException) as exc:
        pw.start_playwithyou(req)
    assert exc.value.status_code == 400
    assert "不能同时开启" in str(exc.value.detail)


def test_roster_rejects_unknown_account(pw_env):
    """P1-2：roster 引用的账号不存在 → 呼出前拒绝。"""
    from fastapi import HTTPException
    from gateway.api.playwithyou import ParticipantBindingRequest, StartPlayWithYouRequest

    req = StartPlayWithYouRequest(
        networks=["mortal", "mortal", "none", "none"],
        roster=[
            ParticipantBindingRequest(account_id="ghost@01", controller_type="human_ui"),
            ParticipantBindingRequest(account_id="70k@01", controller_type="local_model", launcher_slot=0),
            ParticipantBindingRequest(account_id="70k@02", controller_type="local_model", launcher_slot=1),
            ParticipantBindingRequest(account_id="mortal@01", controller_type="external_agent"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        pw.start_playwithyou(req)
    assert exc.value.status_code == 400
    assert "ghost@01" in str(exc.value.detail)
    # 无残留：未创建 capture 目录 / session 别名
    from participants import aliases

    assert aliases.list_aliases() == []
    assert not list((pw_env / "ladder" / "captures" / "playwithyou").glob("*")) if (pw_env / "ladder" / "captures" / "playwithyou").exists() else True


def test_roster_rejects_disabled_account(pw_env):
    """P1-2：roster 引用的账号已停用 → 呼出前拒绝。"""
    from fastapi import HTTPException
    from gateway.api.playwithyou import ParticipantBindingRequest, StartPlayWithYouRequest
    from participants import registry
    from participants.schemas import AccountUpdate

    registry.update_account("70k@01", AccountUpdate(enabled=False))
    req = StartPlayWithYouRequest(
        networks=["mortal", "mortal", "none", "none"],
        roster=[
            ParticipantBindingRequest(account_id="nick@01", controller_type="human_ui"),
            ParticipantBindingRequest(account_id="70k@01", controller_type="local_model", launcher_slot=0),
            ParticipantBindingRequest(account_id="70k@02", controller_type="local_model", launcher_slot=1),
            ParticipantBindingRequest(account_id="mortal@01", controller_type="external_agent"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        pw.start_playwithyou(req)
    assert exc.value.status_code == 400
    assert "停用" in str(exc.value.detail)
