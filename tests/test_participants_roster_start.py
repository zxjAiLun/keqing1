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

    # session-scoped 别名：NoName-1 → 70k@01 / NoName-2 → 70k@02
    from participants import aliases

    session_aliases = [
        a for a in aliases.list_aliases()
        if a.scope == "session" and a.session_id == status.session_id
    ]
    assert len(session_aliases) == 2
    assert any(a.external_id == "NoName-1" and a.account_id == "70k@01" for a in session_aliases)
    assert any(a.external_id == "NoName-2" and a.account_id == "70k@02" for a in session_aliases)


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
