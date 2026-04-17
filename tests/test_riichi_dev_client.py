from __future__ import annotations

from pathlib import Path
import json
import asyncio

import gateway.riichi_dev_client as rdc


class FakeObservation:
    def __init__(self, player_id: int = 2):
        self.player_id = player_id

    def to_dict(self):
        return {
            "actor": self.player_id,
            "hand": ["1m"] * 14,
            "melds": [[], [], [], []],
            "discards": [[], [], [], []],
            "scores": [25000, 25000, 25000, 25000],
            "dora_markers": ["1p"],
            "reached": [False, False, False, False],
        }

    def legal_actions(self):
        return []


class StubAgent(rdc.RiichiDevDecisionAgent):
    def __init__(self, action):
        self.action = action
        self.calls = []

    def select_action(self, message, seat):
        self.calls.append((message, seat))
        return dict(self.action)


def test_keqing_agent_scores_request_action_from_observation(monkeypatch) -> None:
    fake_obs = FakeObservation(player_id=2)

    def fake_decode(message):
        assert message["observation"] == "encoded"
        return fake_obs, fake_obs.to_dict()

    class FakeScorer:
        def score(self, ctx):
            assert ctx.actor == 2
            assert ctx.legal_actions == [{"type": "none"}]
            assert ctx.model_snap["actor"] == 2
            return type("Decision", (), {"chosen": {"type": "none"}})()

    monkeypatch.setattr(rdc, "_decode_observation", fake_decode)
    monkeypatch.setattr(
        rdc.KeqingModelAdapter,
        "from_checkpoint",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    monkeypatch.setattr(rdc, "DefaultActionScorer", lambda **kwargs: FakeScorer())

    agent = rdc.ObservationScoringAgent(model_path=Path("fake.pth"), device="cpu")
    action = agent.select_action(
        {
            "type": "request_action",
            "observation": "encoded",
            "possible_actions": [{"type": "none"}],
        },
        seat=None,
    )

    assert action == {"type": "none"}


def test_client_tracks_seat_and_only_replies_to_request_action() -> None:
    agent = StubAgent({"type": "dahai", "pai": "3m", "actor": 1})
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(token="t", model_path=Path("fake.pth")),
        agent=agent,
    )

    assert client.handle_message({"type": "start_game", "id": 1}) is None
    assert client.seat == 1
    assert client.handle_message({"type": "tsumo", "actor": 1, "pai": "4m"}) is None

    response = client.handle_message(
        {"type": "request_action", "possible_actions": [{"type": "none"}]}
    )

    assert response == {"type": "dahai", "pai": "3m", "actor": 1}
    assert agent.calls[0][1] == 1


def test_sanitize_none_removes_actor() -> None:
    assert rdc._sanitize_action({"type": "none", "actor": 3}, actor_hint=3) == {
        "type": "none"
    }


def test_sanitize_hora_keeps_tsumo_target() -> None:
    assert rdc._sanitize_action({"type": "hora", "actor": 1, "target": 1}, actor_hint=1) == {
        "type": "hora",
        "actor": 1,
        "target": 1,
    }


def test_sanitize_hora_keeps_ron_fields() -> None:
    assert rdc._sanitize_action(
        {"type": "hora", "actor": 1, "target": 3, "pai": "C"},
        actor_hint=1,
    ) == {
        "type": "hora",
        "actor": 1,
        "target": 3,
        "pai": "C",
    }


def test_sanitize_ryukyoku_drops_actor() -> None:
    assert rdc._sanitize_action({"type": "ryukyoku", "actor": 0}, actor_hint=0) == {
        "type": "ryukyoku"
    }


def test_sanitize_dahai_keeps_only_protocol_fields() -> None:
    assert rdc._sanitize_action(
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True, "foo": 1},
        actor_hint=0,
    ) == {
        "type": "dahai",
        "actor": 0,
        "pai": "3m",
        "tsumogiri": True,
    }


def test_sanitize_ankan_keeps_required_fields() -> None:
    assert rdc._sanitize_action(
        {"type": "ankan", "actor": 0, "consumed": ["F", "F", "F", "F"], "extra": 1},
        actor_hint=0,
    ) == {
        "type": "ankan",
        "actor": 0,
        "consumed": ["F", "F", "F", "F"],
    }


def test_sanitize_chi_keeps_required_fields() -> None:
    assert rdc._sanitize_action(
        {"type": "chi", "actor": 0, "target": 3, "pai": "4p", "consumed": ["5p", "6p"], "x": 1},
        actor_hint=0,
    ) == {
        "type": "chi",
        "actor": 0,
        "target": 3,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }


def test_sanitize_pon_without_target_does_not_crash() -> None:
    assert rdc._sanitize_action(
        {"type": "pon", "actor": 0, "pai": "5sr", "consumed": ["5s", "5s"]},
        actor_hint=0,
    ) == {
        "type": "pon",
        "actor": 0,
        "pai": "5sr",
        "consumed": ["5s", "5s"],
    }


def test_action_to_mjai_dict_accepts_json_string() -> None:
    assert rdc._action_to_mjai_dict('{"type":"none"}') == {"type": "none"}


def test_ws_url_override_takes_precedence() -> None:
    cfg = rdc.RiichiDevClientConfig(
        token="t",
        queue="validate",
        base_url="wss://riichi.dev",
        ws_url_override="wss://games.riichi.dev/ws/validate",
        model_path=Path("fake.pth"),
    )

    assert cfg.ws_url() == "wss://games.riichi.dev/ws/validate"


def test_agent_converts_string_device_before_building_adapter(monkeypatch) -> None:
    captured = {}

    def fake_from_checkpoint(cls, *args, **kwargs):
        captured["device"] = kwargs["device"]
        return object()

    class FakeScorer:
        def score(self, ctx):
            return type("Decision", (), {"chosen": {"type": "none"}})()

    monkeypatch.setattr(
        rdc.KeqingModelAdapter,
        "from_checkpoint",
        classmethod(fake_from_checkpoint),
    )
    monkeypatch.setattr(rdc, "DefaultActionScorer", lambda **kwargs: FakeScorer())

    rdc.ObservationScoringAgent(model_path=Path("fake.pth"), device="cpu")

    assert str(captured["device"]) == "cpu"


def test_resolve_ws_url_accepts_https_base_url() -> None:
    assert (
        rdc._resolve_ws_url("https://riichi.dev", "validate")
        == "wss://riichi.dev/ws/validate"
    )


def test_resolve_default_token_prefers_mochakey_for_keqingv31(monkeypatch) -> None:
    monkeypatch.setenv("LATTEKEY", "latte-token")
    monkeypatch.setenv("MOCHAKEY", "mocha-token")

    assert rdc._resolve_default_token("keqingv31") == "mocha-token"


def test_resolve_default_token_uses_lattekey_for_other_bots(monkeypatch) -> None:
    monkeypatch.setenv("LATTEKEY", "latte-token")
    monkeypatch.setenv("MOCHAKEY", "mocha-token")

    assert rdc._resolve_default_token("keqingv3") == "latte-token"


def test_resolve_default_token_with_source_prefers_mochakey(monkeypatch) -> None:
    monkeypatch.setenv("LATTEKEY", "latte-token")
    monkeypatch.setenv("MOCHAKEY", "mocha-token")

    assert rdc._resolve_default_token_with_source("keqingv31") == ("mocha-token", "MOCHAKEY")


def test_resolve_model_path_uses_default_checkpoint() -> None:
    assert rdc._resolve_model_path(
        bot_name="keqingv31",
        project_root=Path("/tmp/project"),
        model_path=None,
    ) == Path("/tmp/project/artifacts/models/keqingv31/best.pth")


def test_decode_jwt_payload_unverified_reads_name_and_bot_id() -> None:
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJuYW1lIjoibW9jaGEiLCJ0eXBlIjoiYm90IiwiYm90X2lkIjoiYm90LTEyMyJ9."
        "sig"
    )

    payload = rdc._decode_jwt_payload_unverified(token)

    assert payload == {"name": "mocha", "type": "bot", "bot_id": "bot-123"}


def test_create_agent_supports_non_keqingv3_model_path(monkeypatch) -> None:
    created = {}

    class FakeObservationAgent:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(rdc, "ObservationScoringAgent", FakeObservationAgent)
    agent = rdc.create_riichi_dev_agent(
        bot_name="keqingv31",
        project_root=Path("."),
        model_path=None,
        device="cpu",
        verbose=False,
    )

    assert isinstance(agent, FakeObservationAgent)
    assert created["model_version"] == "keqingv31"
    assert created["hidden_dim"] == 320
    assert created["num_res_blocks"] == 6


def test_create_agent_supports_keqingv4_spec(monkeypatch) -> None:
    created = {}

    class FakeObservationAgent:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(rdc, "ObservationScoringAgent", FakeObservationAgent)
    agent = rdc.create_riichi_dev_agent(
        bot_name="keqingv4",
        project_root=Path("."),
        model_path=None,
        device="cpu",
        verbose=False,
    )

    assert isinstance(agent, FakeObservationAgent)
    assert created["model_version"] == "keqingv4"
    assert created["hidden_dim"] == 320
    assert created["num_res_blocks"] == 6


def test_create_agent_rejects_rulebase() -> None:
    try:
        rdc.create_riichi_dev_agent(
            bot_name="rulebase",
            project_root=Path("."),
            model_path=None,
            device="cpu",
            verbose=False,
        )
    except ValueError as exc:
        assert "rulebase" in str(exc)
    else:
        raise AssertionError("expected rulebase to be rejected")


def test_local_game_returns_summary(monkeypatch) -> None:
    class FakeAction:
        def to_mjai(self):
            return {"type": "none"}

    class FakeObs:
        def __init__(self, pid):
            self.player_id = pid

        def legal_actions(self):
            return [FakeAction()] if self.player_id == 0 else []

    class FakeEnv:
        def __init__(self, game_mode=None, seed=None):
            self._done = False
            self.mjai_log = [{"type": "start_game"}, {"type": "end_game"}]

        def get_observations(self):
            return {i: FakeObs(i) for i in range(4)}

        def done(self):
            return self._done

        def step(self, action):
            self._done = True
            return {i: FakeObs(i) for i in range(4)}

        def scores(self):
            return [25000, 25000, 25000, 25000]

        def ranks(self):
            return [1, 2, 3, 4]

    class FakeAgent(rdc.RiichiDevDecisionAgent):
        def act(self, obs):
            return object()

        def select_action(self, message, seat):
            return {"type": "none"}

    monkeypatch.setattr(rdc, "RiichiEnv", FakeEnv)
    result = rdc.run_local_game(agent=FakeAgent(), game_mode=2, seed=42)

    assert result["scores"] == [25000, 25000, 25000, 25000]
    assert result["ranks"] == [1, 2, 3, 4]
    assert result["step_count"] == 1


def test_create_agent_passes_model_version_and_weights(monkeypatch) -> None:
    captured = {}

    class FakeObservationAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(rdc, "ObservationScoringAgent", FakeObservationAgent)

    agent = rdc.create_riichi_dev_agent(
        bot_name="futurev5",
        project_root=Path("."),
        model_path=Path("artifacts/models/futurev5/custom_best.pth"),
        device="cpu",
        verbose=False,
        model_version="futurev5_exp",
    )

    assert isinstance(agent, FakeObservationAgent)
    assert captured["model_path"] == Path("artifacts/models/futurev5/custom_best.pth")
    assert captured["model_version"] == "futurev5_exp"
    assert captured["beam_k"] == rdc.DEFAULT_DECISION_AGENT_SPEC.beam_k
    assert captured["beam_lambda"] == rdc.DEFAULT_DECISION_AGENT_SPEC.beam_lambda
    assert captured["score_delta_lambda"] == rdc.DEFAULT_DECISION_AGENT_SPEC.score_delta_lambda
    assert captured["win_prob_lambda"] == rdc.DEFAULT_DECISION_AGENT_SPEC.win_prob_lambda
    assert captured["dealin_prob_lambda"] == rdc.DEFAULT_DECISION_AGENT_SPEC.dealin_prob_lambda


def test_audit_logger_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = rdc.RiichiDevAuditLogger(log_path)
    logger.write({"kind": "event", "message": {"type": "start_game"}})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["kind"] == "event"
    assert payload["message"]["type"] == "start_game"
    assert "ts" in payload


def test_enrich_message_for_audit_adds_observation_hash_without_state_on_decode_failure(monkeypatch) -> None:
    monkeypatch.setattr(rdc, "_decode_observation", lambda message: (_ for _ in ()).throw(ValueError("bad obs")))

    enriched = rdc._enrich_message_for_audit({"type": "request_action", "observation": "encoded"})

    assert enriched["_observation_sha256"]
    assert "ValueError: bad obs" == enriched["_observation_decode_error"]
    assert "_normalized_observation" not in enriched


def test_client_logs_request_action(tmp_path: Path) -> None:
    agent = StubAgent({"type": "dahai", "pai": "3m", "actor": 1})
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(
            token="t",
            model_path=Path("fake.pth"),
            audit_log_path=tmp_path / "riichi.jsonl",
        ),
        agent=agent,
    )
    client.handle_message({"type": "start_game", "id": 1})
    message = {
        "type": "request_action",
        "possible_actions": [{"type": "none"}],
        "_normalized_observation": {
            "actor": 1,
            "hand": ["1m"],
            "melds": [[], [], [], []],
            "discards": [[], [], [], []],
            "reached": [False, False, False, False],
            "dora_markers": [],
            "scores": [25000] * 4,
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
        },
    }
    response = client.handle_message(message)
    client.audit_logger.log_request_action(
        queue=client.config.queue,
        bot_name=client.config.bot_name,
        model_version=client.config.model_version,
        seat=client.seat,
        request_seq=7,
        message=message,
        response=response,
        latency_ms=12.34,
    )

    lines = (tmp_path / "riichi.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    assert payload["kind"] == "request_action"
    assert payload["response"] == {"type": "dahai", "pai": "3m", "actor": 1}
    assert payload["possible_actions"] == [{"type": "none"}]
    assert payload["possible_action_count"] == 1
    assert payload["request_seq"] == 7
    assert payload["message_meta"]["type"] == "request_action"
    assert payload["state"]["hand"] == ["1m"]
    assert payload["normalized_observation"]["actor"] == 1


def test_audit_logger_logs_send_result(tmp_path: Path) -> None:
    logger = rdc.RiichiDevAuditLogger(tmp_path / "riichi.jsonl")

    logger.log_send_result(
        queue="ranked",
        bot_name="keqingv31",
        model_version=None,
        seat=0,
        request_seq=11,
        response={"type": "dahai", "actor": 0, "pai": "S"},
        success=False,
        error="RuntimeError: socket closed",
    )

    payload = json.loads((tmp_path / "riichi.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert payload["kind"] == "send_result"
    assert payload["request_seq"] == 11
    assert payload["success"] is False
    assert "socket closed" in payload["error"]


def test_audit_logger_logs_protocol_action(tmp_path: Path) -> None:
    logger = rdc.RiichiDevAuditLogger(tmp_path / "riichi.jsonl")

    logger.log_protocol_action(
        queue="ranked",
        bot_name="keqingv3",
        model_version=None,
        seat=0,
        trigger_message={"type": "reach", "actor": 0},
        response={"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
        latency_ms=1.23,
    )

    payload = json.loads((tmp_path / "riichi.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert payload["kind"] == "protocol_action"
    assert payload["trigger_type"] == "reach"
    assert payload["response"]["pai"] == "4m"


def test_observation_agent_reuses_cached_reach_discard(monkeypatch) -> None:
    fake_obs = FakeObservation(player_id=1)

    def fake_decode(message):
        return fake_obs, fake_obs.to_dict()

    class FakeScorer:
        def score(self, ctx):
            from inference.contracts import DecisionResult, ScoredCandidate, ModelAuxOutputs

            return DecisionResult(
                chosen={"type": "reach", "actor": 1},
                candidates=[
                    ScoredCandidate(
                        action={"type": "reach", "actor": 1},
                        logit=1.0,
                        final_score=2.0,
                        beam_score=2.0,
                        meta={"reach_discard": {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": False}},
                    )
                ],
                model_value=0.0,
                model_aux=ModelAuxOutputs(),
            )

    monkeypatch.setattr(rdc, "_decode_observation", fake_decode)
    monkeypatch.setattr(
        rdc.KeqingModelAdapter,
        "from_checkpoint",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    monkeypatch.setattr(rdc, "DefaultActionScorer", lambda **kwargs: FakeScorer())

    agent = rdc.ObservationScoringAgent(model_path=Path("fake.pth"), device="cpu")
    first = agent.select_action(
        {"type": "request_action", "observation": "encoded", "possible_actions": [{"type": "reach", "actor": 1}]},
        seat=1,
    )
    second = agent.select_action({"type": "reach", "actor": 1}, seat=1)

    assert first == {"type": "reach", "actor": 1}
    assert second == {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": False}


def test_client_forwards_reach_event_to_agent_followup() -> None:
    class ReachFollowupAgent(rdc.RiichiDevDecisionAgent):
        def __init__(self):
            self.calls = []

        def select_action(self, message, seat):
            self.calls.append((message, seat))
            if message.get("type") == "reach":
                return {"type": "dahai", "actor": seat, "pai": "4m", "tsumogiri": False}
            return {"type": "reach", "actor": seat}

    agent = ReachFollowupAgent()
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(token="t", model_path=Path("fake.pth")),
        agent=agent,
    )

    client.handle_message({"type": "start_game", "id": 1})
    response = client.handle_message({"type": "reach", "actor": 1})

    assert response == {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False}
    assert agent.calls[-1] == ({"type": "reach", "actor": 1}, 1)


def test_observation_agent_ignores_non_request_reach_without_pending_followup() -> None:
    agent = rdc.ObservationScoringAgent.__new__(rdc.ObservationScoringAgent)
    agent._pending_reach_discard = None

    response = rdc.ObservationScoringAgent.select_action(
        agent,
        {"type": "reach", "actor": 1},
        seat=1,
    )

    assert response == {"type": "none"}


def test_validation_safe_agent_prefers_tsumogiri_when_legal() -> None:
    agent = rdc.ValidationSafeAgent()
    agent._last_tsumo = "5m"
    action = agent.select_action(
        {
            "type": "request_action",
            "possible_actions": [
                {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
                {"type": "none"},
            ],
        },
        seat=0,
    )

    assert action == {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True}


def test_validation_safe_agent_falls_back_to_none() -> None:
    agent = rdc.ValidationSafeAgent()
    action = agent.select_action(
        {"type": "request_action", "possible_actions": [{"type": "none"}]},
        seat=0,
    )

    assert action == {"type": "none"}


def test_disconnect_message_mentions_validate_queue(tmp_path: Path) -> None:
    message = rdc._format_connection_closed_message("validate", 1005, None)

    assert "validate connection closed by server" in message
    assert "validate queue request" in message


def test_log_startup_self_check_rejects_missing_model_path(caplog) -> None:
    try:
        rdc._log_startup_self_check(
            queue="validate",
            bot_name="keqingv31",
            model_version=None,
            token=(
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                "eyJuYW1lIjoibW9jaGEiLCJ0eXBlIjoiYm90IiwiYm90X2lkIjoiYm90LTEyMyJ9."
                "sig"
            ),
            token_source="MOCHAKEY",
            project_root=Path("/tmp/project"),
            model_path=Path("/tmp/project/missing.pth"),
            validation_safe=False,
        )
    except SystemExit as exc:
        assert "model checkpoint not found" in str(exc)
    else:
        raise AssertionError("expected missing model checkpoint to stop startup")


def test_log_startup_self_check_is_silent_without_debug(caplog) -> None:
    rdc._log_startup_self_check(
        queue="validate",
        bot_name="keqingv31",
        model_version=None,
        token=(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJuYW1lIjoibW9jaGEiLCJ0eXBlIjoiYm90IiwiYm90X2lkIjoiYm90LTEyMyJ9."
            "sig"
        ),
        token_source="MOCHAKEY",
        project_root=Path("."),
        model_path=None,
        validation_safe=True,
    )

    assert "startup self-check" not in caplog.text


def test_client_auto_reconnects_after_end_game_1005_and_resets_agent(tmp_path: Path) -> None:
    class ResetAwareAgent(StubAgent):
        def __init__(self):
            super().__init__({"type": "none"})
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeConnectionClosed(Exception):
        def __init__(self, code: int, reason: str | None):
            self.code = code
            self.reason = reason

    class FakeWebSocket:
        def __init__(self, messages):
            self._messages = iter(messages)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._messages)
            except StopIteration:
                raise StopAsyncIteration

        async def send(self, payload):
            return None

    class FakeConnect:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self

        async def __aenter__(self):
            return FakeWebSocket(['{"type":"start_game","id":0}', '{"type":"end_game"}'])

        async def __aexit__(self, exc_type, exc, tb):
            if self.calls == 1:
                raise FakeConnectionClosed(1005, None)
            raise StopAsyncIteration

    agent = ResetAwareAgent()
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(
            token="t",
            queue="ranked",
            model_path=Path("fake.pth"),
            audit_log_path=tmp_path / "riichi.jsonl",
            reconnect_delay_sec=0.0,
        ),
        agent=agent,
    )

    fake_connect = FakeConnect()
    original_connect = rdc.websockets.connect
    original_connection_closed = rdc.ConnectionClosed
    original_sleep = rdc.asyncio.sleep
    rdc.websockets.connect = fake_connect
    rdc.ConnectionClosed = FakeConnectionClosed
    rdc.asyncio.sleep = lambda _: original_sleep(0)
    try:
        try:
            asyncio.run(client.run())
        except StopAsyncIteration:
            pass
    finally:
        rdc.websockets.connect = original_connect
        rdc.ConnectionClosed = original_connection_closed
        rdc.asyncio.sleep = original_sleep

    assert fake_connect.calls == 2
    assert agent.reset_calls == 2


def test_client_auto_reconnects_ranked_queue_wait_1006_before_start_game(tmp_path: Path) -> None:
    class ResetAwareAgent(StubAgent):
        def __init__(self):
            super().__init__({"type": "none"})
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeConnectionClosed(Exception):
        def __init__(self, code: int, reason: str | None):
            self.code = code
            self.reason = reason

    class FakeWebSocket:
        def __init__(self, messages):
            self._messages = iter(messages)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._messages)
            except StopIteration:
                raise StopAsyncIteration

        async def send(self, payload):
            return None

    class FakeConnect:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self

        async def __aenter__(self):
            return FakeWebSocket([])

        async def __aexit__(self, exc_type, exc, tb):
            if self.calls == 1:
                raise FakeConnectionClosed(1006, None)
            raise StopAsyncIteration

    agent = ResetAwareAgent()
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(
            token="t",
            queue="ranked",
            model_path=Path("fake.pth"),
            audit_log_path=tmp_path / "riichi.jsonl",
            reconnect_delay_sec=0.0,
        ),
        agent=agent,
    )

    fake_connect = FakeConnect()
    original_connect = rdc.websockets.connect
    original_connection_closed = rdc.ConnectionClosed
    original_sleep = rdc.asyncio.sleep
    rdc.websockets.connect = fake_connect
    rdc.ConnectionClosed = FakeConnectionClosed
    rdc.asyncio.sleep = lambda _: original_sleep(0)
    try:
        try:
            asyncio.run(client.run())
        except StopAsyncIteration:
            pass
    finally:
        rdc.websockets.connect = original_connect
        rdc.ConnectionClosed = original_connection_closed
        rdc.asyncio.sleep = original_sleep

    assert fake_connect.calls == 2
    assert agent.reset_calls == 2


def test_client_auto_reconnects_ranked_open_timeout_before_start_game(tmp_path: Path) -> None:
    class ResetAwareAgent(StubAgent):
        def __init__(self):
            super().__init__({"type": "none"})
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeConnect:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError
            raise StopAsyncIteration

    agent = ResetAwareAgent()
    client = rdc.RiichiDevBotClient(
        rdc.RiichiDevClientConfig(
            token="t",
            queue="ranked",
            model_path=Path("fake.pth"),
            audit_log_path=tmp_path / "riichi.jsonl",
            reconnect_delay_sec=0.0,
        ),
        agent=agent,
    )

    fake_connect = FakeConnect()
    original_connect = rdc.websockets.connect
    original_sleep = rdc.asyncio.sleep
    rdc.websockets.connect = fake_connect
    rdc.asyncio.sleep = lambda _: original_sleep(0)
    try:
        try:
            asyncio.run(client.run())
        except StopAsyncIteration:
            pass
    finally:
        rdc.websockets.connect = original_connect
        rdc.asyncio.sleep = original_sleep

    assert fake_connect.calls == 2
    assert agent.reset_calls == 2
