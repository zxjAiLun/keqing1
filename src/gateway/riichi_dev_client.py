from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed
from riichienv import Observation, Observation3P, RiichiEnv

from inference.contracts import DecisionContext, DecisionResult
from inference.keqing_adapter import KeqingModelAdapter
from inference.scoring import DefaultActionScorer
from keqingv1.action_space import IDX_TO_TILE_NAME
from mahjong.tile import FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_BASE_URL = "wss://game.riichi.dev"
DEFAULT_VALIDATE_PATH = "/ws/validate"
DEFAULT_RANKED_PATH = "/ws/ranked"
_ROUND_WINDS = ("E", "S", "W", "N")
_MELD_TYPE_MAP = {
    "Chi": "chi",
    "Pon": "pon",
    "Kan": "daiminkan",
    "ClosedKan": "ankan",
    "AddedKan": "kakan",
}


def _resolve_ws_url(base_url: str, queue: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.startswith("https://"):
        normalized = "wss://" + normalized.removeprefix("https://")
    elif normalized.startswith("http://"):
        normalized = "ws://" + normalized.removeprefix("http://")
    if queue == "validate":
        return f"{normalized}{DEFAULT_VALIDATE_PATH}"
    if queue == "ranked":
        return f"{normalized}{DEFAULT_RANKED_PATH}"
    raise ValueError(f"unsupported queue: {queue}")


def _resolve_default_token(bot_name: str) -> str:
    if bot_name == "keqingv31":
        return os.getenv("MOCHAKEY", "")
    return os.getenv("LATTEKEY", os.getenv("RIICHI_BOT_TOKEN", ""))


def _resolve_default_token_with_source(bot_name: str) -> tuple[str, str]:
    if bot_name == "keqingv31":
        return os.getenv("MOCHAKEY", ""), "MOCHAKEY"
    if os.getenv("LATTEKEY"):
        return os.getenv("LATTEKEY", ""), "LATTEKEY"
    return os.getenv("RIICHI_BOT_TOKEN", ""), "RIICHI_BOT_TOKEN"


def _format_connection_closed_message(queue: str, code: int, reason: str | None) -> str:
    return (
        f"{queue} connection closed by server "
        f"(code={code}, reason={reason or 'none'}). "
        "Likely causes: bot is not active yet, another session is using the same bot, "
        f"or the server rejected the {queue} queue request."
    )


def _resolve_model_path(
    *,
    bot_name: str,
    project_root: str | Path,
    model_path: str | Path | None,
) -> Path | None:
    if bot_name == "rulebase":
        return None
    if model_path is not None:
        return Path(model_path)
    return Path(project_root) / "artifacts" / "models" / bot_name / "best.pth"


def _decode_jwt_payload_unverified(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) < 2 or not parts[1]:
        return None
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        decoded = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def _log_startup_self_check(
    *,
    queue: str,
    bot_name: str,
    model_version: str | None,
    token: str,
    token_source: str,
    project_root: Path,
    model_path: Path | None,
    validation_safe: bool,
) -> None:
    should_log = logger.isEnabledFor(logging.DEBUG)
    token_payload = _decode_jwt_payload_unverified(token)
    token_name = token_payload.get("name") if token_payload else None
    token_bot_id = token_payload.get("bot_id") if token_payload else None
    token_type = token_payload.get("type") if token_payload else None
    token_summary = (
        f"name={token_name or 'unknown'} "
        f"type={token_type or 'unknown'} "
        f"bot_id={token_bot_id or 'unknown'}"
    )
    if should_log:
        logger.info(
            "startup self-check: queue=%s bot_name=%s model_version=%s token_source=%s token_identity=%s",
            queue,
            bot_name,
            model_version or bot_name,
            token_source,
            token_summary,
        )
        logger.info("startup self-check: project_root=%s", project_root)
    if validation_safe:
        if should_log:
            logger.info("startup self-check: validation_safe=true, model checkpoint skipped")
        return
    if model_path is None:
        if should_log:
            logger.info("startup self-check: model checkpoint skipped for bot_name=%s", bot_name)
        return
    if should_log:
        logger.info(
            "startup self-check: model_path=%s exists=%s",
            model_path,
            model_path.exists(),
        )
    if not model_path.exists():
        raise SystemExit(f"model checkpoint not found: {model_path}")


def _decode_observation(message: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    encoded = message.get("observation")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("request_action message missing observation")

    player_count = message.get("player_count")
    if player_count == 3:
        obs = Observation3P.deserialize_from_base64(encoded)
    elif player_count == 4:
        obs = Observation.deserialize_from_base64(encoded)
    else:
        try:
            obs = Observation.deserialize_from_base64(encoded)
        except Exception:
            obs = Observation3P.deserialize_from_base64(encoded)
    return obs, _normalize_observation_state(obs, obs.to_dict())


def _hash_observation_payload(encoded: str | None) -> str | None:
    if not encoded:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audit_message_meta(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": message.get("type"),
        "seat": message.get("seat"),
        "player_count": message.get("player_count"),
        "has_observation": isinstance(message.get("observation"), str) and bool(message.get("observation")),
        "observation_sha256": message.get("_observation_sha256"),
        "decode_error": message.get("_observation_decode_error"),
    }


def _enrich_message_for_audit(message: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(message)
    encoded = enriched.get("observation")
    if isinstance(encoded, str) and encoded:
        enriched["_observation_sha256"] = _hash_observation_payload(encoded)
        if "_normalized_observation" not in enriched:
            try:
                _obs, normalized = _decode_observation(enriched)
                enriched["_normalized_observation"] = normalized
            except Exception as exc:
                enriched["_observation_decode_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


def _sanitize_action(action: dict[str, Any], actor_hint: int | None) -> dict[str, Any]:
    action_type = action.get("type", "none")
    if action_type == "none":
        return {"type": "none"}
    if action_type == "ryukyoku":
        return {"type": "ryukyoku"}

    if action_type == "hora":
        actor = action.get("actor", actor_hint)
        target = action.get("target", actor)
        out = {"type": "hora", "actor": actor, "target": target}
        if target != actor and action.get("pai") is not None:
            out["pai"] = action["pai"]
        return out
    if action_type == "reach":
        actor = action.get("actor", actor_hint)
        return {"type": "reach", "actor": actor}
    if action_type == "dahai":
        actor = action.get("actor", actor_hint)
        out = {"type": "dahai", "actor": actor, "pai": action["pai"]}
        if "tsumogiri" in action:
            out["tsumogiri"] = bool(action["tsumogiri"])
        return out
    if action_type == "chi":
        actor = action.get("actor", actor_hint)
        out = {
            "type": "chi",
            "actor": actor,
            "pai": action["pai"],
            "consumed": list(action["consumed"]),
        }
        if "target" in action:
            out["target"] = action["target"]
        return out
    if action_type == "pon":
        actor = action.get("actor", actor_hint)
        out = {
            "type": "pon",
            "actor": actor,
            "pai": action["pai"],
            "consumed": list(action["consumed"]),
        }
        if "target" in action:
            out["target"] = action["target"]
        return out
    if action_type == "daiminkan":
        actor = action.get("actor", actor_hint)
        out = {
            "type": "daiminkan",
            "actor": actor,
            "pai": action["pai"],
            "consumed": list(action["consumed"]),
        }
        if "target" in action:
            out["target"] = action["target"]
        return out
    if action_type == "ankan":
        actor = action.get("actor", actor_hint)
        return {
            "type": "ankan",
            "actor": actor,
            "consumed": list(action["consumed"]),
        }
    if action_type == "kakan":
        actor = action.get("actor", actor_hint)
        return {
            "type": "kakan",
            "actor": actor,
            "pai": action["pai"],
            "consumed": list(action["consumed"]),
        }

    out = dict(action)
    if "actor" not in out and actor_hint is not None:
        out["actor"] = actor_hint
    return out


def _action_to_mjai_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return dict(action)
    if hasattr(action, "to_mjai"):
        mjai = action.to_mjai()
        if isinstance(mjai, str):
            return json.loads(mjai)
        if isinstance(mjai, dict):
            return dict(mjai)
    if isinstance(action, str):
        return json.loads(action)
    raise TypeError(f"unsupported action payload type: {type(action)!r}")

def _tile136_to_str(tile136: int) -> str:
    if tile136 == FIVE_RED_MAN:
        return "5mr"
    if tile136 == FIVE_RED_PIN:
        return "5pr"
    if tile136 == FIVE_RED_SOU:
        return "5sr"
    return IDX_TO_TILE_NAME[int(tile136) // 4]


def _normalize_observation_state(obs: Any, raw_state: dict[str, Any]) -> dict[str, Any]:
    actor = int(getattr(obs, "player_id", raw_state.get("player_id", 0)))
    raw_melds = raw_state.get("melds") or [[], [], [], []]
    melds: list[list[dict[str, Any]]] = []
    for seat, seat_melds in enumerate(raw_melds):
        normalized_seat: list[dict[str, Any]] = []
        for meld in seat_melds:
            meld_type_name = getattr(getattr(meld, "meld_type", None), "name", None)
            tiles = [_tile136_to_str(int(tile)) for tile in list(meld.tiles)]
            called_tile = getattr(meld, "called_tile", None)
            pai = (
                _tile136_to_str(int(called_tile))
                if called_tile is not None
                else (tiles[0] if tiles else None)
            )
            consumed = list(tiles)
            if called_tile is not None and pai in consumed:
                consumed.remove(pai)
            normalized_seat.append(
                {
                    "type": _MELD_TYPE_MAP.get(meld_type_name, "pon"),
                    "pai": pai,
                    "consumed": consumed,
                    "target": int(meld.from_who) if getattr(meld, "opened", False) else None,
                }
            )
        melds.append(normalized_seat)

    discards = [
        [_tile136_to_str(int(tile)) for tile in seat_discards]
        for seat_discards in (raw_state.get("discards") or [[], [], [], []])
    ]
    hands = raw_state.get("hands") or [[], [], [], []]
    dora_indicators = raw_state.get("dora_indicators") or []
    riichi_declared = raw_state.get("riichi_declared") or [False, False, False, False]
    scores = raw_state.get("scores") or [25000, 25000, 25000, 25000]

    return {
        "actor": actor,
        "hand": [_tile136_to_str(int(tile)) for tile in hands[actor]],
        "discards": discards,
        "melds": melds,
        "scores": list(scores),
        "dora_markers": [_tile136_to_str(int(tile)) for tile in dora_indicators],
        "reached": list(riichi_declared),
        "bakaze": _ROUND_WINDS[int(raw_state.get("round_wind", 0)) % 4],
        "kyoku": int(raw_state.get("kyoku", raw_state.get("kyoku_index", 1) or 1)),
        "honba": int(raw_state.get("honba", 0)),
        "kyotaku": int(raw_state.get("riichi_sticks", 0)),
        "oya": int(raw_state.get("oya", 0)),
    }


class RiichiDevDecisionAgent:
    def reset(self) -> None:
        return None

    def act(self, obs: Any):
        raise NotImplementedError

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any]:
        raise NotImplementedError


class RiichiDevAuditLogger:
    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _state_summary(message: dict[str, Any]) -> dict[str, Any] | None:
        observation = message.get("_normalized_observation")
        if not isinstance(observation, dict):
            return None
        actor = observation.get("actor", 0)
        melds = observation.get("melds") or [[], [], [], []]
        discards = observation.get("discards") or [[], [], [], []]
        reached = observation.get("reached") or [False, False, False, False]
        return {
            "actor": actor,
            "bakaze": observation.get("bakaze"),
            "kyoku": observation.get("kyoku"),
            "honba": observation.get("honba"),
            "kyotaku": observation.get("kyotaku"),
            "oya": observation.get("oya"),
            "hand": observation.get("hand"),
            "melds": melds[actor] if actor < len(melds) else [],
            "discards": discards[actor] if actor < len(discards) else [],
            "reached": reached[actor] if actor < len(reached) else False,
            "dora_markers": observation.get("dora_markers", []),
            "scores": observation.get("scores", []),
        }

    def write(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_request_action(
        self,
        *,
        queue: str,
        bot_name: str,
        model_version: str | None,
        seat: int | None,
        request_seq: int | None,
        message: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
    ) -> None:
        possible_actions = message.get("possible_actions") or []
        self.write(
            {
                "kind": "request_action",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "request_seq": request_seq,
                "message_type": message.get("type"),
                "message_meta": _audit_message_meta(message),
                "response": response,
                "latency_ms": round(latency_ms, 3),
                "possible_action_count": len(possible_actions),
                "possible_actions": possible_actions,
                "state": self._state_summary(message),
                "normalized_observation": message.get("_normalized_observation"),
            }
        )

    def log_send_result(
        self,
        *,
        queue: str,
        bot_name: str,
        model_version: str | None,
        seat: int | None,
        request_seq: int | None,
        response: dict[str, Any],
        success: bool,
        error: str | None = None,
    ) -> None:
        self.write(
            {
                "kind": "send_result",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "request_seq": request_seq,
                "success": success,
                "response": response,
                "error": error,
            }
        )

    def log_protocol_action(
        self,
        *,
        queue: str,
        bot_name: str,
        model_version: str | None,
        seat: int | None,
        trigger_message: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
    ) -> None:
        self.write(
            {
                "kind": "protocol_action",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "trigger_type": trigger_message.get("type"),
                "message_meta": _audit_message_meta(trigger_message),
                "response": response,
                "latency_ms": round(latency_ms, 3),
                "state": self._state_summary(trigger_message),
                "normalized_observation": trigger_message.get("_normalized_observation"),
            }
        )

    def log_agent_error(
        self,
        *,
        queue: str,
        bot_name: str,
        model_version: str | None,
        seat: int | None,
        request_seq: int | None,
        message: dict[str, Any],
        error: str,
        traceback_text: str,
    ) -> None:
        self.write(
            {
                "kind": "agent_error",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "request_seq": request_seq,
                "message_meta": _audit_message_meta(message),
                "possible_actions": message.get("possible_actions") or [],
                "state": self._state_summary(message),
                "normalized_observation": message.get("_normalized_observation"),
                "error": error,
                "traceback": traceback_text,
            }
        )

    def log_event(self, *, queue: str, bot_name: str, model_version: str | None, seat: int | None, message: dict[str, Any]) -> None:
        self.write(
            {
                "kind": "event",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "message": message,
            }
        )

    def log_disconnect(self, *, queue: str, bot_name: str, model_version: str | None, seat: int | None, code: int, reason: str) -> None:
        self.write(
            {
                "kind": "disconnect",
                "queue": queue,
                "bot_name": bot_name,
                "model_version": model_version,
                "seat": seat,
                "code": code,
                "reason": reason,
            }
        )


@dataclass(frozen=True)
class DecisionAgentSpec:
    model_version: str
    hidden_dim: int = 256
    num_res_blocks: int = 4
    beam_k: int = 3
    beam_lambda: float = 1.0
    score_delta_lambda: float = 0.20
    win_prob_lambda: float = 0.20
    dealin_prob_lambda: float = 0.25


DEFAULT_DECISION_AGENT_SPEC = DecisionAgentSpec(model_version="keqingv3")
DECISION_AGENT_SPECS: dict[str, DecisionAgentSpec] = {
    "keqingv1": DecisionAgentSpec(model_version="keqingv1"),
    "keqingv2": DecisionAgentSpec(model_version="keqingv2"),
    "keqingv3": DecisionAgentSpec(model_version="keqingv3"),
    "keqingv31": DecisionAgentSpec(model_version="keqingv31", hidden_dim=320, num_res_blocks=6),
    "keqingv4": DecisionAgentSpec(model_version="keqingv4", hidden_dim=320, num_res_blocks=6),
    "xmodel1": DecisionAgentSpec(
        model_version="xmodel1",
        beam_k=0,
        beam_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    ),
}


class ValidationSafeAgent(RiichiDevDecisionAgent):
    def __init__(self) -> None:
        self._seat: int | None = None
        self._last_tsumo: str | None = None

    def reset(self) -> None:
        self._seat = None
        self._last_tsumo = None

    def act(self, obs: Any):
        legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
        actor = int(getattr(obs, "player_id", 0))
        if self._last_tsumo:
            candidate = {
                "type": "dahai",
                "actor": actor,
                "pai": self._last_tsumo,
                "tsumogiri": True,
            }
            if any(_action_to_mjai_dict(a) == candidate for a in legal_actions):
                return obs.select_action_from_mjai(json.dumps(candidate, ensure_ascii=False))
        return obs.select_action_from_mjai(json.dumps({"type": "none"}, ensure_ascii=False))

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any]:
        mtype = message.get("type")
        if mtype == "start_game":
            if seat is not None:
                self._seat = int(seat)
            return {"type": "none"}
        if mtype == "tsumo":
            pai = message.get("pai")
            actor = message.get("actor")
            if pai and pai != "?" and actor == seat:
                self._last_tsumo = pai
            return {"type": "none"}
        if mtype != "request_action":
            return {"type": "none"}

        actor = seat if seat is not None else self._seat
        legal_actions = [_action_to_mjai_dict(a) for a in (message.get("possible_actions") or [])]
        if self._last_tsumo and actor is not None:
            candidate = {
                "type": "dahai",
                "actor": actor,
                "pai": self._last_tsumo,
                "tsumogiri": True,
            }
            if any(a == candidate for a in legal_actions):
                self._last_tsumo = None
                return candidate
        self._last_tsumo = None
        return {"type": "none"}


class ObservationScoringAgent(RiichiDevDecisionAgent):
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cuda",
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        beam_k: int = 3,
        beam_lambda: float = 1.0,
        score_delta_lambda: float = 0.20,
        win_prob_lambda: float = 0.20,
        dealin_prob_lambda: float = 0.25,
        model_version: str | None = "keqingv3",
    ):
        resolved_device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self._adapter = KeqingModelAdapter.from_checkpoint(
            model_path,
            device=resolved_device,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            model_version=model_version,
        )
        self._scorer = DefaultActionScorer(
            adapter=self._adapter,
            beam_k=beam_k,
            beam_lambda=beam_lambda,
            style_lambda=0.0,
            score_delta_lambda=score_delta_lambda,
            win_prob_lambda=win_prob_lambda,
            dealin_prob_lambda=dealin_prob_lambda,
        )
        self._pending_reach_discard: dict[str, Any] | None = None

    def reset(self) -> None:
        self._pending_reach_discard = None

    def _remember_reach_followup(self, decision: DecisionResult, actor: int) -> None:
        self._pending_reach_discard = None
        if decision.chosen.get("type") != "reach":
            return
        for candidate in decision.candidates:
            if candidate.action.get("type") != "reach":
                continue
            reach_discard = candidate.meta.get("reach_discard")
            if not reach_discard:
                continue
            self._pending_reach_discard = _sanitize_action(reach_discard, actor)
            return

    def choose_mjai_action(
        self,
        *,
        obs: Any,
        snap: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actor = int(getattr(obs, "player_id", snap.get("actor", 0)))
        ctx = DecisionContext(
            actor=actor,
            event=event or {"type": "request_action"},
            runtime_snap=snap,
            model_snap=snap,
            legal_actions=legal_actions,
        )
        decision = self._scorer.score(ctx)
        self._remember_reach_followup(decision, actor)
        return _sanitize_action(decision.chosen, actor)

    def act(self, obs: Any):
        legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
        snap = _normalize_observation_state(obs, obs.to_dict())
        chosen = self.choose_mjai_action(
            obs=obs,
            snap=snap,
            legal_actions=legal_actions,
        )
        return obs.select_action_from_mjai(json.dumps(chosen, ensure_ascii=False))

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any]:
        mtype = message.get("type")
        if mtype != "request_action":
            if (
                mtype == "reach"
                and seat is not None
                and message.get("actor") == seat
                and self._pending_reach_discard is not None
            ):
                discard = self._pending_reach_discard
                self._pending_reach_discard = None
                return discard
            return {"type": "none"}
        obs, snap = _decode_observation(message)
        legal_actions = list(message.get("possible_actions") or [])
        if not legal_actions:
            legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
        else:
            legal_actions = [_action_to_mjai_dict(action) for action in legal_actions]
        return self.choose_mjai_action(
            obs=obs,
            snap=snap,
            legal_actions=legal_actions,
            event=message,
        )


def create_riichi_dev_agent(
    *,
    bot_name: str,
    project_root: str | Path,
    model_path: str | Path | None,
    device: str,
    verbose: bool,
    model_version: str | None = None,
    validation_safe: bool = False,
) -> RiichiDevDecisionAgent:
    if validation_safe:
        return ValidationSafeAgent()
    if bot_name == "rulebase":
        raise ValueError("rulebase is not supported by the riichi.dev observation gateway")
    spec = DECISION_AGENT_SPECS.get(model_version or bot_name)
    if spec is None:
        inferred_version = model_version or bot_name
        spec = DecisionAgentSpec(model_version=inferred_version)
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(project_root) / "artifacts" / "models" / bot_name / "best.pth"
    )
    return ObservationScoringAgent(
        model_path=resolved_model_path,
        device=device,
        hidden_dim=spec.hidden_dim,
        num_res_blocks=spec.num_res_blocks,
        beam_k=spec.beam_k,
        beam_lambda=spec.beam_lambda,
        score_delta_lambda=spec.score_delta_lambda,
        win_prob_lambda=spec.win_prob_lambda,
        dealin_prob_lambda=spec.dealin_prob_lambda,
        model_version=spec.model_version,
    )


@dataclass(slots=True)
class RiichiDevClientConfig:
    token: str
    bot_name: str = "keqingv3"
    model_version: str | None = None
    queue: str = "ranked"
    base_url: str = DEFAULT_BASE_URL
    ws_url_override: str | None = None
    model_path: Path | None = None
    project_root: Path = Path.cwd()
    device: str = "cuda"
    verbose: bool = False
    origin: str | None = None
    user_agent: str = "keqing1-riichi-dev-client/0.1"
    audit_log_path: Path | None = None
    open_timeout: float = 10.0
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    auto_reconnect: bool = True
    reconnect_delay_sec: float = 1.0

    def ws_url(self) -> str:
        if self.ws_url_override:
            return self.ws_url_override
        return _resolve_ws_url(self.base_url, self.queue)


class RiichiDevBotClient:
    def __init__(
        self,
        config: RiichiDevClientConfig,
        *,
        agent: RiichiDevDecisionAgent | None = None,
    ):
        self.config = config
        self.agent = agent or create_riichi_dev_agent(
            bot_name=config.bot_name,
            project_root=config.project_root,
            model_path=config.model_path,
            device=config.device,
            verbose=config.verbose,
            model_version=config.model_version,
            validation_safe=False,
        )
        self.seat: int | None = None
        self._saw_start_game = False
        self._saw_end_game = False
        self._request_seq = 0
        default_audit_path = Path("logs/riichi_dev") / f"{self.config.queue}-{self.config.bot_name}.jsonl"
        self.audit_logger = RiichiDevAuditLogger(self.config.audit_log_path or default_audit_path)

    def _reset_session_state(self) -> None:
        self.seat = None
        self._saw_start_game = False
        self._saw_end_game = False
        self._request_seq = 0
        self.agent.reset()

    async def _run_once(self) -> None:
        self._reset_session_state()
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "User-Agent": self.config.user_agent,
        }
        logger.info("connecting to %s", self.config.ws_url())
        async with websockets.connect(
            self.config.ws_url(),
            extra_headers=headers,
            origin=self.config.origin,
            open_timeout=self.config.open_timeout,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
        ) as websocket:
            async for raw_message in websocket:
                message = _enrich_message_for_audit(json.loads(raw_message))
                request_seq: int | None = None
                if message.get("type") == "request_action":
                    self._request_seq += 1
                    request_seq = self._request_seq
                t0 = time.perf_counter()
                try:
                    response = self.handle_message(message)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    self.audit_logger.log_agent_error(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        request_seq=request_seq,
                        message=message,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                    if self.config.verbose:
                        logger.exception(
                            "agent decision failed for request_seq=%s latency_ms=%.1f",
                            request_seq,
                            latency_ms,
                        )
                    raise
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if message.get("type") == "request_action" and response is not None:
                    self.audit_logger.log_request_action(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        request_seq=request_seq,
                        message=message,
                        response=response,
                        latency_ms=latency_ms,
                    )
                elif response is not None:
                    self.audit_logger.log_protocol_action(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        trigger_message=message,
                        response=response,
                        latency_ms=latency_ms,
                    )
                elif message.get("type") in {"start_game", "start_kyoku", "end_kyoku", "end_game", "validation_result", "error"}:
                    self.audit_logger.log_event(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        message=message,
                    )
                if self.config.verbose:
                    logger.info("recv: %s", message.get("type"))
                    if response is not None:
                        logger.info("send: %s latency_ms=%.1f", response, latency_ms)
                if response is not None:
                    try:
                        await websocket.send(json.dumps(response, ensure_ascii=False))
                    except Exception as exc:
                        self.audit_logger.log_send_result(
                            queue=self.config.queue,
                            bot_name=self.config.bot_name,
                            model_version=self.config.model_version,
                            seat=self.seat,
                            request_seq=request_seq,
                            response=response,
                            success=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    else:
                        self.audit_logger.log_send_result(
                            queue=self.config.queue,
                            bot_name=self.config.bot_name,
                            model_version=self.config.model_version,
                            seat=self.seat,
                            request_seq=request_seq,
                            response=response,
                            success=True,
                        )

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        mtype = message.get("type")
        if mtype == "start_game":
            self._saw_start_game = True
            seat = message.get("id", message.get("seat"))
            if seat is not None:
                self.seat = int(seat)
            return None
        if mtype == "end_game":
            self._saw_end_game = True
            return None
        if mtype == "reach":
            return self.agent.select_action(message, self.seat)
        if mtype != "request_action":
            return None
        return self.agent.select_action(message, self.seat)

    def _should_retry_disconnect(self, code: int) -> bool:
        retryable_disconnect = code in {1005, 1006}
        retry_after_game = self._saw_end_game and retryable_disconnect
        retry_ranked_queue_wait = (
            self.config.queue == "ranked"
            and not self._saw_start_game
            and retryable_disconnect
        )
        return self.config.auto_reconnect and (retry_after_game or retry_ranked_queue_wait)

    async def run(self) -> None:
        while True:
            try:
                await self._run_once()
            except ConnectionClosed as exc:
                self.audit_logger.log_disconnect(
                    queue=self.config.queue,
                    bot_name=self.config.bot_name,
                    model_version=self.config.model_version,
                    seat=self.seat,
                    code=exc.code,
                    reason=exc.reason or "",
                )
                if self._should_retry_disconnect(exc.code):
                    if self.config.verbose:
                        logger.info(
                            "server closed %s connection (code=%s); reconnecting in %.1fs",
                            self.config.queue,
                            exc.code,
                            self.config.reconnect_delay_sec,
                        )
                    await asyncio.sleep(self.config.reconnect_delay_sec)
                    continue
                raise SystemExit(
                    _format_connection_closed_message(self.config.queue, exc.code, exc.reason)
                ) from None
            except TimeoutError:
                should_retry_timeout = (
                    self.config.auto_reconnect
                    and self.config.queue == "ranked"
                    and not self._saw_start_game
                )
                if should_retry_timeout:
                    if self.config.verbose:
                        logger.info(
                            "opening %s connection timed out; reconnecting in %.1fs",
                            self.config.queue,
                            self.config.reconnect_delay_sec,
                        )
                    await asyncio.sleep(self.config.reconnect_delay_sec)
                    continue
                raise SystemExit(
                    f"timed out while opening {self.config.queue} connection"
                ) from None
            if not self.config.auto_reconnect or not self._saw_end_game:
                return
            if self.config.verbose:
                logger.info(
                    "%s session ended cleanly after end_game; reconnecting in %.1fs",
                    self.config.queue,
                    self.config.reconnect_delay_sec,
                )
            await asyncio.sleep(self.config.reconnect_delay_sec)


def run_local_game(
    *,
    agent: RiichiDevDecisionAgent,
    game_mode: int = 2,
    seed: int | None = 42,
    max_steps: int = 10000,
) -> dict[str, Any]:
    env = RiichiEnv(game_mode=game_mode, seed=seed)
    observations = env.get_observations()
    step_count = 0

    while not env.done():
        acted = False
        for pid, obs in observations.items():
            actions = obs.legal_actions()
            if not actions:
                continue
            action = agent.act(obs)
            observations = env.step({pid: action})
            step_count += 1
            acted = True
            if step_count >= max_steps:
                raise RuntimeError(f"local game exceeded max_steps={max_steps}")
            break
        if not acted:
            raise RuntimeError("local game stalled without any legal action")

    return {
        "scores": list(env.scores()),
        "ranks": list(env.ranks()),
        "step_count": step_count,
        "mjai_log_length": len(env.mjai_log),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supported keqing bots on riichi.dev")
    parser.add_argument(
        "--bot-name",
        default="keqingv3",
        help="bot family / checkpoint namespace to run (e.g. keqingv3, keqingv31, xmodel1)",
    )
    parser.add_argument(
        "--model-version",
        default="",
        help="optional adapter/model-version override; defaults to --bot-name",
    )
    parser.add_argument(
        "--token",
        default="",
        help="riichi.dev bot token; if omitted, resolve from env based on --bot-name",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "local"),
        default="online",
        help="run against riichi.dev over WebSocket or locally with RiichiEnv",
    )
    parser.add_argument(
        "--queue",
        choices=("validate", "ranked"),
        default="ranked",
        help="target riichi.dev queue (default: ranked)",
    )
    parser.add_argument(
        "--validation-safe",
        action="store_true",
        help="use a minimal tsumogiri/pass agent for bot activation robustness",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RIICHI_BASE_URL", DEFAULT_BASE_URL),
        help="WebSocket base URL, e.g. wss://game.riichi.dev",
    )
    parser.add_argument(
        "--ws-url",
        default=os.getenv("RIICHI_WS_URL", ""),
        help="full WebSocket URL override; takes precedence over --base-url/--queue",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="checkpoint path override for model-backed bots",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="project root for resolving default model checkpoints",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("RIICHI_BOT_DEVICE", "cuda"),
        help="inference device preference",
    )
    parser.add_argument(
        "--origin",
        default=os.getenv("RIICHI_BOT_ORIGIN", ""),
        help="optional Origin header for the WebSocket handshake",
    )
    parser.add_argument(
        "--user-agent",
        default=os.getenv("RIICHI_BOT_USER_AGENT", "keqing1-riichi-dev-client/0.1"),
        help="User-Agent header for the WebSocket handshake",
    )
    parser.add_argument(
        "--audit-log-path",
        default="",
        help="optional local jsonl audit log path; default logs/riichi_dev/<queue>-<bot>.jsonl",
    )
    parser.add_argument("--game-mode", type=int, default=2, help="local RiichiEnv game mode")
    parser.add_argument("--seed", type=int, default=42, help="local RiichiEnv seed")
    parser.add_argument("--max-steps", type=int, default=10000, help="local test safety cap")
    parser.add_argument(
        "--no-auto-reconnect",
        action="store_true",
        help="exit after the current online session instead of reconnecting for the next game",
    )
    parser.add_argument(
        "--reconnect-delay-sec",
        type=float,
        default=1.0,
        help="delay before reconnecting after an online game ends or the server closes with code 1005",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root)
    explicit_model_path = Path(args.model_path) if args.model_path else None

    if args.mode == "local":
        agent = create_riichi_dev_agent(
            bot_name=args.bot_name,
            project_root=project_root,
            model_path=explicit_model_path,
            device=args.device,
            verbose=args.verbose,
            model_version=args.model_version or None,
            validation_safe=args.validation_safe,
        )
        result = run_local_game(
            agent=agent,
            game_mode=args.game_mode,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    token_source = "cli"
    if not args.token:
        args.token, token_source = _resolve_default_token_with_source(args.bot_name)

    if not args.token:
        raise SystemExit("missing bot token; pass --token or set MOCHAKEY/LATTEKEY in project .env")

    resolved_model_path = _resolve_model_path(
        bot_name=args.bot_name,
        project_root=project_root,
        model_path=explicit_model_path,
    )
    _log_startup_self_check(
        queue=args.queue,
        bot_name=args.bot_name,
        model_version=args.model_version or None,
        token=args.token,
        token_source=token_source,
        project_root=project_root,
        model_path=resolved_model_path,
        validation_safe=args.validation_safe,
    )

    agent = create_riichi_dev_agent(
        bot_name=args.bot_name,
        project_root=project_root,
        model_path=explicit_model_path,
        device=args.device,
        verbose=args.verbose,
        model_version=args.model_version or None,
        validation_safe=args.validation_safe,
    )

    config = RiichiDevClientConfig(
        token=args.token,
        bot_name=args.bot_name,
        model_version=args.model_version or None,
        queue=args.queue,
        base_url=args.base_url,
        ws_url_override=args.ws_url or None,
        model_path=explicit_model_path,
        project_root=project_root,
        device=args.device,
        verbose=args.verbose,
        origin=args.origin or None,
        user_agent=args.user_agent,
        audit_log_path=Path(args.audit_log_path) if args.audit_log_path else None,
        auto_reconnect=not args.no_auto_reconnect,
        reconnect_delay_sec=args.reconnect_delay_sec,
    )
    client = RiichiDevBotClient(config, agent=agent)
    await client.run()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    asyncio.run(_async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
