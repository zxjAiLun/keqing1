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
from typing import Any, Mapping, Sequence

import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed
from riichienv import Observation, Observation3P, RiichiEnv

from mahjong_env.action_space import IDX_TO_TILE_NAME
from mahjong.tile import FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU

torch = None
DecisionContext = None
DecisionResult = None


class _KeqingModelAdapterPlaceholder:
    @classmethod
    def from_checkpoint(cls, *args, **kwargs):
        raise RuntimeError("KeqingModelAdapter is not loaded")


KeqingModelAdapter = _KeqingModelAdapterPlaceholder
DefaultActionScorer = None

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


def _ensure_model_runtime_imports() -> None:
    global torch, DecisionContext, DecisionResult, KeqingModelAdapter, DefaultActionScorer
    if torch is None:
        import torch as _torch

        torch = _torch
    if DecisionContext is None or DecisionResult is None:
        from inference.contracts import (
            DecisionContext as _DecisionContext,
            DecisionResult as _DecisionResult,
        )

        if DecisionContext is None:
            DecisionContext = _DecisionContext
        if DecisionResult is None:
            DecisionResult = _DecisionResult
    placeholder_from_checkpoint = getattr(
        _KeqingModelAdapterPlaceholder.from_checkpoint,
        "__func__",
        _KeqingModelAdapterPlaceholder.from_checkpoint,
    )
    current_from_checkpoint = getattr(
        KeqingModelAdapter.from_checkpoint,
        "__func__",
        KeqingModelAdapter.from_checkpoint,
    )
    if (
        KeqingModelAdapter is _KeqingModelAdapterPlaceholder
        and current_from_checkpoint is placeholder_from_checkpoint
    ):
        from inference.keqing_adapter import KeqingModelAdapter as _KeqingModelAdapter

        KeqingModelAdapter = _KeqingModelAdapter
    if DefaultActionScorer is None:
        from inference.scoring import DefaultActionScorer as _DefaultActionScorer

        DefaultActionScorer = _DefaultActionScorer


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
    return os.getenv("LATTEKEY", os.getenv("RIICHI_BOT_TOKEN", ""))


def _resolve_default_token_with_source(bot_name: str) -> tuple[str, str]:
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
    if bot_name == "mortal":
        return Path(project_root) / "artifacts" / "mortal_serving" / "mortal.pth"
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
        actor = action.get("actor", actor_hint)
        out = {"type": "none"}
        if actor is not None:
            out["actor"] = actor
        return out
    if action_type == "ryukyoku":
        return {"type": "ryukyoku"}

    if action_type == "hora":
        actor = action.get("actor", actor_hint)
        out = {"type": "hora", "actor": actor}
        target = action.get("target")
        if target is not None:
            out["target"] = target
        if target is not None and target != actor and action.get("pai") is not None:
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


def _last_self_tsumo_pai(new_events: list[dict[str, Any]] | None, actor: int | None) -> str | None:
    for event in reversed(new_events or []):
        if event.get("type") == "tsumo" and event.get("actor") == actor:
            pai = event.get("pai")
            if isinstance(pai, str) and pai != "?":
                return pai
    return None


def _complete_wire_action(
    action: Any,
    *,
    actor_hint: int | None = None,
    new_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = _sanitize_action(_action_to_mjai_dict(action), actor_hint)
    if out.get("type") == "none":
        return {"type": "none"}
    if out.get("type") == "dahai" and "tsumogiri" not in out:
        last_tsumo = _last_self_tsumo_pai(new_events, out.get("actor", actor_hint))
        out["tsumogiri"] = bool(last_tsumo is not None and out.get("pai") == last_tsumo)
    return out


def _action_to_mjai_wire_payload(
    action: Any,
    *,
    actor_hint: int | None = None,
    new_events: list[dict[str, Any]] | None = None,
) -> str:
    if isinstance(action, str):
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError:
            return action
        return json.dumps(
            _complete_wire_action(parsed, actor_hint=actor_hint, new_events=new_events),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if hasattr(action, "to_mjai"):
        mjai = action.to_mjai()
        parsed = json.loads(mjai) if isinstance(mjai, str) else dict(mjai)
        return json.dumps(
            _complete_wire_action(parsed, actor_hint=actor_hint, new_events=new_events),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return json.dumps(
        _complete_wire_action(action, actor_hint=actor_hint, new_events=new_events),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _observation_new_events(obs: Any) -> list[dict[str, Any]]:
    raw_events = getattr(obs, "new_events", None)
    if raw_events is None:
        raw_events = getattr(obs, "events", [])
    if callable(raw_events):
        raw_events = raw_events()
    events: list[dict[str, Any]] = []
    for event in raw_events or []:
        if isinstance(event, dict):
            events.append(dict(event))
        elif isinstance(event, str):
            events.append(json.loads(event))
        else:
            events.append(_action_to_mjai_dict(event))
    return events


def _action_exact_key(action: dict[str, Any]) -> str:
    normalized = dict(action)
    if "consumed" in normalized:
        normalized["consumed"] = sorted(str(tile) for tile in normalized["consumed"])
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _same_meld_action(candidate: dict[str, Any], legal: dict[str, Any]) -> bool:
    action_type = candidate.get("type")
    if action_type not in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        return False
    if legal.get("type") != action_type:
        return False
    if candidate.get("actor") != legal.get("actor"):
        return False
    if candidate.get("pai") != legal.get("pai"):
        return False
    return sorted(str(tile) for tile in candidate.get("consumed", [])) == sorted(
        str(tile) for tile in legal.get("consumed", [])
    )


def _same_hora_action(candidate: dict[str, Any], legal: dict[str, Any]) -> bool:
    if candidate.get("type") != "hora" or legal.get("type") != "hora":
        return False
    if candidate.get("actor") != legal.get("actor"):
        return False
    if "target" in legal and "target" in candidate and legal.get("target") != candidate.get("target"):
        return False
    if "pai" in legal and "pai" in candidate and legal.get("pai") != candidate.get("pai"):
        return False
    return True


def _action_log_text(action: dict[str, Any] | None) -> str:
    if action is None:
        return "null"
    return json.dumps(action, sort_keys=True, ensure_ascii=False)


def _warn_mortal_legality_fallback(
    *,
    reason: str,
    actor: int | None,
    candidate: dict[str, Any] | None,
    fallback: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> None:
    legal_sample = legal_actions[:8]
    logger.warning(
        "mortal legality guard fallback: reason=%s actor=%s candidate=%s fallback=%s legal_count=%d legal_sample=%s",
        reason,
        actor,
        _action_log_text(candidate),
        _action_log_text(fallback),
        len(legal_actions),
        json.dumps(legal_sample, sort_keys=True, ensure_ascii=False),
    )


def _legalize_action(
    action: dict[str, Any] | None,
    *,
    legal_actions: list[dict[str, Any]],
    actor: int | None,
) -> dict[str, Any]:
    normalized_legal = [
        _sanitize_action(_action_to_mjai_dict(item), actor)
        for item in legal_actions
    ]
    if not normalized_legal:
        return {"type": "none"}

    candidate = _sanitize_action(action or {"type": "none"}, actor)
    reason = "missing_action" if action is None else "illegal_action"
    candidate_key = _action_exact_key(candidate)
    for legal in normalized_legal:
        if _action_exact_key(legal) == candidate_key:
            return legal

    if candidate.get("type") == "dahai":
        for legal in normalized_legal:
            if (
                legal.get("type") == "dahai"
                and legal.get("actor") == candidate.get("actor")
                and legal.get("pai") == candidate.get("pai")
            ):
                if "tsumogiri" in candidate and "tsumogiri" not in legal:
                    legal = dict(legal)
                    legal["tsumogiri"] = bool(candidate["tsumogiri"])
                return legal
    if candidate.get("type") in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        for legal in normalized_legal:
            if _same_meld_action(candidate, legal):
                return legal
    if candidate.get("type") == "hora":
        for legal in normalized_legal:
            if _same_hora_action(candidate, legal):
                return legal

    for legal in normalized_legal:
        if legal.get("type") == "none":
            _warn_mortal_legality_fallback(
                reason=reason,
                actor=actor,
                candidate=action,
                fallback=legal,
                legal_actions=normalized_legal,
            )
            return legal
    fallback = normalized_legal[0]
    _warn_mortal_legality_fallback(
        reason=reason,
        actor=actor,
        candidate=action,
        fallback=fallback,
        legal_actions=normalized_legal,
    )
    return fallback


def _select_observation_action(obs: Any, action: dict[str, Any], actor: int) -> Any:
    selected = obs.select_action_from_mjai(json.dumps(action, ensure_ascii=False))
    if selected is not None:
        return selected

    legal_actions = [_action_to_mjai_dict(item) for item in obs.legal_actions()]
    fallback = _legalize_action(action, legal_actions=legal_actions, actor=actor)
    selected = obs.select_action_from_mjai(json.dumps(fallback, ensure_ascii=False))
    if selected is None:
        raise RuntimeError(f"action could not be converted by RiichiEnv: {fallback}")
    return selected


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

    def observe(self, obs: Any) -> None:
        return None

    def act(self, obs: Any):
        raise NotImplementedError

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any] | str:
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
        wire_payload: str | None = None,
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
                "wire_payload": wire_payload,
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
        wire_payload: str | None = None,
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
                "wire_payload": wire_payload,
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
        wire_payload: str | None = None,
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
                "wire_payload": wire_payload,
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
    rank_pt_lambda: float = 0.0


DEFAULT_DECISION_AGENT_SPEC = DecisionAgentSpec(model_version="xmodel1")
DECISION_AGENT_SPECS: dict[str, DecisionAgentSpec] = {
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
                return _select_observation_action(obs, candidate, actor)
        for action_type in ("hora", "reach", "dahai", "none"):
            for action in legal_actions:
                if action.get("type") == action_type:
                    return _select_observation_action(obs, action, actor)
        if legal_actions:
            return _select_observation_action(obs, legal_actions[0], actor)
        raise RuntimeError("validation-safe local observation has no legal actions")

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any] | str:
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
        return _sanitize_action({"type": "none"}, actor)


class RulebaseObservationAgent(RiichiDevDecisionAgent):
    def __init__(self) -> None:
        self._pending_reach_discard: dict[str, Any] | None = None

    def reset(self) -> None:
        self._pending_reach_discard = None

    def _remember_reach_followup(
        self,
        *,
        snap: dict[str, Any],
        actor: int,
        legal_actions: list[dict[str, Any]],
        chosen: dict[str, Any],
    ) -> None:
        self._pending_reach_discard = None
        if chosen.get("type") != "reach":
            return

        import keqing_core

        followup_legal_actions = [
            action for action in legal_actions if action.get("type") != "reach"
        ]
        if not followup_legal_actions:
            return
        followup = keqing_core.choose_rulebase_action(
            snap,
            actor,
            followup_legal_actions,
        )
        if followup and followup.get("type") == "dahai":
            self._pending_reach_discard = _sanitize_action(followup, actor)

    def choose_mjai_action(
        self,
        *,
        snap: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        actor: int,
    ) -> dict[str, Any]:
        import keqing_core

        if not legal_actions:
            self._pending_reach_discard = None
            return _sanitize_action({"type": "none"}, actor)
        chosen = keqing_core.choose_rulebase_action(snap, actor, legal_actions)
        if chosen is None:
            self._pending_reach_discard = None
            return _sanitize_action({"type": "none"}, actor)
        self._remember_reach_followup(
            snap=snap,
            actor=actor,
            legal_actions=legal_actions,
            chosen=chosen,
        )
        return _sanitize_action(chosen, actor)

    def act(self, obs: Any):
        actor = int(getattr(obs, "player_id", 0))
        snap = _normalize_observation_state(obs, obs.to_dict())
        legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
        chosen = self.choose_mjai_action(
            snap=snap,
            legal_actions=legal_actions,
            actor=actor,
        )
        return _select_observation_action(obs, chosen, actor)

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any] | str:
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
        _obs, snap = _decode_observation(message)
        legal_actions = list(message.get("possible_actions") or [])
        actor = seat if seat is not None else int(snap.get("actor", 0))
        if not legal_actions:
            legal_actions = [{"type": "none"}]
        return self.choose_mjai_action(
            snap=snap,
            legal_actions=[_action_to_mjai_dict(action) for action in legal_actions],
            actor=actor,
        )


class MortalObservationAgent(RiichiDevDecisionAgent):
    def __init__(
        self,
        *,
        model_path: str | Path,
        project_root: str | Path,
        device: str = "cuda",
        verbose: bool = False,
    ) -> None:
        self._model_path = Path(model_path)
        self._mortal_root = Path(project_root) / "third_party" / "Mortal"
        self._device = device
        self._verbose = verbose
        self._seat: int | None = None
        self._bot: Any | None = None

    def reset(self) -> None:
        if self._bot is not None:
            self._bot.reset()

    def _ensure_bot(self, seat: int) -> None:
        if self._bot is not None and self._seat == seat:
            return
        from inference.mortal_bot import MortalReviewBot

        self._seat = seat
        self._bot = MortalReviewBot(
            player_id=seat,
            model_path=self._model_path,
            mortal_root=self._mortal_root,
            device=self._device,
            verbose=self._verbose,
            enable_review_log=False,
        )

    def _react_new_events(self, events: list[dict[str, Any]], seat: int) -> dict[str, Any] | None:
        self._ensure_bot(seat)
        if self._bot is None:
            return None

        reaction: dict[str, Any] | None = None
        for event in events:
            response = self._bot.react(event)
            reaction = _action_to_mjai_dict(response) if response is not None else None
        return reaction

    def choose_mjai_action(
        self,
        *,
        new_events: list[dict[str, Any]],
        legal_actions: list[dict[str, Any]],
        actor: int,
    ) -> dict[str, Any]:
        reaction = self._react_new_events(new_events, actor)
        return _legalize_action(reaction, legal_actions=legal_actions, actor=actor)

    def observe(self, obs: Any) -> None:
        actor = int(getattr(obs, "player_id", 0))
        self._react_new_events(_observation_new_events(obs), actor)

    def act(self, obs: Any):
        actor = int(getattr(obs, "player_id", 0))
        legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
        chosen = self.choose_mjai_action(
            new_events=_observation_new_events(obs),
            legal_actions=legal_actions,
            actor=actor,
        )
        action = obs.select_action_from_mjai(json.dumps(chosen, ensure_ascii=False))
        if action is None:
            raise RuntimeError(f"Mortal selected action could not be converted by RiichiEnv: {chosen}")
        return action

    def select_action(self, message: dict[str, Any], seat: int | None) -> dict[str, Any] | str:
        mtype = message.get("type")
        if mtype == "request_action":
            obs, snap = _decode_observation(message)
            actor = seat if seat is not None else int(snap.get("actor", 0))
            new_events = _observation_new_events(obs)
            legal_actions = list(message.get("possible_actions") or [])
            if not legal_actions:
                legal_actions = [_action_to_mjai_dict(action) for action in obs.legal_actions()]
            else:
                legal_actions = [_action_to_mjai_dict(action) for action in legal_actions]
            chosen = self.choose_mjai_action(
                new_events=new_events,
                legal_actions=legal_actions,
                actor=actor,
            )
            action = obs.select_action_from_mjai(json.dumps(chosen, ensure_ascii=False))
            if action is None:
                logger.warning(
                    "mortal riichienv conversion fallback: actor=%s chosen=%s",
                    actor,
                    _action_log_text(chosen),
                )
            return _action_to_mjai_wire_payload(chosen, actor_hint=actor, new_events=new_events)

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
        rank_pt_lambda: float = 0.0,
        model_version: str | None = "xmodel1",
    ):
        _ensure_model_runtime_imports()
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
            rank_pt_lambda=rank_pt_lambda,
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
    rank_pt_lambda: float | None = None,
    validation_safe: bool = False,
) -> RiichiDevDecisionAgent:
    if validation_safe:
        return ValidationSafeAgent()
    if bot_name == "rulebase":
        return RulebaseObservationAgent()
    if bot_name == "mortal":
        resolved_model_path = _resolve_model_path(
            bot_name=bot_name,
            project_root=project_root,
            model_path=model_path,
        )
        if resolved_model_path is None:
            raise ValueError("mortal requires a Mortal checkpoint path")
        return MortalObservationAgent(
            model_path=resolved_model_path,
            project_root=project_root,
            device=device,
            verbose=verbose,
        )
    spec = DECISION_AGENT_SPECS.get(model_version or bot_name)
    if spec is None:
        inferred_version = model_version or bot_name
        spec = DecisionAgentSpec(model_version=inferred_version)
    effective_rank_pt_lambda = (
        spec.rank_pt_lambda if rank_pt_lambda is None else float(rank_pt_lambda)
    )
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
        rank_pt_lambda=effective_rank_pt_lambda,
        model_version=spec.model_version,
    )


@dataclass(slots=True)
class RiichiDevClientConfig:
    token: str
    bot_name: str = "xmodel1"
    model_version: str | None = None
    queue: str = "ranked"
    base_url: str = DEFAULT_BASE_URL
    ws_url_override: str | None = None
    model_path: Path | None = None
    project_root: Path = Path.cwd()
    device: str = "cuda"
    rank_pt_lambda: float | None = None
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
            rank_pt_lambda=config.rank_pt_lambda,
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
                message = json.loads(raw_message)
                request_seq: int | None = None
                if message.get("type") == "request_action":
                    self._request_seq += 1
                    request_seq = self._request_seq
                t0 = time.perf_counter()
                try:
                    response = self.handle_message(message)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    audit_message = _enrich_message_for_audit(message)
                    self.audit_logger.log_agent_error(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        request_seq=request_seq,
                        message=audit_message,
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
                audit_response = _action_to_mjai_dict(response) if response is not None else None
                wire_payload = _action_to_mjai_wire_payload(response) if response is not None else None
                if response is not None:
                    try:
                        await websocket.send(wire_payload)
                    except Exception as exc:
                        self.audit_logger.log_send_result(
                            queue=self.config.queue,
                            bot_name=self.config.bot_name,
                            model_version=self.config.model_version,
                            seat=self.seat,
                            request_seq=request_seq,
                            response=audit_response,
                            success=False,
                            wire_payload=wire_payload,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise

                audit_message = _enrich_message_for_audit(message)
                if message.get("type") == "request_action" and response is not None:
                    self.audit_logger.log_request_action(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        request_seq=request_seq,
                        message=audit_message,
                        response=audit_response,
                        latency_ms=latency_ms,
                        wire_payload=wire_payload,
                    )
                elif response is not None:
                    self.audit_logger.log_protocol_action(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        trigger_message=audit_message,
                        response=audit_response,
                        latency_ms=latency_ms,
                        wire_payload=wire_payload,
                    )
                elif message.get("type") in {"start_game", "start_kyoku", "end_kyoku", "end_game", "validation_result", "error"}:
                    self.audit_logger.log_event(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        message=audit_message,
                    )
                if response is not None:
                    self.audit_logger.log_send_result(
                        queue=self.config.queue,
                        bot_name=self.config.bot_name,
                        model_version=self.config.model_version,
                        seat=self.seat,
                        request_seq=request_seq,
                        response=audit_response,
                        success=True,
                        wire_payload=wire_payload,
                    )
                    if self.config.verbose:
                        logger.info("recv: %s", message.get("type"))
                        logger.info("send: %s latency_ms=%.1f", wire_payload, latency_ms)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | str | None:
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
    agent: RiichiDevDecisionAgent | None = None,
    agents: Mapping[int, RiichiDevDecisionAgent] | Sequence[RiichiDevDecisionAgent] | None = None,
    game_mode: int = 2,
    seed: int | None = 42,
    max_steps: int = 10000,
) -> dict[str, Any]:
    if agent is None and agents is None:
        raise ValueError("run_local_game requires agent or agents")

    def local_agent_for(pid: int) -> RiichiDevDecisionAgent:
        if agents is None:
            assert agent is not None
            return agent
        return agents[pid]

    reset_seen: set[int] = set()
    reset_candidates = agents.values() if isinstance(agents, Mapping) else agents
    if reset_candidates is None:
        reset_candidates = [agent]
    for local_agent in reset_candidates:
        if local_agent is None:
            continue
        marker = id(local_agent)
        if marker in reset_seen:
            continue
        reset_seen.add(marker)
        local_agent.reset()

    env = RiichiEnv(game_mode=game_mode, seed=seed)
    observations = env.get_observations()
    step_count = 0

    while not env.done():
        actions_to_step: dict[int, Any] = {}
        for pid, obs in observations.items():
            actions = obs.legal_actions()
            if not actions:
                local_agent_for(int(pid)).observe(obs)
                continue
            actions_to_step[int(pid)] = local_agent_for(int(pid)).act(obs)
        if not actions_to_step:
            raise RuntimeError("local game stalled without any legal action")
        observations = env.step(actions_to_step)
        step_count += 1
        if step_count >= max_steps:
            raise RuntimeError(f"local game exceeded max_steps={max_steps}")

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
        default="xmodel1",
        help="bot family / checkpoint namespace to run (xmodel1, keqingv4, mortal, rulebase)",
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
        "--rank-pt-lambda",
        type=float,
        default=0.0,
        help="keqingv4 runtime placement rerank scale; default disabled",
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
        isolated_agents = args.bot_name == "mortal" and not args.validation_safe
        agent = None
        agents = None
        if isolated_agents:
            agents = {
                pid: create_riichi_dev_agent(
                    bot_name=args.bot_name,
                    project_root=project_root,
                    model_path=explicit_model_path,
                    device=args.device,
                    verbose=args.verbose,
                    model_version=args.model_version or None,
                    rank_pt_lambda=args.rank_pt_lambda,
                    validation_safe=args.validation_safe,
                )
                for pid in range(4)
            }
        else:
            agent = create_riichi_dev_agent(
                bot_name=args.bot_name,
                project_root=project_root,
                model_path=explicit_model_path,
                device=args.device,
                verbose=args.verbose,
                model_version=args.model_version or None,
                rank_pt_lambda=args.rank_pt_lambda,
                validation_safe=args.validation_safe,
            )
        result = run_local_game(
            agent=agent,
            agents=agents,
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
        raise SystemExit("missing bot token; pass --token or set LATTEKEY/RIICHI_BOT_TOKEN in project .env")

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
        rank_pt_lambda=args.rank_pt_lambda,
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
        rank_pt_lambda=args.rank_pt_lambda,
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
