# -*- coding: utf-8 -*-
"""天凤链接统一摄入（R10-D）：URL → preview（不落账）→ 身份解析 → 统一账本 + replay artifact。

数据流：
    parse_tenhou_url(text)
    → download_tenhou6(log_id)          # 天凤 mjlog2json 下载
    → tenhou6_to_mjai_events            # 规范化事件（复用 convert.tenhou6_utils）
    → build_preview(...)                # 四座原始名/分数/顺位/逐局摘要，不落账
    → resolve_and_create_match(...)     # 身份解析 → 原子落账 + 持久化 artifact

幂等键：provider + external_match_id（tenhou log_id）。重复导入被拒绝（409）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import aliases, ledger, registry
from .paths import TZ_OFFSET, data_root, now_iso, atomic_write_text
from .schemas import (
    REPLAY_ARTIFACT_SCHEMA,
    AccountCreate,
    ExternalAliasCreate,
    MatchCreate,
    MatchSeat,
)
from .ledger import ValidationError


def parse_tenhou_url(text: str) -> dict:
    """解析天凤链接，返回 {provider, log_id, tw}。仅接受 tenhou.net。"""
    from convert.link_converter import parse_log_url

    info = parse_log_url(text)
    if info.get("site") != "tenhou":
        raise ValueError("仅支持天凤牌谱链接（tenhou.net）")
    return {
        "provider": "tenhou",
        "external_match_id": info["log_id"],
        "log_id": info["log_id"],
        "tw": int(info.get("tw", "0")),
    }


def download_tenhou6(log_id: str) -> dict:
    """从天凤接口下载 tenhou6 JSON。"""
    from urllib.request import Request, urlopen

    endpoint = f"https://tenhou.net/5/mjlog2json.cgi?{log_id}"
    req = Request(
        endpoint,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://tenhou.net/"},
    )
    with urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace").strip()
    if not body:
        raise ValueError("天凤服务器返回空内容，牌谱可能不存在")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:120].replace("\n", " ")
        raise ValueError(f"天凤返回非 JSON 内容: '{preview}...'") from exc


def tenhou6_events(tenhou6: dict) -> list[dict]:
    from convert.tenhou6_utils import tenhou6_to_mjai_events

    return tenhou6_to_mjai_events(tenhou6)


def player_names(events: list[dict]) -> list[str]:
    for event in events:
        if event.get("type") == "start_game":
            names = list(event.get("names") or [])
            return (names + ["P3"])[:4]
    return ["P0", "P1", "P2", "P3"]


def hand_summaries(events: list[dict]) -> list[dict]:
    """从 mjai 事件流提取逐局摘要（R10-D 第一版：分数变化/和了/流局）。"""
    hands: list[dict] = []
    current: dict | None = None
    for event in events:
        etype = event.get("type")
        if etype == "start_kyoku":
            if current is not None:
                current["scores_after"] = event.get("scores")
                hands.append(current)
            current = {
                "bakaze": event.get("bakaze"),
                "kyoku": event.get("kyoku"),
                "honba": event.get("honba"),
                "oya": event.get("oya"),
                "scores_before": event.get("scores"),
                "winners": [],
                "ryukyoku": None,
            }
        elif etype == "hora" and current is not None:
            actor = int(event.get("actor", -1))
            target = int(event.get("target", -1))
            current["winners"].append(
                {
                    "actor": actor,
                    "win_type": "tsumo" if target == actor else "ron",
                    "target": None if target == actor else target,
                    "deltas": list(event.get("deltas") or [0, 0, 0, 0]),
                }
            )
        elif etype == "ryukyoku" and current is not None:
            current["ryukyoku"] = {
                "reason": event.get("reason"),
                "deltas": list(event.get("deltas") or [0, 0, 0, 0]),
            }
    if current is not None:
        hands.append(current)
    return hands


def _final_scores(events: list[dict]) -> list[int]:
    scores: list[int] | None = None
    for event in events:
        etype = event.get("type")
        if etype == "start_kyoku":
            scores = [int(s) for s in (event.get("scores") or [25000] * 4)]
        elif etype in ("hora", "ryukyoku") and scores is not None:
            deltas = [int(d) for d in (event.get("deltas") or [0, 0, 0, 0])]
            scores = [scores[i] + deltas[i] for i in range(4)]
    return scores or [25000] * 4


def _occurred_at_from_log_id(log_id: str) -> str:
    try:
        return (
            datetime.strptime(log_id[:8], "%Y%m%d")
            .replace(tzinfo=TZ_OFFSET)
            .isoformat(timespec="seconds")
        )
    except ValueError:
        return now_iso()


def build_preview(text: str, *, session_id: str | None = None) -> dict:
    """解析 + 下载 + 生成不落账的 preview，含逐座候选身份。"""
    parsed = parse_tenhou_url(text)
    tenhou6 = download_tenhou6(parsed["log_id"])
    events = tenhou6_events(tenhou6)
    names = player_names(events)
    hands = hand_summaries(events)
    final_scores = _final_scores(events)
    ranks = list(ledger.final_ranks(final_scores, initial_oya=0))
    game_length = "hanchan" if len(hands) > 4 else "tonpu"
    duplicate = ledger.find_match_by_external("tenhou", parsed["log_id"])

    seats = []
    for seat, name in enumerate(names):
        candidates = aliases.resolve_candidates("tenhou", name, session_id=session_id)
        seats.append(
            {
                "seat": seat,
                "raw_name": name,
                "candidates": [c.model_dump() for c in candidates],
                "auto_account_id": (
                    candidates[0].account_id if len(candidates) == 1 and candidates[0].confidence == "confirmed" else None
                ),
            }
        )
    return {
        "schema": "keqing.participant.intake_preview.v1",
        "provider": "tenhou",
        "external_match_id": parsed["log_id"],
        "log_id": parsed["log_id"],
        "occurred_at": _occurred_at_from_log_id(parsed["log_id"]),
        "game_length": game_length,
        "starting_points": 25000,
        "raw_player_names": names,
        "final_scores": final_scores,
        "ranks": ranks,
        "data_completeness": "full_replay",
        "hand_count": len(hands),
        "events_count": len(events),
        "seats": seats,
        "duplicate_match_id": duplicate.match_id if duplicate else None,
    }


def _replay_artifact_dir(log_id: str) -> Path:
    return data_root() / "replays" / log_id


def persist_replay_artifact(log_id: str, *, tenhou6: dict, events: list[dict], hands: list[dict], summary: dict) -> None:
    """持久化 replay artifact（按 log_id 目录，幂等覆盖）。"""
    directory = _replay_artifact_dir(log_id)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / "tenhou6.json", json.dumps(tenhou6, ensure_ascii=False))
    with open(directory / "events.jsonl", "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    with open(directory / "hands.jsonl", "w", encoding="utf-8") as fh:
        for hand in hands:
            fh.write(json.dumps(hand, ensure_ascii=False) + "\n")
    atomic_write_text(
        directory / "summary.json",
        json.dumps({"schema": REPLAY_ARTIFACT_SCHEMA, "log_id": log_id, **summary}, ensure_ascii=False),
    )


def read_replay_artifact(log_id: str) -> dict | None:
    directory = _replay_artifact_dir(log_id)
    if not (directory / "summary.json").exists():
        return None
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    hands = [
        json.loads(line)
        for line in (directory / "hands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary["hands"] = hands
    summary["has_events"] = (directory / "events.jsonl").exists()
    return summary


def resolve_and_create_match(
    *,
    log_id: str,
    resolutions: list[dict],
    session_id: str | None = None,
    note: str | None = None,
) -> dict:
    """按用户逐座决议落账：建/指派账号、注册别名、写 ledger match + replay artifact。

    ``resolutions``: [{seat, action: "assign"|"create", account_id?, display_name?,
                      account_type?, alias_scope, confidence}]（长度为 4，按 seat 索引）。
    """
    existing = ledger.find_match_by_external("tenhou", log_id)
    if existing is not None:
        raise ValueError(f"该天凤牌谱已导入为对局 {existing.match_id}")

    preview = build_preview(f"https://tenhou.net/3/?log={log_id}", session_id=session_id)
    tenhou6 = download_tenhou6(log_id)
    events = tenhou6_events(tenhou6)
    names = preview["raw_player_names"]

    if len(resolutions) != 4:
        raise ValueError("必须提供恰好 4 个座位的解析决议")
    if {r["seat"] for r in resolutions} != {0, 1, 2, 3}:
        raise ValueError("座位解析决议必须覆盖 seat 0..3")

    seats: list[MatchSeat] = []
    resolution_audit: dict = {}
    for raw_res in sorted(resolutions, key=lambda r: r["seat"]):
        seat = int(raw_res["seat"])
        name = names[seat]
        action = raw_res.get("action", "assign")
        alias_scope = raw_res.get("alias_scope") or "match"
        confidence = raw_res.get("confidence") or "confirmed"
        # session 绑定必须有会话；无会话时 session scope 回落为 match
        if alias_scope == "session" and not session_id:
            alias_scope = "match"

        if action == "create":
            account = registry.create_account(
                AccountCreate(
                    display_name=raw_res.get("display_name") or name or f"seat{seat + 1}",
                    account_type=raw_res.get("account_type") or "external_bot",
                    default_controller=raw_res.get("default_controller"),
                )
            )
            account_id = account.account_id
        else:
            account_id = raw_res.get("account_id")
            if not account_id or registry.get_account(account_id) is None:
                raise ValueError(f"座位 {seat} 指派了不存在的账号: {account_id}")

        if alias_scope and alias_scope != "none":
            aliases.register_alias(
                ExternalAliasCreate(
                    provider="tenhou",
                    external_id=name,
                    display_name=name,
                    account_id=account_id,
                    scope=alias_scope,
                    session_id=session_id if alias_scope == "session" else None,
                    confidence=confidence,
                )
            )
        seats.append(MatchSeat(seat=seat, account_id=account_id))
        resolution_audit[str(seat)] = {
            "raw_name": name,
            "action": action,
            "account_id": account_id,
            "alias_scope": alias_scope,
            "confidence": confidence,
        }

    summary = {
        "match_id": None,  # create_match 后回填
        "log_id": log_id,
        "names": names,
        "final_scores": preview["final_scores"],
        "ranks": preview["ranks"],
        "game_length": preview["game_length"],
        "started_at": preview["occurred_at"],
    }
    persist_replay_artifact(
        log_id,
        tenhou6=tenhou6,
        events=events,
        hands=hand_summaries(events),
        summary=summary,
    )

    match = ledger.create_match(
        MatchCreate(
            occurred_at=preview["occurred_at"],
            game_length=preview["game_length"],
            source="imported",
            source_ref=log_id,
            data_completeness="full_replay",
            replay_id=log_id,
            provider="tenhou",
            external_match_id=log_id,
            raw_player_names=names,
            resolution=resolution_audit,
            seats=seats,
            final_scores=preview["final_scores"],
            note=note,
        ),
        registry,
    )
    return {"match_id": match.match_id, "log_id": log_id}


__all__ = [
    "parse_tenhou_url",
    "download_tenhou6",
    "tenhou6_events",
    "player_names",
    "hand_summaries",
    "build_preview",
    "resolve_and_create_match",
    "read_replay_artifact",
    "persist_replay_artifact",
]
