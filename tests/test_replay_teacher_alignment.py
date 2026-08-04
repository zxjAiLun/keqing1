# -*- coding: utf-8 -*-
"""Teacher overlay 事件身份对齐测试（P1：source_event_index + actor + decision_kind）。

覆盖：
- runtime teacher report 携带事件身份字段；
- _attach_teacher_report_overlays 按事件身份挂载，锁定 step 3 / 8 / 30；
- 多匹配（ambiguous）与零匹配（unmatched）不被静默挂载；
- 旧报告（无身份字段）走 legacy step fallback 并输出 degraded 元数据。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from replay.server import _attach_teacher_report_overlays, _build_runtime_teacher_report

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "review_state"
DECISIONS_PATH = FIXTURE_DIR / "replay_97508ebc_p3" / "decisions.json"


def _load_decisions() -> dict:
    return json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))


def _write_report(tmp_path: Path, decisions: dict, model: str) -> Path:
    report = _build_runtime_teacher_report(
        replay_id="replay_97508ebc_1785858273",
        model_type=model,
        player_id=3,
        checkpoint=Path("/nonexistent/checkpoint.pth"),
        decisions=decisions,
    )
    path = tmp_path / f"{model.replace(' ', '_')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def _own_entry(decisions: dict, step: int) -> dict:
    return next(e for e in decisions["log"] if e.get("step") == step)


def _review_pai_set(review: dict) -> set:
    return {
        (c.get("action") or {}).get("pai")
        for c in review.get("candidates", [])
        if isinstance(c.get("action"), dict) and c["action"].get("pai")
    }


def test_runtime_report_entries_carry_event_identity(tmp_path):
    decisions = _load_decisions()
    report = json.loads(_write_report(tmp_path, decisions, "External Mortal").read_text(encoding="utf-8"))
    entries = [
        entry
        for kyoku in report["review"]["kyokus"]
        for entry in kyoku.get("entries", [])
    ]
    assert len(entries) > 0
    for entry in entries:
        assert "source_event_index" in entry
        assert "actor" in entry
        assert "decision_kind" in entry
    # 真实 step 3（摸 5s 打 W）应带身份：draw_discard / actor 3 / sei 9
    step3 = next(e for e in entries if e.get("step") == 3)
    assert step3["decision_kind"] == "draw_discard"
    assert step3["actor"] == 3
    assert step3["source_event_index"] == 9


def test_attach_by_event_identity_locks_steps_3_8_30(tmp_path):
    decisions = _load_decisions()
    report_paths = [
        _write_report(tmp_path, decisions, "External Mortal"),
        _write_report(tmp_path, decisions, "70k"),
    ]
    attached = _attach_teacher_report_overlays(decisions, report_paths)
    overlays = attached["teacher_review_overlays"]
    assert len(overlays) == 2
    for overlay in overlays:
        assert overlay["alignment"] == "event_identity"
        assert overlay["alignment_stats"]["exact_event_identity"] > 0
        assert overlay["alignment_stats"]["ambiguous"] == 0
        assert overlay["alignment_stats"]["unmatched"] == 0

    # 锁定 step 3 / 8 / 30：两份模型各挂一次，actual_action == 本地 gt_action，
    # 候选 pai 集合属于该 entry 的 hand+tsumo。
    for step in (3, 8, 30):
        entry = _own_entry(attached, step)
        reviews = entry.get("teacher_reviews") or []
        assert len(reviews) == 2, f"step {step} 应挂上两份 teacher review（得到 {len(reviews)}）"
        models = {review["model"] for review in reviews}
        assert models == {"External Mortal", "70k"}, f"step {step} 模型集合应为两份"
        hand_set = set(entry.get("hand") or [])
        if entry.get("tsumo_pai"):
            hand_set.add(entry["tsumo_pai"])
        for review in reviews:
            actual = review.get("actual_action")
            gt = entry.get("gt_action")
            assert actual is not None and gt is not None
            assert actual.get("type") == gt.get("type") and actual.get("pai") == gt.get("pai"), (
                f"step {step} teacher actual {actual} 应与本地 gt_action {gt} 相同"
            )
            for pai in _review_pai_set(review):
                assert pai in hand_set, f"step {step} teacher 候选 {pai} 不属于当前 entry 手牌"


def test_identity_mismatch_is_not_silently_attached(tmp_path):
    decisions = _load_decisions()
    report_path = _write_report(tmp_path, decisions, "70k")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 篡改 step 3 的身份：指向错误事件（真实事件 9，改成 3）
    step3_teacher = None
    for kyoku in report["review"]["kyokus"]:
        for entry in kyoku["entries"]:
            if entry.get("step") == 3:
                step3_teacher = entry
    assert step3_teacher is not None
    step3_teacher["source_event_index"] = 3
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    attached = _attach_teacher_report_overlays(decisions, [report_path])
    overlay = attached["teacher_review_overlays"][0]
    assert overlay["alignment_stats"]["unmatched"] >= 1
    step3_local = _own_entry(attached, 3)
    assert all(review["model"] != "70k" for review in (step3_local.get("teacher_reviews") or [])), (
        "错误身份不得挂载到 step 3"
    )


def test_ambiguous_identity_is_rejected(tmp_path):
    decisions = _load_decisions()
    # teacher 报告来自原始 decisions（只有一个 step 3 entry）
    report_path = _write_report(tmp_path, decisions, "70k")
    # 构造本地重复：clone step 3（共享同一 source_event_index 9），制造多候选
    duplicated = dict(decisions)
    log = list(decisions["log"])
    clone = dict(_own_entry(decisions, 3))
    clone["step"] = 9999
    log.append(clone)
    duplicated["log"] = log

    attached = _attach_teacher_report_overlays(duplicated, [report_path])
    overlay = attached["teacher_review_overlays"][0]
    assert overlay["alignment_stats"]["ambiguous"] >= 1, "多候选应标记 ambiguous"
    step3_local = _own_entry(attached, 3)
    assert all(review["model"] != "70k" for review in (step3_local.get("teacher_reviews") or [])), (
        "ambiguous 不得静默选择第一个挂载"
    )


def test_legacy_step_fallback_is_degraded(tmp_path):
    decisions = _load_decisions()
    report_path = _write_report(tmp_path, decisions, "70k")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 旧报告：去掉身份字段
    for kyoku in report["review"]["kyokus"]:
        for entry in kyoku["entries"]:
            entry.pop("source_event_index", None)
            entry.pop("actor", None)
            entry.pop("decision_kind", None)
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    attached = _attach_teacher_report_overlays(decisions, [report_path])
    overlay = attached["teacher_review_overlays"][0]
    assert overlay["alignment"] == "legacy_step"
    assert overlay["alignment_stats"]["legacy_step"] > 0
    assert overlay["alignment_stats"]["exact_event_identity"] == 0
    step8 = _own_entry(attached, 8)
    assert any(review["model"] == "70k" for review in (step8.get("teacher_reviews") or [])), (
        "旧报告应通过 step legacy fallback 挂载到 step 8"
    )
