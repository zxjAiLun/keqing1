# -*- coding: utf-8 -*-
"""participants 迁移：种子注册表 + replay 回填的幂等性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.participants import seed_participants as seed
from participants import ledger, registry


@pytest.fixture(autouse=True)
def participants_env(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    monkeypatch.setenv("KEQING_LADDER_CONFIG_DIR", str(configs))
    replays = tmp_path / "replays"
    replays.mkdir(exist_ok=True)
    monkeypatch.setenv("KEQING_PARTICIPANT_REPLAYS_DIR", str(replays))
    return {"root": root, "configs": configs, "replays": replays}


def _write_registry(configs: Path) -> None:
    (configs / "official-ladder-v1.json").write_text(
        json.dumps(
            {
                "schema": "keqing.ladder.season.v1",
                "season_id": "official-ladder-v1",
                "status": "running",
                "models": [
                    {
                        "model_id": "human",
                        "accounts": [{"account_id": "nick@01", "display_name": "Nick"}],
                    },
                    {
                        "model_id": "70k",
                        "checkpoint": "artifacts/70k.pth",
                        "accounts": [
                            {"account_id": "70k@01", "display_name": "70k-1"},
                            {"account_id": "70k@02", "display_name": "70k-2"},
                        ],
                    },
                    {
                        "model_id": "ext_mortal",
                        "accounts": [{"account_id": "ext_mortal@01", "display_name": "ext_mortal-1"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_replay(replays: Path, replay_id: str, names, scores, kyoku_count=8) -> None:
    (replays / replay_id).mkdir(exist_ok=True)
    (replays / replay_id / "meta.json").write_text(
        json.dumps(
            {
                "replay_id": replay_id,
                "created_at": "2026-08-01T12:00:00+08:00",
                "kyoku_count": kyoku_count,
                "player_names": names,
                "final_scores": scores,
            }
        ),
        encoding="utf-8",
    )


def test_seed_registry_creates_accounts_and_models(participants_env):
    _write_registry(participants_env["configs"])
    state = {"seeded_registries": [], "ingested_replays": {}}
    seed.seed_registries(state, dry_run=False)

    assert registry.get_account("nick@01").default_controller == "human_ui"
    assert registry.get_account("70k@01").account_type == "managed_bot"
    assert registry.get_account("ext_mortal@01").account_type == "external_bot"
    assert registry.get_model_identity("70k") is not None
    assert len(registry.get_model_identity("70k").artifacts) == 1
    # 幂等：再跑一次不新增
    before = len(registry.list_accounts())
    seed.seed_registries(state, dry_run=False)
    assert len(registry.list_accounts()) == before


def test_backfill_replays_creates_matches_and_placeholders(participants_env):
    _write_registry(participants_env["configs"])
    seed.seed_registries({"seeded_registries": [], "ingested_replays": {}}, dry_run=False)

    _write_replay(participants_env["replays"], "replay_a", ["Nick", "70k-1", "Friend Alice", "Bot X"], [30000, 25000, 25000, 20000])
    _write_replay(participants_env["replays"], "replay_b", ["Nick", "70k-2", "Friend Alice", "Bot X"], [25000, 25000, 25000, 25000])

    state = {"seeded_registries": ["official-ladder-v1.json"], "ingested_replays": {}}
    seed.backfill_replays(state, dry_run=False)

    assert ledger.list_matches().total == 2
    # 未知名自动建占位账号
    assert registry.get_account("imp_Friend-Alice") is not None
    assert registry.get_account("imp_Bot-X") is not None
    assert registry.get_account("imp_Friend-Alice").migrated_from_replay is False  # 显式创建默认
    # 幂等：再回填不重复
    seed.backfill_replays(state, dry_run=False)
    assert ledger.list_matches().total == 2


def test_backfill_skips_incomplete_and_placeholder_names(participants_env):
    _write_registry(participants_env["configs"])
    seed.seed_registries({"seeded_registries": [], "ingested_replays": {}}, dry_run=False)
    _write_replay(participants_env["replays"], "replay_e", ["E", "S", "W", "N"], [25000] * 4)
    _write_replay(participants_env["replays"], "replay_f", ["Nick", "70k-1"], [30000, 25000])
    state = {"seeded_registries": ["official-ladder-v1.json"], "ingested_replays": {}}
    seed.backfill_replays(state, dry_run=False)
    assert ledger.list_matches().total == 0


def test_dry_run_writes_nothing(participants_env):
    _write_registry(participants_env["configs"])
    _write_replay(participants_env["replays"], "replay_a", ["Nick", "70k-1", "Friend Alice", "Bot X"], [30000, 25000, 25000, 20000])
    state = {"seeded_registries": [], "ingested_replays": {}}
    seed.seed_registries(state, dry_run=True)
    seed.backfill_replays(state, dry_run=True)
    assert registry.list_accounts() == []
    assert ledger.list_matches().total == 0
    assert not (participants_env["root"] / "migration_state.json").exists()
