"""Lock Tenhou rank/rating semantics and legacy-profile compatibility.

Golden cases from the Round 8 spec:

- 新人 0 + 一般南一位 +30 -> 9级 0（溢出舍弃）
- 1级 90 + 一般南二位 +15 -> 初段 200
- 初段 0 + 南四 -45 -> 1级 0
- 七段 2770 + 鳳凰南一位 +90 -> 八段 1600
- 八段 10 + 鳳凰南四位 -150 -> 七段 1400
- 十段 3950 + 鳳凰南一位 +90 -> 天凤位
- 级位 0 受四位负分 -> 保持级位 0 不降级
- 桌均 R 低于 1500 -> 按 1500 参与公式
- 四位共用同一份赛前 Rating（不逐个更新污染桌均值）
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from replay.rank_systems import (
    LegacyFixedProfile,
    PlayerRankState,
    RankResolutionError,
    Tenhou4pRanked,
    create_rank_system,
)
from replay.rank_systems.tenhou import POSITIVE_PT


def _state(rank_id: str, pt: int, rating: float = 1500.0, games: int = 0) -> PlayerRankState:
    return PlayerRankState(rank_id=rank_id, pt=pt, rating=Decimal(str(rating)), games=games)


def _table(room: str = "ippan", game_length: str = "hanchan", avg: float = 1600.0) -> object:
    from replay.rank_systems.base import TableContext

    return TableContext(
        room=room,
        game_length=game_length,
        positive_pt=POSITIVE_PT[game_length][room],
        avg_rating=Decimal(str(avg)),
        strict=True,
    )


@pytest.fixture()
def ranked() -> Tenhou4pRanked:
    return Tenhou4pRanked(version="test")


# --- Golden PT transitions --------------------------------------------------

def test_newcomer_promotion_discards_overflow(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("newcomer", 0), placement=1, table=_table("ippan"))
    assert update.pt_delta == 30
    assert update.transition == "promotion"
    assert update.rank_after == "9kyu"
    assert update.pt_after == 0


def test_1kyu_promotion_to_1dan(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("1kyu", 90), placement=2, table=_table("ippan"))
    assert update.pt_delta == 15
    assert update.transition == "promotion"
    assert update.rank_after == "1dan"
    assert update.pt_after == 200


def test_1dan_demotes_on_negative_pt(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("1dan", 0), placement=4, table=_table("houou", "hanchan"))
    assert update.pt_delta == -45
    assert update.transition == "demotion"
    assert update.rank_after == "1kyu"
    assert update.pt_after == 0


def test_7dan_promotion_resets_to_next_initial(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("7dan", 2770), placement=1, table=_table("houou", "hanchan"))
    assert update.pt_delta == 90
    assert update.transition == "promotion"
    assert update.rank_after == "8dan"
    assert update.pt_after == 1600


def test_8dan_demotes_to_7dan_initial(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("8dan", 10), placement=4, table=_table("houou", "hanchan"))
    assert update.pt_delta == -150
    assert update.transition == "demotion"
    assert update.rank_after == "7dan"
    assert update.pt_after == 1400


def test_10dan_promotes_to_tenhou(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("10dan", 3950), placement=1, table=_table("houou", "hanchan"))
    assert update.transition == "tenhou"
    assert update.rank_after == "tenhou"


def test_tenhou_pt_delta_is_zero_for_all_placements(ranked: Tenhou4pRanked) -> None:
    for placement in (1, 2, 3, 4):
        update = ranked.apply_result(
            _state("tenhou", 0, rating=2000.0),
            placement=placement,
            table=_table("houou", "hanchan", avg=2000.0),
        )
        assert update.pt_delta == 0
        assert update.pt_after == 0
        assert update.transition == "none"
        assert update.rank_after == "tenhou"
        # Rating 仍正常变化
        assert update.rating_after != update.rating_before


def test_tenhou_rank_meta_has_no_pt(ranked: Tenhou4pRanked) -> None:
    meta = ranked.rank_meta("tenhou")
    assert meta.initial_pt is None
    assert meta.target_pt is None
    assert meta.is_tenhou is True


def test_config_validation_requires_version() -> None:
    with pytest.raises(ValueError, match="version"):
        Tenhou4pRanked(version="")


def test_config_validation_rejects_unknown_room() -> None:
    with pytest.raises(ValueError, match="unknown room"):
        Tenhou4pRanked(version="test", room_policy="fixed", room="lobby")


def test_config_validation_rejects_non_finite_rating() -> None:
    with pytest.raises(ValueError, match="initial_rating"):
        Tenhou4pRanked(version="test", initial_rating=float("nan"))


def test_kyu_never_demotes_pt_floored_at_zero(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("1kyu", 5), placement=4, table=_table("houou", "hanchan"))
    assert update.pt_delta == -30
    assert update.transition == "none"
    assert update.rank_after == "1kyu"
    assert update.pt_after == 0


def test_newcomer_fourth_has_no_penalty(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("newcomer", 0), placement=4, table=_table("ippan", "hanchan"))
    assert update.pt_delta == 0
    assert update.pt_after == 0


def test_2kyu_fourth_hanchan_penalty(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("2kyu", 10), placement=4, table=_table("ippan", "hanchan"))
    assert update.pt_delta == -15
    assert update.pt_after == 0  # floored, no demotion


def test_tonpuu_tables_are_lower(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(_state("2kyu", 10), placement=4, table=_table("ippan", "tonpuu"))
    assert update.pt_delta == -10
    first = ranked.apply_result(_state("newcomer", 0), placement=1, table=_table("houou", "tonpuu"))
    assert first.pt_delta == 60


# --- Rating semantics -------------------------------------------------------

def test_rating_uses_1500_floor_for_low_table_average(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(
        _state("newcomer", 0, rating=1500.0),
        placement=1,
        table=_table("ippan", "hanchan", avg=1400.0),
    )
    # Without the floor: 30 + (1400-1500)/40 = 27.5. Floor applies -> 30.
    assert update.rating_delta_raw == Decimal("30")
    assert update.rating_after == Decimal("1530")


def test_rating_round_up_two_decimals(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(
        _state("newcomer", 0, rating=1511.04),
        placement=2,
        table=_table("ippan", "hanchan", avg=1600.0),
    )
    assert update.rating_delta_raw == Decimal("12.224")
    assert update.rating_after == Decimal("1511.04") + Decimal("12.23")


def test_rating_round_up_negative_toward_plus_infinity(ranked: Tenhou4pRanked) -> None:
    update = ranked.apply_result(
        _state("1dan", 200, rating=2001.11),
        placement=4,
        table=_table("houou", "hanchan", avg=1500.0),
    )
    # delta_raw = -30 + (1500 - 2001.11)/40 = -42.52775 -> ceil -> -42.52
    assert update.rating_delta_raw == Decimal("-42.52775")
    assert update.rating_after == Decimal("2001.11") + Decimal("-42.52")


def test_rating_correction_before_and_after_400_games(ranked: Tenhou4pRanked) -> None:
    early = ranked.apply_result(
        _state("newcomer", 0, rating=1500.0, games=100),
        placement=1,
        table=_table("ippan", "hanchan", avg=1500.0),
    )
    # 1 - 100*0.002 = 0.8 -> 0.8 * 30 = 24
    assert early.rating_delta_raw == Decimal("24")
    late = ranked.apply_result(
        _state("newcomer", 0, rating=1500.0, games=400),
        placement=1,
        table=_table("ippan", "hanchan", avg=1500.0),
    )
    assert late.rating_delta_raw == Decimal("6")  # 0.2 * 30


# --- Table resolution -------------------------------------------------------

def test_room_selection_highest_common_eligible(ranked: Tenhou4pRanked) -> None:
    # 1級 可进上级卓；级位低于 1級 只能进一般卓
    assert ranked.resolve_table([_state("1kyu", 50, rating=1600)] * 4).room == "joukyuu"
    assert ranked.resolve_table([_state("2kyu", 30, rating=1600)] * 4).room == "ippan"
    assert ranked.resolve_table([_state("1dan", 200, rating=1600)] * 4).room == "joukyuu"
    assert ranked.resolve_table([_state("4dan", 800, rating=1900)] * 4).room == "tokujou"
    assert ranked.resolve_table([_state("7dan", 1400, rating=2100)] * 4).room == "houou"


def test_1kyu_hanchan_first_place_scores_joukyuu(ranked: Tenhou4pRanked) -> None:
    table = ranked.resolve_table([_state("1kyu", 90, rating=1600)] * 4)
    assert table.room == "joukyuu"
    update = ranked.apply_result(_state("1kyu", 90), placement=1, table=table)
    assert update.pt_delta == 60


def test_1dan_and_high_rating_4dan_share_joukyuu(ranked: Tenhou4pRanked) -> None:
    players = [
        _state("1dan", 200, rating=1500.0),
        _state("4dan", 800, rating=1900.0),
        _state("1dan", 200, rating=1500.0),
        _state("4dan", 800, rating=1900.0),
    ]
    assert ranked.resolve_table(players).room == "joukyuu"


def test_houou_requires_rank_and_rating_only() -> None:
    # 凤凰卓只要求段位与 Rating，不引入天凤账号付费领域。
    table = Tenhou4pRanked(version="test").resolve_table([_state("7dan", 1400, rating=2100)] * 4)
    assert table.room == "houou"
    # 7dan R1999：凤凰门槛未到 -> 特上（R>=1800 且段位>=四段）
    low = Tenhou4pRanked(version="test").resolve_table([_state("7dan", 1400, rating=1999)] * 4)
    assert low.room == "tokujou"


def test_no_common_room_raises_loudly(ranked: Tenhou4pRanked) -> None:
    # 2kyu（仅一般）与 8段 R2100（仅特上/凤凰）：无共同卓
    players = [
        _state("2kyu", 30, rating=1600),
        _state("8dan", 1600, rating=2100),
        _state("8dan", 1600, rating=2100),
        _state("8dan", 1600, rating=2100),
    ]
    with pytest.raises(RankResolutionError):
        ranked.resolve_table(players)


def test_fixed_room_policy_uses_configured_room() -> None:
    fixed = Tenhou4pRanked(version="test", room_policy="fixed", room="houou", game_length="hanchan")
    table = fixed.resolve_table([_state("newcomer", 0, rating=1500)] * 4)
    assert table.room == "houou"
    assert table.strict is False
    assert table.positive_pt == (90, 45, 0)


def test_factory_passes_fixed_room_to_tenhou() -> None:
    system = create_rank_system(
        {
            "system": "tenhou_4p_ranked",
            "version": "2026-08-04",
            "room_policy": "fixed",
            "room": "tokujou",
            "game_length": "hanchan",
        }
    )
    assert system.system_id == "tenhou_4p_ranked"
    table = system.resolve_table([_state("newcomer", 0, rating=1500)] * 4)
    assert table.room == "tokujou"
    assert table.positive_pt == (75, 30, 0)
    update = system.apply_result(_state("newcomer", 0), placement=1, table=table)
    assert update.pt_delta == 75


def test_fixed_policy_requires_explicit_room() -> None:
    with pytest.raises(ValueError, match="fixed room policy requires"):
        Tenhou4pRanked(version="test", room_policy="fixed")


def test_table_averages_pre_match_ratings(ranked: Tenhou4pRanked) -> None:
    players = [
        _state("1dan", 200, rating=1500.0),
        _state("1dan", 200, rating=1600.0),
        _state("1dan", 200, rating=1700.0),
        _state("1dan", 200, rating=1800.0),
    ]
    table = ranked.resolve_table(players)
    assert table.avg_rating == Decimal("1650")


# --- Legacy profile compatibility -------------------------------------------

def test_legacy_profile_reproduces_historical_behavior() -> None:
    legacy = LegacyFixedProfile(rank_points=(90, 45, 0, -135))
    state = legacy.initial_state()
    assert state.rank_id == "7dan"
    assert state.pt == 1400
    assert state.rating == Decimal("1500")
    table = legacy.resolve_table([state] * 4)
    update = legacy.apply_result(state, placement=1, table=table)
    assert update.pt_delta == 90
    assert update.pt_after == 1490
    assert update.transition == "none"
    assert update.rank_after == "7dan"
    # Historical rating formula keeps exact 30.0 (no 1500 floor effect here).
    assert update.rating_after == Decimal("1530")


def test_legacy_does_not_floor_table_average() -> None:
    legacy = LegacyFixedProfile()
    state = legacy.initial_state()
    table = legacy.resolve_table([_state("7dan", 1400, rating=1500)] * 4)
    assert table.avg_rating == Decimal("1500")
    # Simulate the historical report builder path where the average came from
    # the actual pre-match ratings (may be below 1500 in theory).
    low_table = _table("houou", "hanchan", avg=1400.0)
    update = legacy.apply_result(state, placement=1, table=low_table)
    # Old formula: 1.0 * (30 + (1400-1500)/40) = 27.5 (no floor).
    assert update.rating_delta_raw == Decimal("27.5")


# --- Factory -----------------------------------------------------------------

def test_create_rank_system_factory() -> None:
    assert create_rank_system(None).system_id == "tenhou_houou_7dan_fixed"
    ranked = create_rank_system(
        {
            "system": "tenhou_4p_ranked",
            "version": "2026-08-04",
            "game_length": "hanchan",
            "room_policy": "highest_common_eligible",
            "initial_rank": "newcomer",
            "initial_rating": 1500,
        }
    )
    assert ranked.system_id == "tenhou_4p_ranked"
    assert ranked.version == "2026-08-04"
    initial = ranked.initial_state()
    assert initial.rank_id == "newcomer"
    assert initial.pt == 0
    assert initial.rating == Decimal("1500")


def test_create_rank_system_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown rank system"):
        create_rank_system({"system": "majsoul_ranked_v1", "version": "x"})
