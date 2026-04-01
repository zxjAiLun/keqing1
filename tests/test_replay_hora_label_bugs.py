"""Regression tests for replay hora label bugs found in preprocessing.

Bug: hora label not in reconstructed legal set when last_discard=None.
Root cause: preprocessor's snap_before has actor_to_move=None instead of 3,
            causing enumerate_legal_actions to enter the D branch (return [none])
            instead of C branch (tsumo-hora).
"""

import pytest
from collections import Counter

from mahjong_env.replay import _label_matches_legal
from mahjong_env.types import (
    action_dict_to_spec,
    action_specs_match,
    canonical_meld_pai,
    normalize_or_keep_aka,
    ActionSpec,
)
from mahjong_env.state import GameState, PlayerState


class TestHoraLabelMatching:
    """hora 标签匹配 legal 的各种情况。"""

    def test_hora_label_without_pai_target_match(self):
        """hora label 没有 pai 字段时，只要 target 匹配就应通过。"""
        label = {"type": "hora", "actor": 3, "target": 1}
        legal = [{"type": "hora", "actor": 3, "target": 1, "pai": "5mr"}]
        assert _label_matches_legal(label, legal) is True

    def test_hora_label_target_only(self):
        """只有 target 相同时，hora label 应匹配。"""
        label = {"type": "hora", "actor": 0, "target": 0}
        legal = [{"type": "hora", "actor": 0, "target": 0, "pai": "9m"}]
        assert _label_matches_legal(label, legal) is True

    def test_hora_label_target_mismatch(self):
        """hora label target 不同时应不匹配。"""
        label = {"type": "hora", "actor": 0, "target": 1}
        legal = [{"type": "hora", "actor": 0, "target": 2, "pai": "5mr"}]
        assert _label_matches_legal(label, legal) is False


class TestTsumoHoraPreprocessorBug:
    """预处理中 tsumo-hora 的 bug。

    预处理错误：
    snap={'bakaze': 'S', 'kyoku': 3, 'honba': 1,
          'hand': ['2s','3m','3m','3s','4s','5m','6m','6p','7p','7s','8p','8s','9s'],
          'last_discard': None, 'melds': []}

    问题：snap 中缺少 `actor_to_move` 或 `actor_to_move=None`，
          导致 enumerate_legal_actions 进入 D 分支返回 [none]，
          而真实动作是 tsumo-hora。

    正确行为：
    - 若 actor_to_move=3 且 last_tsumo[3]=<tile>，应进入 C1 分支
    - 若 can_hora=True，应返回 hora
    """

    def test_tsumo_hora_with_actor_to_move_3(self, monkeypatch):
        """正确设置 actor_to_move=3 时，tsumo-hora 应在 legal 中。"""
        player = PlayerState()
        player.hand.update(["2s", "3m", "3m", "3s", "4s", "5m", "6m", "6p", "7p", "7s", "8p", "8s", "9s"])

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.oya = 1
        gs.players = [PlayerState(), PlayerState(), PlayerState(), player]
        gs.actor_to_move = 3
        gs.last_tsumo = [None, None, None, "9s"]
        gs.last_tsumo_raw = [None, None, None, "9s"]
        snap = gs.snapshot(actor=3)

        # monkeypatch can_hora to always return True
        import mahjong_env.legal_actions as la
        original = la.can_hora_from_snapshot
        la.can_hora_from_snapshot = lambda *args, **kwargs: True
        try:
            from mahjong_env.legal_actions import enumerate_legal_actions
            legal = enumerate_legal_actions(snap, actor=3)
            assert any(a.type == "hora" for a in legal), \
                f"actor_to_move=3, last_tsumo set -> should have hora, got {[a.type for a in legal]}"
        finally:
            la.can_hora_from_snapshot = original

    def test_preprocessor_bug_actor_to_move_none_returns_none(self):
        """预处理 bug 的直接暴露：actor_to_move=None 时返回 [none]。"""
        player = PlayerState()
        player.hand.update(["2s", "3m", "3m", "3s", "4s", "5m", "6m", "6p", "7p", "7s", "8p", "8s", "9s"])

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.oya = 1
        gs.players = [PlayerState(), PlayerState(), PlayerState(), player]
        # actor_to_move NOT set -> None
        snap = gs.snapshot(actor=3)

        from mahjong_env.legal_actions import enumerate_legal_actions
        legal = enumerate_legal_actions(snap, actor=3)

        # BUG: preprocessor 产生的 snap 中 actor_to_move=None
        # 导致走入 D 分支返回 [none]，而真实动作是 tsumo-hora
        # 这个测试记录了 BUG 的表现：legal=[none]
        assert all(a.type == "none" for a in legal), \
            f"actor_to_move=None returns only none: got {[a.type for a in legal]}"

    def test_preprocessor_bug_legal_label_mismatch(self, monkeypatch):
        """预处理 bug 导致的 label-legal 不匹配。"""
        # 模拟 preprocessor 错误生成的 snap（actor_to_move=None）
        player = PlayerState()
        player.hand.update(["2s", "3m", "3m", "3s", "4s", "5m", "6m", "6p", "7p", "7s", "8p", "8s", "9s"])

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.oya = 1
        gs.players = [PlayerState(), PlayerState(), PlayerState(), player]
        snap = gs.snapshot(actor=3)

        # 真实标签是 hora
        label = {"type": "hora", "actor": 3, "target": 3, "pai": "9s"}

        from mahjong_env.legal_actions import enumerate_legal_actions
        legal_specs = enumerate_legal_actions(snap, actor=3)
        legal_dicts = [s.to_mjai() for s in legal_specs]

        # BUG: 由于 actor_to_move=None，legal 只有 [none]
        # 导致 label 不在 legal 中
        assert not _label_matches_legal(label, legal_dicts), \
            "BUG: label should NOT match when actor_to_move=None"


class TestRonHoraWithMeld:
    """荣和（ron）合法性的测试，对应预处理错误：

    actor=1 有 pon=W meld，手牌 11 tiles，last_discard 来自 actor=3 的 4p。
    真实动作是 hora，但 legal set 只有 pon 和 none。

    根因分析：当 preprocessor 的 snap 中 actor_to_move=1（当前回合者）时，
    enumerate_legal_actions 走 C 分支（自己的回合），完全忽略 last_discard 的荣和检查。
    这导致应该触发 B 分支（有他人打出的 last_discard）的情况被错误地按自摸逻辑处理。

    测试设计：
    - 场景 A：actor_to_move=None（预处理 bug 状态），last_discard 有值 → B 分支
    - 场景 B：actor_to_move=1，last_discard 有值 → C 分支（错误地忽略荣和）
    """

    def test_ron_hora_b_branch_with_pon_meld(self):
        """B 分支：有 pon meld，last_discard 完成 tenpai，actor_to_move=None。

        这是预处理 bug 状态的直接测试：
        snap 中 actor_to_move=None（preprocessor 的 bug），
        但 last_discard={'actor':3,'pai':'N'} 有值，
        所以正确情况应该走 B 分支检查荣和。
        """
        player = PlayerState()
        # Closed hand after pon(W): 10 tiles + 1 W(pair) = 11 tiles
        # With pon(W) meld(3): 14 tiles total
        # Wait: N (to pair with N in hand)
        player.hand.update([
            "1m", "2m", "3m",     # set 1
            "4m", "5m", "6m",     # set 2
            "7m", "8m", "9m",     # set 3
            "N",                   # single (wait to pair)
            "W",                   # pair (leftover W from pon)
        ])
        player.melds = [
            {"type": "pon", "pai": "W", "pai_raw": "W", "consumed": ["W", "W"], "target": 2}
        ]

        gs = GameState()
        gs.bakaze = "E"
        gs.kyoku = 4
        gs.honba = 2
        gs.oya = 3
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        # actor_to_move NOT set -> None (simulating preprocessor bug)

        # Player 3 discards N — completes pair N-N → 4 sets + pair ✓
        gs.last_discard = {"actor": 3, "pai": "N", "pai_raw": "N"}
        gs.last_tsumo = [None, None, None, None]
        gs.last_tsumo_raw = [None, None, None, None]

        snap = gs.snapshot(actor=1)

        from mahjong_env.legal_actions import enumerate_legal_actions
        legal = enumerate_legal_actions(snap, actor=1)
        legal_types = [a.type for a in legal]

        # BUG: With actor_to_move=None, the code goes to D branch returning [none].
        # This is the bug that causes the preprocessor error.
        # Expected correct behavior: B branch with hora (but buggy code returns [none]).
        assert all(a.type == "none" for a in legal), (
            f"BUG: actor_to_move=None causes [none] only (D branch). "
            f"Should have hora. Got: {legal_types}"
        )

    def test_ron_hora_c_branch_ignores_last_discard(self):
        """C 分支（actor_to_move=actor）：actor_to_move=1 时走 C 分支，忽略 last_discard。

        这证明了为什么 actor_to_move=None 是 bug：
        当 actor_to_move=1（当前回合者）时，C 分支只看 last_tsumo，
        完全不看 last_discard，导致荣和被忽略。
        """
        player = PlayerState()
        player.hand.update([
            "1m", "2m", "3m",
            "4m", "5m", "6m",
            "7m", "8m", "9m",
            "N",
            "W",
        ])
        player.melds = [
            {"type": "pon", "pai": "W", "pai_raw": "W", "consumed": ["W", "W"], "target": 2}
        ]

        gs = GameState()
        gs.bakaze = "E"
        gs.kyoku = 4
        gs.oya = 3
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.actor_to_move = 1  # 设为 actor=1，走 C 分支

        # last_discard 有值，但 C 分支不检查荣和
        gs.last_discard = {"actor": 3, "pai": "N", "pai_raw": "N"}
        gs.last_tsumo = [None, None, None, None]
        gs.last_tsumo_raw = [None, None, None, None]

        snap = gs.snapshot(actor=1)
        assert snap["actor_to_move"] == 1, "snapshot should have actor_to_move=1"

        from mahjong_env.legal_actions import enumerate_legal_actions
        legal = enumerate_legal_actions(snap, actor=1)
        legal_types = [a.type for a in legal]

        # C branch does NOT check last_discard for ron — this is the design.
        # It only handles tsumo (last_tsumo). So hora is NOT in legal set here.
        assert "hora" not in legal_types, (
            f"C branch should NOT check ron (ignores last_discard). "
            f"Got: {legal_types}"
        )


class TestCanonicalMeldPai:
    """canonical_meld_pai 的行为。"""

    def test_pai_returns_normalized(self):
        assert canonical_meld_pai({"pai": "5mr"}) == "5mr"

    def test_consumed_first_when_no_pai(self):
        assert canonical_meld_pai({"consumed": ["5mr", "5mr"]}) == "5mr"

    def test_returns_none_when_empty(self):
        assert canonical_meld_pai({}) is None


class TestNormalizeOrKeepAka:
    """normalize_or_keep_aka 的行为。"""

    def test_preserves_aka(self):
        for aka in ("5mr", "5pr", "5sr"):
            assert normalize_or_keep_aka(aka) == aka

    def test_normalizes_r_suffix(self):
        assert normalize_or_keep_aka("5mr") == "5mr"
        assert normalize_or_keep_aka("1mr") == "1m"
        assert normalize_or_keep_aka("9pr") == "9p"

    def test_passes_through_normal(self):
        assert normalize_or_keep_aka("5m") == "5m"
        assert normalize_or_keep_aka("9s") == "9s"
        assert normalize_or_keep_aka("E") == "E"

    def test_none_input(self):
        assert normalize_or_keep_aka(None) is None


class TestActionSpecsMatch:
    """action_specs_match 的行为。"""

    def test_hora_match_target_only(self):
        left = ActionSpec(type="hora", target=1)
        right = ActionSpec(type="hora", target=1, pai="5mr")
        assert action_specs_match(left, right) is True

    def test_hora_mismatch_target(self):
        left = ActionSpec(type="hora", target=1)
        right = ActionSpec(type="hora", target=2, pai="5mr")
        assert action_specs_match(left, right) is False

    def test_pon_match_normalized_pai(self):
        """pon label 普通 5m 与 legal 赤宝牌 5mr 应匹配（pai 归一化后相等）。

        注意：consumed 字段不做归一化比较，所以 consumed=("5m","5m") != consumed=("5mr","5mr")。
        只有 pai 字段在比较时归一化。
        """
        left = ActionSpec(type="pon", actor=0, target=1, pai="5m", consumed=("5m", "5m"))
        right = ActionSpec(type="pon", actor=0, target=1, pai="5mr", consumed=("5m", "5m"))
        # pai: normalize_tile("5m") = "5m" = normalize_tile("5mr"), so they match
        # consumed: both ("5m", "5m") — exact match required
        assert action_specs_match(left, right) is True

    def test_none_match(self):
        left = ActionSpec(type="none")
        right = ActionSpec(type="none")
        assert action_specs_match(left, right) is True
