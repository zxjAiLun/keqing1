"""
测试 keqingv3 核心：MC 终局回传的 value_target 计算。

设计依据：docs/keqingv3_design.md
- 替换启发式局部信号为终局得分折扣回传
- gamma = 0.99, VALUE_NORM = 30000.0
- value_target = (score_delta / VALUE_NORM) * (gamma ** steps_remaining)
"""

import pytest
from mahjong_env.replay import MCReturnValueStrategy, PendingValueSample, ReplaySample


def _pending_sample(actor: int, step_index: int) -> PendingValueSample:
    return PendingValueSample(
        sample=ReplaySample(
            state={},
            actor=actor,
            actor_name=f"p{actor}",
            label_action={"type": "dahai", "actor": actor},
            legal_actions=[],
            value_target=0.0,
        ),
        round_step_index=step_index,
    )


class TestMCReturnValueStrategy:
    def test_final_step_value_target_equals_full_score_delta(self):
        """终局步骤 steps_remaining=0，gamma^0=1，value_target = score_delta / VALUE_NORM"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        pending = [_pending_sample(0, step_index=100)]
        terminal = {"type": "hora", "actor": 0, "deltas": [30000, -10000, -10000, -10000]}

        strategy.finalize_round(pending, terminal)

        # p0 和牌 +30000，/30000 = 1.0
        assert pending[0].sample.value_target == pytest.approx(1.0)

    def test_early_step_value_target_is_discounted(self):
        """早期步骤的价值目标应按 gamma 折扣，越早越小"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # last_step = 100, p0 在 step=0，steps_remaining = 100 - 0 = 100
        pending = [_pending_sample(0, step_index=0), _pending_sample(1, step_index=100)]
        terminal = {"type": "hora", "actor": 0, "deltas": [30000, -10000, -10000, -10000]}

        strategy.finalize_round(pending, terminal)

        # 30000/30000 * 0.99^100 ≈ 0.366
        expected = (30000 / 30000.0) * (0.99 ** 100)
        assert pending[0].sample.value_target == pytest.approx(expected, rel=1e-3)

    def test_multi_step_discount_progression(self):
        """验证 gamma 折扣在多步上的递进关系"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # 同一终局，最后样本在 step=99
        pending = [
            _pending_sample(0, step_index=50),
            _pending_sample(0, step_index=75),
            _pending_sample(0, step_index=99),  # last_step = 99
        ]
        terminal = {"type": "hora", "actor": 0, "deltas": [30000, -10000, -10000, -10000]}

        strategy.finalize_round(pending, terminal)

        gamma = 0.99
        # steps_remaining = last_step - step_index
        expected_50 = (30000 / 30000.0) * (gamma ** (99 - 50))
        expected_75 = (30000 / 30000.0) * (gamma ** (99 - 75))
        expected_99 = (30000 / 30000.0) * (gamma ** (99 - 99))

        assert pending[0].sample.value_target == pytest.approx(expected_50, rel=1e-3)
        assert pending[1].sample.value_target == pytest.approx(expected_75, rel=1e-3)
        assert pending[2].sample.value_target == pytest.approx(expected_99, rel=1e-3)
        # 越晚的样本，折扣越小
        assert pending[0].sample.value_target < pending[1].sample.value_target
        assert pending[1].sample.value_target < pending[2].sample.value_target

    def test_negative_score_delta_for_dealin(self):
        """放铳样本的 value_target 应为负值"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # last_step = 100, p2 在 step_index=80, steps_remaining=20
        pending = [_pending_sample(2, step_index=80), _pending_sample(0, step_index=100)]
        terminal = {"type": "hora", "actor": 1, "target": 2, "deltas": [1000, 8000, -8000, -1000]}

        strategy.finalize_round(pending, terminal)

        # p2 放铳 -8000
        assert pending[0].sample.value_target == pytest.approx(-8000 / 30000.0 * (0.99 ** 20), rel=1e-3)

    def test_different_actors_get_correct_score_deltas(self):
        """不同玩家的 score_delta 应正确映射到各自的 value_target"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # last_step = 100, 所有样本在 step_index=50, steps_remaining=50
        pending = [_pending_sample(i, step_index=50) for i in range(4)]
        pending.append(_pending_sample(0, step_index=100))  # 添加 last sample
        terminal = {
            "type": "hora",
            "actor": 1,
            "target": 3,
            "deltas": [-1000, 12000, 0, -11000],
        }

        strategy.finalize_round(pending, terminal)

        gamma_50 = 0.99 ** 50
        assert pending[0].sample.value_target == pytest.approx(-1000 / 30000.0 * gamma_50, rel=1e-3)
        assert pending[1].sample.value_target == pytest.approx(12000 / 30000.0 * gamma_50, rel=1e-3)
        assert pending[2].sample.value_target == pytest.approx(0.0, rel=1e-3)
        assert pending[3].sample.value_target == pytest.approx(-11000 / 30000.0 * gamma_50, rel=1e-3)

    def test_ryukyoku_distributes_value_correctly(self):
        """流局时各玩家按 deltas 获取正确的 value_target"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # last_step = 100, 所有样本在 step_index=60, steps_remaining=40
        pending = [_pending_sample(i, step_index=60) for i in range(4)]
        pending.append(_pending_sample(0, step_index=100))  # 添加 last sample
        terminal = {
            "type": "ryukyoku",
            "deltas": [5000, -5000, 5000, -5000],
        }

        strategy.finalize_round(pending, terminal)

        gamma_40 = 0.99 ** 40
        assert pending[0].sample.value_target == pytest.approx(5000 / 30000.0 * gamma_40, rel=1e-3)
        assert pending[1].sample.value_target == pytest.approx(-5000 / 30000.0 * gamma_40, rel=1e-3)
        assert pending[2].sample.value_target == pytest.approx(5000 / 30000.0 * gamma_40, rel=1e-3)
        assert pending[3].sample.value_target == pytest.approx(-5000 / 30000.0 * gamma_40, rel=1e-3)

    def test_no_terminal_event_does_not_modify_targets(self):
        """无终局事件时不应修改任何 value_target"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        pending = [_pending_sample(0, step_index=50)]
        pending[0].sample.value_target = 0.42  # 预设值

        strategy.finalize_round(pending, None)

        # 不应被修改
        assert pending[0].sample.value_target == 0.42

    def test_invalid_deltas_format_does_not_crash(self):
        """无效的 deltas 格式应安全处理"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        pending = [_pending_sample(0, step_index=50)]

        # score_delta 而不是 deltas
        strategy.finalize_round(pending, {"type": "hora", "actor": 0})

        # 不应崩溃，value_target 保持初始值 0.0
        assert pending[0].sample.value_target == 0.0

    def test_large_handicap_score_delta(self):
        """满贯/役满级别的大幅得分应有明显的 value_target"""
        strategy = MCReturnValueStrategy(gamma=0.99, value_norm=30000.0)
        # last_step = 100, p0 在 step_index=90, steps_remaining=10
        pending = [_pending_sample(0, step_index=90), _pending_sample(1, step_index=100)]
        # 役满 +64000
        terminal = {"type": "hora", "actor": 0, "deltas": [64000, -32000, -32000, 0]}

        strategy.finalize_round(pending, terminal)

        # gamma^10 ≈ 0.904
        expected = (64000 / 30000.0) * (0.99 ** 10)
        assert pending[0].sample.value_target == pytest.approx(expected, rel=1e-3)
        assert pending[0].sample.value_target > 1.5  # 役满应明显超过 1.0

    def test_custom_gamma_and_norm(self):
        """自定义 gamma 和 value_norm 应正确生效"""
        strategy = MCReturnValueStrategy(gamma=0.95, value_norm=40000.0)
        # last_step = 100, p0 在 step_index=20, steps_remaining=80
        pending = [_pending_sample(0, step_index=20), _pending_sample(1, step_index=100)]
        terminal = {"type": "hora", "actor": 0, "deltas": [40000, -10000, -20000, -10000]}

        strategy.finalize_round(pending, terminal)

        # 40000/40000 * 0.95^80
        expected = (40000 / 40000.0) * (0.95 ** 80)
        assert pending[0].sample.value_target == pytest.approx(expected, rel=1e-3)
