from __future__ import annotations

import torch

from keqingrl import (
    ActionType,
    DiscardOnlyMahjongEnv,
    NeuralInteractivePolicy,
    OpponentPool,
    OpponentPoolEntry,
    RandomInteractivePolicy,
    evaluate_discard_only_policy,
    evaluate_policy,
    measure_latency_smoke,
    run_fixed_seed_evaluation_smoke,
    run_critic_pretrain_smoke,
    run_discard_only_ppo_smoke,
    run_discard_only_training,
    run_zero_delta_selfplay_smoke,
    run_training,
)


def test_evaluate_discard_only_policy_returns_finite_metrics() -> None:
    torch.manual_seed(0)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )

    metrics = evaluate_discard_only_policy(
        env,
        policy,
        num_episodes=2,
        seed=81,
    )

    assert metrics.episode_count == 2
    assert metrics.total_steps > 0
    assert 1.0 <= metrics.mean_rank <= 4.0
    assert 0.0 <= metrics.first_place_rate <= 1.0
    assert 0.0 <= metrics.fourth_place_rate <= 1.0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert 0.0 <= metrics.deal_in_rate <= 1.0
    assert 0.0 <= metrics.call_rate <= 1.0
    assert 0.0 <= metrics.riichi_rate <= 1.0
    assert metrics.illegal_action_rate == 0.0
    assert metrics.fallback_rate == 0.0
    assert metrics.forced_terminal_missed == 0
    assert isinstance(metrics.terminal_reason_count, dict)


def test_run_discard_only_training_builds_iteration_history() -> None:
    torch.manual_seed(1)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = run_discard_only_training(
        env,
        policy,
        optimizer,
        num_iterations=2,
        episodes_per_iteration=1,
        update_epochs=1,
        eval_episodes=1,
        seed=91,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert len(history.iterations) == 2
    assert [iteration.iteration for iteration in history.iterations] == [0, 1]
    assert [iteration.policy_version for iteration in history.iterations] == [0, 1]
    assert history.iterations[0].train_metrics.episode_count == 1
    assert history.iterations[0].eval_metrics is not None
    assert 1.0 <= history.iterations[0].eval_metrics.mean_rank <= 4.0


def test_run_discard_only_training_accepts_expanded_self_turn_scope() -> None:
    torch.manual_seed(2)
    env = DiscardOnlyMahjongEnv(
        max_kyokus=1,
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.REACH_DISCARD,
            ActionType.TSUMO,
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.RYUKYOKU,
        ),
        response_action_types=(
            ActionType.RON,
            ActionType.CHI,
            ActionType.PON,
            ActionType.DAIMINKAN,
            ActionType.PASS,
        ),
    )
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = run_discard_only_training(
        env,
        policy,
        optimizer,
        num_iterations=1,
        episodes_per_iteration=1,
        update_epochs=1,
        eval_episodes=1,
        seed=101,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert len(history.iterations) == 1
    train_metrics = history.iterations[0].train_metrics
    eval_metrics = history.iterations[0].eval_metrics
    assert train_metrics.episode_count == 1
    assert train_metrics.total_steps > 0
    assert torch.isfinite(torch.tensor(train_metrics.mean_total_loss))
    assert torch.isfinite(torch.tensor(train_metrics.mean_policy_loss))
    assert torch.isfinite(torch.tensor(train_metrics.mean_value_loss))
    assert eval_metrics is not None
    assert eval_metrics.episode_count == 1
    assert torch.isfinite(torch.tensor(eval_metrics.mean_terminal_reward))
    assert torch.isfinite(torch.tensor(eval_metrics.mean_rank))


def test_evaluate_policy_accepts_opponent_pool() -> None:
    torch.manual_seed(3)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    opponent_pool = OpponentPool(
        (OpponentPoolEntry(policy=RandomInteractivePolicy(), policy_version=17, greedy=True),)
    )

    metrics = evaluate_policy(
        env,
        policy,
        num_episodes=1,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        seed=111,
    )

    assert metrics.episode_count == 1
    assert metrics.total_steps > 0
    assert 1.0 <= metrics.mean_rank <= 4.0
    assert torch.isfinite(torch.tensor(metrics.mean_terminal_reward))


def test_fixed_seed_evaluation_smoke_reports_seat_rotation() -> None:
    torch.manual_seed(4)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )

    report = run_fixed_seed_evaluation_smoke(
        env,
        policy,
        num_games=1,
        seed=117,
        seat_rotation=(0, 1),
    )

    assert report.episode_count == 2
    assert report.fixed_seed_count == 1
    assert report.games_per_seed == 2
    assert report.seat_rotation == (0, 1)
    assert report.seat_rotation_enabled is True
    assert report.policy_mode == "greedy"
    assert report.opponent_name == "rule_prior_greedy"
    assert report.reuse_training_rollout is False
    assert 1.0 <= report.average_rank <= 4.0
    assert -2.0 <= report.rank_pt <= 2.0
    assert 0.0 <= report.fourth_rate <= 1.0
    assert 0.0 <= report.win_rate <= 1.0
    assert 0.0 <= report.deal_in_rate <= 1.0
    assert 0.0 <= report.call_rate <= 1.0
    assert 0.0 <= report.riichi_rate <= 1.0
    assert report.illegal_action_rate == 0.0
    assert report.fallback_rate == 0.0
    assert report.forced_terminal_missed == 0
    assert isinstance(report.terminal_reason_count, dict)
    assert report.passed_smoke_checks is True
    assert report.failure_reasons == ()
    assert set(report.per_seat) == {0, 1}
    assert report.per_seat[0].episode_count == 1
    assert report.per_seat[1].episode_count == 1



def test_fixed_seed_evaluation_smoke_rejects_training_rollout_reuse() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )

    try:
        run_fixed_seed_evaluation_smoke(
            env,
            policy,
            num_games=1,
            seed=118,
            reuse_training_rollout=True,
        )
    except ValueError as exc:
        assert "must not reuse training rollouts" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected fixed-seed eval to reject training rollout reuse")

def test_latency_smoke_reports_positive_rates() -> None:
    torch.manual_seed(5)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )

    report = measure_latency_smoke(env, policy, num_games=1, seed=119)

    assert report.decision_count > 0
    assert report.game_count == 1
    assert report.decisions_per_sec > 0.0
    assert report.games_per_sec > 0.0
    assert report.avg_decision_latency_ms > 0.0
    assert report.p95_decision_latency_ms > 0.0


def test_zero_delta_selfplay_smoke_checks_rule_prior_equivalence() -> None:
    torch.manual_seed(7)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)

    report = run_zero_delta_selfplay_smoke(
        env,
        greedy_episodes=1,
        sample_episodes=1,
        seed=123,
    )

    assert report.greedy_episode_count == 1
    assert report.sample_episode_count == 1
    assert report.learner_step_count > 0
    assert report.autopilot_step_count > 0
    assert report.illegal_action_rate == 0.0
    assert report.fallback_rate == 0.0
    assert report.forced_terminal_missed == 0
    assert report.old_log_prob_finite is True
    assert report.entropy_finite is True
    assert report.neural_delta_abs_mean == 0.0
    assert report.neural_delta_abs_max == 0.0
    assert report.rule_kl_mean <= 1e-3
    assert report.final_logits_max_abs_diff == 0.0
    assert report.probs_max_abs_diff == 0.0
    assert report.action_order_valid is True
    assert report.metadata_strict_valid is True
    assert report.autopilot_policy_fields_null is True
    assert report.autopilot_rows_excluded_from_ppo is True


def test_critic_pretrain_smoke_preserves_actor_outputs() -> None:
    torch.manual_seed(8)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)

    report = run_critic_pretrain_smoke(
        env,
        episodes=2,
        pretrain_steps=2,
        seed=131,
    )

    assert report.episode_count == 2
    assert report.batch_size > 0
    assert report.pretrain_steps == 2
    assert torch.isfinite(torch.tensor(report.value_loss))
    assert report.rank_loss is not None
    assert torch.isfinite(torch.tensor(report.rank_loss))
    assert torch.isfinite(torch.tensor(report.explained_variance))
    assert report.rank_acc is not None
    assert 0.0 <= report.rank_acc <= 1.0
    assert report.grad_norm_by_module
    assert report.trainable_param_names
    assert all(name.startswith(("value_head.", "rank_head.")) for name in report.trainable_param_names)
    assert all(name.startswith(("value_head.", "rank_head.")) for name in report.optimizer_param_names)
    assert report.actor_logits_before_after_diff == 0.0
    assert report.actor_probs_before_after_diff == 0.0
    assert report.neural_delta_before_after_diff == 0.0
    assert report.optimizer_actor_param_count == 0



def test_discard_only_ppo_smoke_runs_and_records_metrics() -> None:
    torch.manual_seed(9)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)

    report = run_discard_only_ppo_smoke(
        env,
        iterations=1,
        rollout_episodes_per_iter=1,
        update_epochs=1,
        seed=141,
        lr=1e-4,
        clip_eps=0.1,
        entropy_coef=0.005,
        rule_kl_coef=0.02,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert report.iteration_count == 1
    assert report.rollout_episodes_per_iter == 1
    assert report.update_epochs == 1
    assert report.stopped_early is False
    assert report.stop_reason is None
    assert report.final_neural_delta_abs_mean >= 0.0
    assert report.final_neural_delta_abs_max >= 0.0
    iteration = report.iterations[0]
    assert iteration.episode_count == 1
    assert iteration.batch_size > 0
    assert torch.isfinite(torch.tensor(iteration.policy_loss))
    assert torch.isfinite(torch.tensor(iteration.value_loss))
    assert iteration.rank_loss is not None
    assert torch.isfinite(torch.tensor(iteration.rank_loss))
    assert iteration.entropy > 0.0
    assert iteration.approx_kl >= 0.0
    assert iteration.clip_fraction >= 0.0
    assert torch.isfinite(torch.tensor(iteration.ratio_mean))
    assert torch.isfinite(torch.tensor(iteration.ratio_std))
    assert iteration.rule_kl is not None
    assert torch.isfinite(torch.tensor(iteration.rule_kl))
    assert iteration.rule_agreement is not None
    assert 0.0 <= iteration.rule_agreement <= 1.0
    assert torch.isfinite(torch.tensor(iteration.advantage_mean))
    assert torch.isfinite(torch.tensor(iteration.advantage_std))
    assert torch.isfinite(torch.tensor(iteration.return_mean))
    assert iteration.grad_norm >= 0.0
    assert iteration.grad_norm_by_module
    assert iteration.neural_delta_abs_mean >= 0.0
    assert iteration.neural_delta_abs_max >= 0.0
    assert 0.0 <= iteration.top1_action_changed_rate <= 1.0
    assert 1.0 <= iteration.avg_rank <= 4.0
    assert 0.0 <= iteration.first_rate <= 1.0
    assert 0.0 <= iteration.fourth_rate <= 1.0
    assert 0.0 <= iteration.win_rate <= 1.0
    assert 0.0 <= iteration.deal_in_rate <= 1.0
    assert 0.0 <= iteration.call_rate <= 1.0
    assert 0.0 <= iteration.riichi_rate <= 1.0
    assert iteration.illegal_action_rate == 0.0
    assert iteration.fallback_rate == 0.0
    assert iteration.forced_terminal_missed == 0
    assert isinstance(iteration.terminal_reason_count, dict)

def test_run_training_accepts_opponent_pool() -> None:
    torch.manual_seed(6)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    opponent_pool = OpponentPool(
        (OpponentPoolEntry(policy=RandomInteractivePolicy(), policy_version=23, greedy=True),)
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = run_training(
        env,
        policy,
        optimizer,
        num_iterations=1,
        episodes_per_iteration=1,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        update_epochs=1,
        eval_episodes=1,
        seed=121,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert len(history.iterations) == 1
    train_metrics = history.iterations[0].train_metrics
    eval_metrics = history.iterations[0].eval_metrics
    assert train_metrics.episode_count == 1
    assert train_metrics.batch_size > 0
    assert train_metrics.total_steps >= train_metrics.batch_size
    assert torch.isfinite(torch.tensor(train_metrics.mean_total_loss))
    assert eval_metrics is not None
    assert eval_metrics.episode_count == 1
    assert torch.isfinite(torch.tensor(eval_metrics.mean_terminal_reward))
