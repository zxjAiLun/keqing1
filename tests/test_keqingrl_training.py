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
    run_discard_only_training,
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
    assert report.seat_rotation == (0, 1)
    assert 1.0 <= report.average_rank <= 4.0
    assert -2.0 <= report.rank_pt <= 2.0
    assert 0.0 <= report.fourth_rate <= 1.0


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
