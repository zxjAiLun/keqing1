#!/usr/bin/env python3
import sys
print("=" * 60)
print("RiichiEnv 跨目录导入测试")
print("=" * 60)

try:
    print("\n[1/6] 测试导入 riichienv...")
    from riichienv import RiichiEnv
    print("✓ 成功导入 RiichiEnv")

    print("\n[2/6] 测试导入 GameViewer...")
    from riichienv.visualizer import GameViewer
    print("✓ 成功导入 GameViewer")

    print("\n[3/6] 测试导入 RandomAgent...")
    from riichienv.agents import RandomAgent
    print("✓ 成功导入 RandomAgent")

    print("\n[4/6] 测试创建环境...")
    env = RiichiEnv(game_mode="4p-red-half")
    obs = env.reset()
    print(f"✓ 成功创建环境，当前玩家数: {len(obs)}")

    print("\n[5/6] 测试运行一步...")
    agent = RandomAgent()
    actions = {player_id: agent.act(obs) for player_id, obs in obs.items()}
    obs = env.step(actions)
    print("✓ 成功执行一步")

    print("\n[6/6] 测试GameViewer...")
    viewer = GameViewer.from_env(env)
    log_data = env.mjai_log
    print(f"✓ 成功创建GameViewer，对局日志共 {len(log_data)} 步")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！RiichiEnv 可以正常跨目录使用")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ 测试失败: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)
