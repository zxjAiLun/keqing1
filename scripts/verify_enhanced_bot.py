"""
验证增强型机器人的完整对局脚本

本脚本演示如何使用EnhancedBot进行一次完整对局，并展示决策追踪和解释。

运行方式:
    python scripts/verify_enhanced_bot.py

Example:
    python scripts/verify_enhanced_bot.py --player-id 0 --max-steps 100
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from riichienv import RiichiEnv
from v3model import EnhancedBot
from riichienv.agents import RandomAgent


def run_game(
    player_id: int = 0,
    max_steps: int = 100,
    verbose: bool = True,
    save_log: bool = True
):
    """
    运行一次对局
    
    Args:
        player_id: 增强型机器人控制的玩家ID
        max_steps: 最大步数
        verbose: 是否打印详细信息
        save_log: 是否保存决策日志
    
    Returns:
        game_result: 对局结果字典
    """
    if verbose:
        print("=" * 70)
        print("增强型机器人对局验证")
        print("=" * 70)
        print(f"玩家ID: {player_id}")
        print(f"最大步数: {max_steps}")
        print()
    
    # 创建环境
    env = RiichiEnv(game_mode="4p-red-single")
    
    # 创建增强型机器人
    enhanced_bot = EnhancedBot(player_id=player_id)
    
    # 创建其他玩家（使用随机机器人）
    agents = {pid: RandomAgent() for pid in range(4)}
    agents[player_id] = enhanced_bot  # 替换为增强型机器人
    
    # 初始化
    obs_dict = env.reset()
    step = 0
    
    if verbose:
        print("开始对局...")
        print("-" * 70)
    
    # 对局循环
    while not env.done() and step < max_steps:
        # 获取所有玩家的动作
        actions = {}
        for pid in range(4):
            obs = obs_dict.get(pid)
            if obs is not None:
                action = agents[pid].act(obs)
                actions[pid] = action
                
                # 如果是增强型机器人的决策，打印解释
                if pid == player_id and verbose and step < 5:
                    print(f"\n【第{step}步】玩家{player_id}的决策:")
                    explanation = enhanced_bot.get_decision_explanation()
                    if explanation:
                        print(explanation)
        
        # 执行动作
        obs_dict = env.step(actions)
        step += 1
        
        # 打印进度
        if verbose and step % 20 == 0:
            print(f"进度: {step}/{max_steps}步")
    
    # 对局结束
    if verbose:
        print("-" * 70)
        print("对局结束")
        print(f"总步数: {step}")
        print(f"最终得分: {env.scores()}")
        print(f"最终排名: {env.ranks()}")
    
    # 获取决策摘要
    summary = enhanced_bot.tracker.generate_summary()
    
    if verbose:
        print("-" * 70)
        print("决策摘要:")
        print(f"  总决策数: {summary['total_decisions']}")
        print(f"  立直决策数: {summary['riichi_count']}")
        if summary['avg_shanten'] is not None:
            print(f"  平均向听数: {summary['avg_shanten']:.2f}")
        if summary['min_shanten'] is not None:
            print(f"  最小向听数: {summary['min_shanten']}")
    
    # 保存决策日志
    if save_log:
        log_path = Path(__file__).parent.parent / 'logs' / 'decision_log.json'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_bot.save_decision_log(log_path)
        
        if verbose:
            print(f"\n决策日志已保存到: {log_path}")
    
    # 构建结果
    result = {
        'total_steps': step,
        'scores': env.scores(),
        'ranks': env.ranks(),
        'summary': summary,
        'log_path': str(log_path) if save_log else None
    }
    
    if verbose:
        print("=" * 70)
        print("对局验证完成")
        print("=" * 70)
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证增强型机器人')
    parser.add_argument('--player-id', type=int, default=0, help='玩家ID (0-3)')
    parser.add_argument('--max-steps', type=int, default=100, help='最大步数')
    parser.add_argument('--verbose', action='store_true', default=True, help='打印详细信息')
    parser.add_argument('--no-save', action='store_true', help='不保存决策日志')
    
    args = parser.parse_args()
    
    # 运行对局
    result = run_game(
        player_id=args.player_id,
        max_steps=args.max_steps,
        verbose=args.verbose,
        save_log=not args.no_save
    )
    
    # 打印最终结果
    print("\n最终结果:")
    print(f"  得分: {result['scores']}")
    print(f"  排名: {result['ranks']}")


if __name__ == '__main__':
    main()
