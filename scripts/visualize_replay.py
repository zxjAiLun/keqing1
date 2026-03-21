"""
可视化对局回放工具

提供基于终端的麻将对局可视化回放，支持：
- 实时显示对局进度
- 显示手牌和弃牌
- 显示得分和排名
- 显示关键决策点
- 决策解释展示

运行方式:
    python scripts/visualize_replay.py

Example:
    python scripts/visualize_replay.py --player-id 0 --max-steps 50
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from riichienv import RiichiEnv
from v3model import EnhancedBot
from riichienv.agents import RandomAgent


@dataclass
class TileDisplay:
    """牌面显示工具"""
    
    # 牌面字符映射
    TILES = {
        '1m': '🀇', '2m': '🀈', '3m': '🀉', '4m': '🀊', '5m': '🀋', '6m': '🀌', '7m': '🀍', '8m': '🀎', '9m': '🀏',
        '1p': '🀙', '2p': '🀚', '3p': '🀛', '4p': '🀜', '5p': '🀝', '6p': '🀞', '7p': '🀟', '8p': '🀠', '9p': '🀡',
        '1s': '🀐', '2s': '🀑', '3s': '🀒', '4s': '🀓', '5s': '🀔', '6s': '🀕', '7s': '🀖', '8s': '🀗', '9s': '🀘',
        '1z': '🀀', '2z': '🀁', '3z': '🀂', '4z': '🀃', '5z': '🀆', '6z': '🀅', '7z': '🀇'
    }
    
    @classmethod
    def mpsz_to_emoji(cls, tile: str) -> str:
        """将MPSZ格式转换为emoji"""
        # 处理赤5牌
        if tile == '5mr':
            return '🀋'  # 红5m
        elif tile == '5pr':
            return '🀝'  # 红5p
        elif tile == '5sr':
            return '🀔'  # 红5s
        return cls.TILES.get(tile, tile)
    
    @classmethod
    def format_hand(cls, hand) -> str:
        """格式化手牌（支持int或str格式）"""
        formatted = []
        for tile in hand:
            # 转换为字符串
            if isinstance(tile, int):
                # 34牌格式转MPSZ格式
                if tile < 9:
                    formatted.append(f"{tile+1}m")
                elif tile < 18:
                    formatted.append(f"{tile-8}p")
                elif tile < 27:
                    formatted.append(f"{tile-17}s")
                else:
                    formatted.append(f"{tile-26}z")
            else:
                formatted.append(str(tile))
        return ' '.join([cls.mpsz_to_emoji(t) for t in formatted])


class GameVisualizer:
    """
    对局可视化器
    
    在终端中显示对局进程
    """
    
    def __init__(self, enhanced_bot: Optional[EnhancedBot] = None):
        self.enhanced_bot = enhanced_bot
        self.step_count = 0
    
    def print_header(self, game_mode: str):
        """打印标题"""
        print("=" * 80)
        print("🎴 立直麻将对局可视化回放".center(80))
        print("=" * 80)
        print(f"模式: {game_mode}")
        print()
    
    def print_scores(self, scores: List[int], ranks: List[int]):
        """打印得分"""
        print("\n📊 当前得分:")
        player_info = list(zip(scores, ranks))
        sorted_info = sorted(player_info, key=lambda x: x[1])
        
        for i, (score, rank) in enumerate(sorted_info):
            player_id = list(scores).index(score) if score in scores else 0
            rank_symbol = ['🥇', '🥈', '🥉', '4️⃣'][rank - 1] if rank <= 4 else f'{rank}️⃣'
            print(f"  {rank_symbol} P{player_id}: {score:>6}点")
    
    def print_player_hand(self, player_id: int, hand: List[str], is_enhanced: bool = False):
        """打印玩家手牌"""
        print(f"\n🧤 P{player_id}的手牌:")
        print(f"  {TileDisplay.format_hand(hand)}")
        
        # 如果使用增强型机器人，显示决策信息
        if self.enhanced_bot and player_id == self.enhanced_bot.player_id:
            features = self.enhanced_bot.get_last_features()
            if features:
                print(f"\n📈 决策信息:")
                print(f"  向听数: {features.shanten}")
                print(f"  是否听牌: {'✓ 是' if features.is_tenpai else '✗ 否'}")
                print(f"  有效牌数: {features.waits_count}")
                print(f"  立直概率: {features.riichi_prob:.2%}" if hasattr(features, 'riichi_prob') else "")
                print(f"  风险评分: {features.risk_score:.2f}")
    
    def print_action(self, actor: int, action_type: str, tile: Optional[str] = None):
        """打印动作"""
        action_emoji = {
            'tsumo': '🔄',
            'dahai': '🔻',
            'pon': '🔔',
            'chi': '🍜',
            'kan': '🀄',
            'ron': '🎉',
            'hora': '🎊'
        }.get(action_type, '❓')
        
        action_text = {
            'tsumo': '摸牌',
            'dahai': '打牌',
            'pon': '碰',
            'chi': '吃',
            'kan': '杠',
            'ron': '荣和',
            'hora': '自摸和了'
        }.get(action_type, action_type)
        
        if tile:
            print(f"  {action_emoji} P{actor} {action_text}: {TileDisplay.mpsz_to_emoji(tile)}")
        else:
            print(f"  {action_emoji} P{actor} {action_text}")
    
    def print_decision(self, enhanced_bot: EnhancedBot):
        """打印决策解释"""
        explanation = enhanced_bot.get_decision_explanation()
        if explanation:
            print("\n" + "=" * 80)
            print("🧠 增强型机器人决策解释")
            print("=" * 80)
            for line in explanation.split('\n'):
                if line.strip():
                    print(line)
            print("=" * 80)
    
    def print_step(self, step: int, event_type: str, actor: int):
        """打印步骤信息"""
        print(f"\n{'─' * 80}")
        print(f"📍 第{step}步 | {event_type} | P{actor}")
        print('─' * 80)
    
    def print_game_over(self, scores: List[int], ranks: List[int]):
        """打印对局结束"""
        print("\n" + "=" * 80)
        print("🏁 对局结束".center(80))
        print("=" * 80)
        
        player_info = list(zip(scores, ranks))
        sorted_info = sorted(player_info, key=lambda x: x[1])
        
        print("\n🏆 最终排名:")
        for i, (score, rank) in enumerate(sorted_info):
            player_id = list(scores).index(score) if score in scores else 0
            rank_symbol = ['🥇', '🥈', '🥉', '4️⃣'][rank - 1] if rank <= 4 else f'{rank}️⃣'
            print(f"  {rank_symbol} P{player_id}: {score:>6}点")
        
        # 决策统计
        if self.enhanced_bot:
            summary = self.enhanced_bot.tracker.generate_summary()
            print("\n📊 决策统计:")
            print(f"  总决策数: {summary['total_decisions']}")
            print(f"  立直决策数: {summary['riichi_count']}")
            if summary['avg_shanten'] is not None:
                print(f"  平均向听数: {summary['avg_shanten']:.2f}")
            if summary['min_shanten'] is not None:
                print(f"  最小向听数: {summary['min_shanten']}")
        
        print("=" * 80)


def visualize_game(
    player_id: int = 0,
    max_steps: int = 100,
    use_enhanced_bot: bool = True,
    model_path: Optional[str] = None
):
    """
    可视化对局
    
    Args:
        player_id: 增强型机器人控制的玩家ID
        max_steps: 最大步数
        use_enhanced_bot: 是否使用增强型机器人
        model_path: 模型路径（可选）
    
    Returns:
        game_result: 对局结果
    """
    # 创建可视化器
    enhanced_bot = None
    if use_enhanced_bot:
        enhanced_bot = EnhancedBot(
            player_id=player_id,
            model_path=model_path
        )
    
    visualizer = GameVisualizer(enhanced_bot)
    
    # 创建环境
    env = RiichiEnv(game_mode="4p-red-single")
    
    # 创建其他玩家
    agents = {pid: RandomAgent() for pid in range(4)}
    if enhanced_bot:
        agents[player_id] = enhanced_bot
    
    # 初始化
    visualizer.print_header("4人东风战")
    obs_dict = env.reset()
    step = 0
    
    # 对局循环
    while not env.done() and step < max_steps:
        # 获取所有玩家的动作
        actions = {}
        current_actor = None
        current_action_type = None
        current_tile = None
        
        for pid in range(4):
            obs = obs_dict.get(pid)
            if obs is not None:
                action = agents[pid].act(obs)
                actions[pid] = action
                
                # 记录当前动作
                if hasattr(action, 'action_type'):
                    current_actor = pid
                    current_action_type = str(action.action_type).split('.')[-1]
                    current_tile = str(action.tile) if hasattr(action, 'tile') else None
        
        # 显示步骤信息
        visualizer.print_step(
            step,
            current_action_type or "未知",
            current_actor or 0
        )
        
        # 显示得分
        visualizer.print_scores(env.scores(), env.ranks())
        
        # 显示增强型机器人的手牌和决策
        if enhanced_bot:
            obs = obs_dict.get(player_id)
            if obs and hasattr(obs, 'hand'):
                visualizer.print_player_hand(player_id, obs.hand, True)
                visualizer.print_decision(enhanced_bot)
        
        # 执行动作
        obs_dict = env.step(actions)
        step += 1
        
        # 实时显示进度
        if step % 10 == 0:
            print(f"\n⏳ 进度: {step}/{max_steps}步 | 得分: {env.scores()}")
    
    # 显示结束信息
    visualizer.step_count = step
    visualizer.print_game_over(env.scores(), env.ranks())
    
    return {
        'total_steps': step,
        'scores': env.scores(),
        'ranks': env.ranks()
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='麻将对局可视化回放')
    parser.add_argument('--player-id', type=int, default=0, help='玩家ID')
    parser.add_argument('--max-steps', type=int, default=100, help='最大步数')
    parser.add_argument('--no-enhanced', action='store_true', help='不使用增强型机器人')
    parser.add_argument('--model-path', type=str, help='模型路径')
    
    args = parser.parse_args()
    
    result = visualize_game(
        player_id=args.player_id,
        max_steps=args.max_steps,
        use_enhanced_bot=not args.no_enhanced,
        model_path=args.model_path
    )
    
    print(f"\n✅ 对局可视化完成！")


if __name__ == '__main__':
    main()
