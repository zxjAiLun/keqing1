"""
增强型训练数据集

本模块实现基于MJAI格式的训练数据集加载和预处理。

支持:
- MJAI格式数据加载
- 增强型特征提取
- 向听数和立直标签生成
- 批次加载和预处理

Example:
    >>> from dataset_enhanced import EnhancedDataset
    >>> dataset = EnhancedDataset('data/mjai_logs/')
    >>> batch = dataset[0]
    >>> print(batch['features'].shape)  # (13,)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from riichienv import calculate_shanten, HandEvaluator
from .features import EnhancedFeatures, extract_enhanced_features, parse_mjai_hand


@dataclass
class TrainingSample:
    """
    训练样本
    
    包含一个决策的所有必要信息
    """
    features: List[float]           # 13维特征向量
    action: int                   # 动作ID
    reward: float                # 奖励
    shanten: int                # 真实向听数
    should_riichi: float        # 是否应该立直 (0或1)
    player_id: int              # 玩家ID
    step: int                   # 决策步骤
    game_id: str               # 对局ID


class EnhancedDataset(Dataset):
    """
    增强型训练数据集
    
    从MJAI日志文件中加载训练样本
    
    Attributes:
        data_dir: 数据目录路径
        samples: 训练样本列表
        transform: 数据变换函数
    
    Example:
        >>> dataset = EnhancedDataset('data/mjai_logs/')
        >>> print(f"数据集大小: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"特征: {sample['features']}")
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[TrainingSample] = []
        
        # 加载数据
        self._load_data()
        
        # 限制样本数量
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"数据集加载完成: {len(self.samples)} 个样本")
    
    def _load_data(self):
        """从数据目录加载所有MJAI日志文件"""
        if not self.data_dir.exists():
            print(f"警告: 数据目录不存在: {self.data_dir}")
            print("将使用生成的示例数据")
            self._generate_sample_data()
            return
        
        # 遍历所有JSON文件
        json_files = list(self.data_dir.glob("*.json"))
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        all_files = json_files + jsonl_files
        
        if not all_files:
            print(f"警告: 数据目录为空: {self.data_dir}")
            print("将使用生成的示例数据")
            self._generate_sample_data()
            return
        
        # 加载每个文件
        for file_path in all_files:
            try:
                self._load_file(file_path)
            except Exception as e:
                print(f"加载文件失败: {file_path}, 错误: {e}")
    
    def _load_file(self, file_path: Path):
        """加载单个日志文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                # JSONL格式（每行一个JSON）
                events = []
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
            else:
                # JSON格式（数组）
                events = json.load(f)
        
        # 解析事件生成样本
        self._parse_events(events, str(file_path.stem))
    
    def _parse_events(self, events: List[Dict], game_id: str):
        """解析MJAI事件生成训练样本"""
        from riichienv import RiichiEnv
        
        # 使用RiichiEnv复盘
        env = RiichiEnv()
        
        current_player = None
        step = 0
        
        for event in events:
            event_type = event.get('type', '')
            
            if event_type == 'start_game':
                env.apply_event(event)
            
            elif event_type == 'start_kyoku':
                env.apply_event(event)
                current_player = event.get('oya', 0)
                step = 0
            
            elif event_type in ['tsumo', 'dahai', 'pon', 'chi', 'kan', 'ron', 'tsumo', 'reach', 'hora']:
                env.apply_event(event)
                
                # 提取训练样本
                if event_type == 'dahai' and 'actor' in event:
                    actor = event['actor']
                    
                    # 获取当前状态
                    obs = env.get_observation(actor)
                    if obs:
                        sample = self._extract_sample(obs, event, actor, step, game_id)
                        if sample:
                            self.samples.append(sample)
                
                step += 1
    
    def _extract_sample(
        self,
        obs,
        event: Dict,
        player_id: int,
        step: int,
        game_id: str
    ) -> Optional[TrainingSample]:
        """从Observation提取训练样本"""
        try:
            # 获取手牌
            hand = obs.hand if hasattr(obs, 'hand') else []
            
            # 获取立直状态
            riichi_declared = obs.riichi_declared if hasattr(obs, 'riichi_declared') else [False] * 4
            
            # 计算向听数
            shanten = calculate_shanten(hand) if hand else 8
            
            # 是否应该立直
            should_riichi = 1.0 if (shanten == 0 and not riichi_declared[player_id]) else 0.0
            
            # 提取特征
            features = extract_enhanced_features(
                tiles=hand,
                reached=riichi_declared,
                player_id=player_id,
                turn=min(step, 20),
                dangerous_tiles=None
            )
            
            # 动作编码（简化版本）
            action = self._encode_action(event)
            
            # 奖励（简化版本）
            reward = self._calculate_reward(event, shanten)
            
            return TrainingSample(
                features=features.to_list(),
                action=action,
                reward=reward,
                shanten=shanten,
                should_riichi=should_riichi,
                player_id=player_id,
                step=step,
                game_id=game_id
            )
        
        except Exception as e:
            print(f"提取样本失败: {e}")
            return None
    
    def _encode_action(self, event: Dict) -> int:
        """编码动作"""
        action_type = event.get('type', '')
        
        if action_type == 'dahai':
            pai = event.get('pai', '')
            return hash(pai) % 200
        
        elif action_type == 'reach':
            return 150
        
        elif action_type == 'pon':
            return 160
        
        elif action_type == 'chi':
            return 170
        
        elif action_type == 'kan':
            return 180
        
        elif action_type == 'hora':
            return 190
        
        return 0
    
    def _calculate_reward(self, event: Dict, shanten: int) -> float:
        """计算奖励"""
        action_type = event.get('type', '')
        
        if action_type == 'hora':
            return 1.0  # 和了，正奖励
        elif action_type == 'ryukyoku':
            return 0.0  # 流局，无奖励
        else:
            return -0.01  # 其他动作，轻微负奖励
    
    def _generate_sample_data(self):
        """生成示例数据（当没有真实数据时）"""
        print("生成示例训练数据...")
        
        import random
        
        # 预定义的手牌-向听数对（确保兼容）
        hand_shanten_pairs = [
            # (手牌, 向听数)
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 27], 0),  # 12345678m111z
            ([0, 0, 0, 9, 9, 9, 18, 18, 18, 27, 28, 29, 30], 0),  # 111p111s1234z
            ([9, 9, 10, 10, 11, 11, 18, 18, 19, 19, 20, 20, 27], 1),  # 22p333s44s11z
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 27], 0),  # 12345678m111z
        ]
        
        # WaitsQuality枚举
        class WaitsQuality:
            SINGLE = "single"
            DOUBLE = "double"
            MULTI = "multi"
            EDGE = "edge"
        
        # GamePhase枚举
        class GamePhase:
            EARLY = "early"
            MIDDLE = "middle"
            LATE = "late"
        
        for i in range(100):
            # 随机选择
            tiles, shanten = random.choice(hand_shanten_pairs)
            reached = [random.choice([True, False]) for _ in range(4)]
            player_id = random.randint(0, 3)
            turn = random.randint(1, 20)
            
            # 手动构建特征向量，不调用RiichiEnv API
            # 13维特征向量
            is_tenpai = (shanten == 0)
            is_ready = (shanten == -1)
            waits_count = 3 if is_tenpai else 0
            waits_quality = WaitsQuality.DOUBLE if is_tenpai else WaitsQuality.EDGE
            self_reached = reached[player_id]
            reached_count = sum(reached)
            opponent_reached = (reached_count - (1 if self_reached else 0)) > 0
            
            if turn < 5:
                game_phase = GamePhase.EARLY
                is_late_game = False
            elif turn < 15:
                game_phase = GamePhase.MIDDLE
                is_late_game = False
            else:
                game_phase = GamePhase.LATE
                is_late_game = True
            
            risk_score = 0.0
            if any(reached):
                risk_score += 0.5
            if turn > 15:
                risk_score += 0.3
            is_dangerous = risk_score > 0.6
            
            # 构建特征向量
            feature_vec = [
                float(shanten),  # 向听数
                float(is_tenpai),  # 是否听牌
                float(is_ready),  # 是否和了
                float(waits_count),  # 有效牌数量
                1.0 if waits_quality == WaitsQuality.SINGLE else 0.0,  # 是否单吊
                1.0 if waits_quality == WaitsQuality.DOUBLE else 0.0,  # 是否双碰
                float(self_reached),  # 自己是否立直
                float(reached_count),  # 立直家数量
                float(opponent_reached),  # 是否有对手立直
                float(turn),  # 当前巡目
                float(is_late_game),  # 是否后期
                risk_score,  # 风险评分
                float(is_dangerous),  # 是否危险局面
            ]
            
            should_riichi = 1.0 if shanten == 0 and not reached[0] else 0.0
            
            sample = TrainingSample(
                features=feature_vec,
                action=random.randint(0, 199),
                reward=random.choice([-0.01, 0.0, 1.0]),
                shanten=shanten,
                should_riichi=should_riichi,
                player_id=player_id,
                step=i,
                game_id=f"sample_{i // 10}"
            )
            
            self.samples.append(sample)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为PyTorch张量
        return {
            'features': torch.tensor(sample.features, dtype=torch.float32),
            'actions': torch.tensor(sample.action, dtype=torch.long),
            'rewards': torch.tensor(sample.reward, dtype=torch.float32),
            'true_shanten': torch.tensor(sample.shanten, dtype=torch.float32),
            'should_riichi': torch.tensor(sample.should_riichi, dtype=torch.float32),
            'player_id': torch.tensor(sample.player_id, dtype=torch.long),
            'step': torch.tensor(sample.step, dtype=torch.long),
            'game_id': sample.game_id
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.samples:
            return {}
        
        shanten_values = [s.shanten for s in self.samples]
        riichi_count = sum(1 for s in self.samples if s.should_riichi == 1.0)
        
        return {
            'total_samples': len(self.samples),
            'avg_shanten': sum(shanten_values) / len(shanten_values),
            'min_shanten': min(shanten_values),
            'max_shanten': max(shanten_values),
            'riichi_samples': riichi_count,
            'riichi_rate': riichi_count / len(self.samples)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数
    
    将样本列表整理为批次张量
    """
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'actions': torch.stack([b['actions'] for b in batch]),
        'rewards': torch.stack([b['rewards'] for b in batch]),
        'true_shanten': torch.stack([b['true_shanten'] for b in batch]),
        'should_riichi': torch.stack([b['should_riichi'] for b in batch]),
        'player_id': torch.stack([b['player_id'] for b in batch]),
        'step': torch.stack([b['step'] for b in batch]),
        'game_id': [b['game_id'] for b in batch]
    }


def create_dataloader(
    dataset: EnhancedDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    
    Returns:
        DataLoader实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
