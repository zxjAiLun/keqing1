"""
增强型机器人

本模块实现基于增强型特征和模型的麻将AI机器人，支持决策追踪和解释。

核心功能：
1. 集成RiichiEnv进行游戏模拟
2. 使用features_enhanced提取可解释性特征
3. 使用model_enhanced进行多任务推理
4. 实现决策追踪机制
5. 生成决策解释

Example:
    >>> from bot_enhanced import EnhancedBot
    >>> bot = EnhancedBot(player_id=0)
    >>> action = bot.act(obs)
    >>> print(bot.get_decision_explanation())
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch

from riichienv import RiichiEnv, Action
from .model import EnhancedModel, InterpretableOutput
from .features import EnhancedFeatures, extract_enhanced_features, parse_mjai_hand

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.rule_bot import fallback_action


@dataclass
class DecisionRecord:
    """
    决策记录
    
    记录每一步的完整决策信息
    """
    step: int
    timestamp: str
    player_id: int
    
    # 特征信息
    features: Optional[EnhancedFeatures] = None
    feature_vector: Optional[List[float]] = None
    
    # 模型输出
    model_outputs: Optional[Dict[str, Any]] = None
    
    # 决策信息
    legal_actions: Optional[List[Action]] = None
    selected_action: Optional[Action] = None
    action_reason: str = ""
    
    # 解释
    explanation: Optional[str] = None
    
    # 向听信息
    shanten: Optional[int] = None
    waits_count: Optional[int] = None
    riichi_prob: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'player_id': self.player_id,
            'shanten': self.shanten,
            'waits_count': self.waits_count,
            'riichi_prob': self.riichi_prob,
            'selected_action': str(self.selected_action) if self.selected_action else None,
            'action_reason': self.action_reason,
            'explanation': self.explanation,
        }


class DecisionTracker:
    """
    决策追踪器
    
    记录机器人的所有决策，支持回溯和分析
    """
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.records: List[DecisionRecord] = []
        self.step_count = 0
    
    def record(self, record: DecisionRecord):
        """记录一个决策"""
        self.records.append(record)
        self.step_count += 1
    
    def get_record(self, step: int) -> Optional[DecisionRecord]:
        """获取指定步骤的记录"""
        for record in self.records:
            if record.step == step:
                return record
        return None
    
    def get_last_record(self) -> Optional[DecisionRecord]:
        """获取最新的记录"""
        return self.records[-1] if self.records else None
    
    def get_riichi_decisions(self) -> List[DecisionRecord]:
        """获取所有立直决策"""
        return [
            r for r in self.records
            if r.selected_action and 'riichi' in str(r.selected_action).lower()
        ]
    
    def generate_summary(self) -> Dict:
        """生成决策摘要"""
        riichi_count = len(self.get_riichi_decisions())
        
        shanten_list = [r.shanten for r in self.records if r.shanten is not None]
        
        return {
            'total_decisions': len(self.records),
            'riichi_count': riichi_count,
            'avg_shanten': sum(shanten_list) / len(shanten_list) if shanten_list else None,
            'min_shanten': min(shanten_list) if shanten_list else None,
        }
    
    def export_to_json(self, path: str | Path):
        """导出为JSON"""
        data = {
            'player_id': self.player_id,
            'total_records': len(self.records),
            'summary': self.generate_summary(),
            'records': [r.to_dict() for r in self.records]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def clear(self):
        """清空记录"""
        self.records.clear()
        self.step_count = 0


class EnhancedBot:
    """
    增强型机器人
    
    基于增强型特征和模型的麻将AI机器人
    
    Attributes:
        player_id: 玩家ID (0-3)
        model: 增强型模型
        tracker: 决策追踪器
        use_enhanced_features: 是否使用增强型特征
        device: 计算设备
    
    Example:
        >>> bot = EnhancedBot(player_id=0)
        >>> action = bot.act(obs)
        >>> print(bot.get_decision_explanation())
    """
    
    def __init__(
        self,
        player_id: int = 0,
        model_path: Optional[str | Path] = None,
        use_enhanced_features: bool = True,
        device: str = 'cpu'
    ):
        self.player_id = player_id
        self.use_enhanced_features = use_enhanced_features
        self.device = device
        
        # 初始化模型
        if model_path:
            self.model = EnhancedModel.load(model_path)
        else:
            self.model = EnhancedModel()
        
        self.model.to(device)
        self.model.eval()

        # 初始化追踪器
        self.tracker = DecisionTracker(player_id)

        # 缓存当前特征
        self.current_features: Optional[EnhancedFeatures] = None
        self.current_feature_vector: Optional[torch.Tensor] = None
    
    def act(self, obs) -> Action:
        """
        根据观察做出决策
        
        Args:
            obs: RiichiEnv的Observation对象
        
        Returns:
            选择的动作
        """
        # 获取合法动作
        legal_actions = obs.legal_actions()
        
        # 提取增强型特征
        features = self._extract_features(obs)
        self.current_features = features
        
        # 转换为模型输入
        feature_vec = torch.tensor(
            features.to_list(),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # 添加batch维度
        
        self.current_feature_vector = feature_vec
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(feature_vec, return_all_heads=True)
        
        # 获取各Head输出
        shanten_pred = outputs['shanten'].item()
        riichi_prob = outputs['riichi'].item()
        value = outputs['value'].item()
        policy_logits = outputs['policy_logits']
        
        # 构建模型输出字典
        model_outputs = {
            'shanten': shanten_pred,
            'riichi_prob': riichi_prob,
            'value': value,
            'policy_logits': policy_logits.cpu().numpy()
        }
        
        # 选择动作
        selected_action = self._select_action(
            legal_actions,
            policy_logits,
            shanten_pred,
            riichi_prob,
            features
        )
        
        # 生成动作原因
        action_reason = self._generate_action_reason(
            selected_action,
            features,
            shanten_pred,
            riichi_prob
        )
        
        # 生成解释
        explanation = self._generate_explanation(
            features,
            shanten_pred,
            riichi_prob,
            value,
            selected_action
        )
        
        # 记录决策
        record = DecisionRecord(
            step=self.tracker.step_count,
            timestamp=datetime.now().isoformat(),
            player_id=self.player_id,
            features=features,
            feature_vector=features.to_list(),
            model_outputs=model_outputs,
            legal_actions=legal_actions,
            selected_action=selected_action,
            action_reason=action_reason,
            explanation=explanation,
            shanten=int(round(shanten_pred)),
            waits_count=features.waits_count,
            riichi_prob=riichi_prob
        )
        self.tracker.record(record)
        
        return selected_action
    
    def _extract_features(self, obs) -> EnhancedFeatures:
        """
        提取增强型特征
        
        Args:
            obs: Observation对象
        
        Returns:
            增强型特征对象
        """
        # 获取手牌（34牌格式）
        tiles = obs.hand if hasattr(obs, 'hand') else []
        
        # 获取立直状态
        riichi_declared = obs.riichi_declared if hasattr(obs, 'riichi_declared') else [False] * 4
        reached = riichi_declared
        
        # 获取巡目（从observation推断）
        turn = self._estimate_turn(obs)
        
        # 提取特征
        features = extract_enhanced_features(
            tiles=tiles,
            reached=reached,
            player_id=self.player_id,
            turn=turn,
            dangerous_tiles=None
        )
        
        return features
    
    def _estimate_turn(self, obs) -> int:
        """
        估计当前巡目
        
        这是一个简化实现，实际应该从完整的事件历史推断
        """
        # 从observation推断巡目
        # 这里使用启发式方法
        return 5  # 默认中期
    
    def _select_action(
        self,
        legal_actions: List[Action],
        policy_logits: torch.Tensor,
        shanten_pred: float,
        riichi_prob: float,
        features: EnhancedFeatures
    ) -> Action:
        """
        选择动作
        
        结合策略模型和规则选择最优动作
        """
        if not legal_actions:
            # 没有合法动作，返回None
            return None
        
        # 如果有模型输出，使用模型
        if self.model is not None:
            # 从policy_logits中选择概率最高的动作
            probs = torch.softmax(policy_logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            
            # 确保索引在合法动作范围内
            if top_idx.item() < len(legal_actions):
                return legal_actions[top_idx.item()]
        
        # 使用fallback规则
        # fallback_action需要的是Dict格式，不是Action对象
        # 这里简化处理，返回第一个合法动作
        if legal_actions:
            return legal_actions[0]
        return None
    
    def _generate_action_reason(
        self,
        action: Action,
        features: EnhancedFeatures,
        shanten_pred: float,
        riichi_prob: float
    ) -> str:
        """生成动作原因"""
        reasons = []
        
        # 向听分析
        if features.is_ready:
            reasons.append("手牌已经和了")
        elif features.is_tenpai:
            reasons.append(f"手牌听牌，等待{features.waits_count}张")
            if features.waits_quality.value == "single":
                reasons.append("警告：单吊听牌")
        else:
            reasons.append(f"向听数：{int(round(shanten_pred))}（需要{int(round(shanten_pred))}上）")
        
        # 立直分析
        if features.self_reached:
            reasons.append("已立直，目标是自摸或荣和")
        elif features.opponent_reached:
            reasons.append("对手已立直，需谨慎")
        
        # 立直概率
        if riichi_prob > 0.7:
            reasons.append(f"立直概率高({riichi_prob:.2f})")
        elif riichi_prob < 0.3:
            reasons.append(f"立直概率低({riichi_prob:.2f})")
        
        # 风险评估
        if features.is_dangerous:
            reasons.append(f"危险局面！风险评分：{features.risk_score:.2f}")
        
        return " | ".join(reasons)
    
    def _generate_explanation(
        self,
        features: EnhancedFeatures,
        shanten_pred: float,
        riichi_prob: float,
        value: float,
        action: Action
    ) -> str:
        """生成决策解释"""
        lines = []
        
        lines.append("=" * 60)
        lines.append("增强型机器人决策解释")
        lines.append("=" * 60)
        
        # 1. 向听信息
        lines.append("\n【向听分析】")
        lines.append(f"  预测向听数: {shanten_pred:.2f}")
        lines.append(f"  特征向听数: {features.shanten}")
        lines.append(f"  是否听牌: {features.is_tenpai}")
        lines.append(f"  有效牌数: {features.waits_count}")
        lines.append(f"  听牌质量: {features.waits_quality.value}")
        
        # 2. 立直决策
        lines.append("\n【立直决策】")
        lines.append(f"  立直概率: {riichi_prob:.3f}")
        if riichi_prob > 0.7:
            lines.append("  → 模型倾向于立直")
        elif riichi_prob < 0.3:
            lines.append("  → 模型倾向于不立直")
        else:
            lines.append("  → 模型在立直与否之间犹豫")
        lines.append(f"  自己是否立直: {features.self_reached}")
        lines.append(f"  对手是否立直: {features.opponent_reached}")
        
        # 3. 局面评估
        lines.append("\n【局面评估】")
        lines.append(f"  局面价值: {value:.3f}")
        if value > 0.5:
            lines.append("  → 局面优势")
        elif value < -0.5:
            lines.append("  → 局面劣势")
        else:
            lines.append("  → 局面均势")
        
        # 4. 风险评估
        lines.append("\n【风险评估】")
        lines.append(f"  风险评分: {features.risk_score:.2f}")
        lines.append(f"  是否危险: {features.is_dangerous}")
        lines.append(f"  巡目: {features.turn}")
        lines.append(f"  游戏阶段: {features.game_phase.value}")
        
        # 5. 决策
        lines.append("\n【最终决策】")
        lines.append(f"  选择动作: {action}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def get_decision_explanation(self) -> str:
        """获取最新决策的解释"""
        last_record = self.tracker.get_last_record()
        if last_record and last_record.explanation:
            return last_record.explanation
        return "暂无决策解释"
    
    def get_last_features(self) -> Optional[EnhancedFeatures]:
        """获取最新的特征"""
        return self.current_features
    
    def reset(self):
        """重置机器人"""
        self.tracker.clear()
        self.current_features = None
        self.current_feature_vector = None
    
    def save_decision_log(self, path: str | Path):
        """保存决策日志"""
        self.tracker.export_to_json(path)
    
    def load_model(self, path: str | Path):
        """加载模型"""
        self.model = EnhancedModel.load(path)
        self.model.to(self.device)
        self.model.eval()


def create_enhanced_bot(
    player_id: int = 0,
    model_path: Optional[str | Path] = None,
    use_enhanced_features: bool = True
) -> EnhancedBot:
    """
    工厂函数：创建增强型机器人
    
    Args:
        player_id: 玩家ID
        model_path: 模型路径（可选）
        use_enhanced_features: 是否使用增强型特征
    
    Returns:
        增强型机器人实例
    
    Example:
        >>> bot = create_enhanced_bot(player_id=0)
        >>> action = bot.act(obs)
    """
    return EnhancedBot(
        player_id=player_id,
        model_path=model_path,
        use_enhanced_features=use_enhanced_features
    )
