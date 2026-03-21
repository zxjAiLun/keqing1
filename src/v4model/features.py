"""
特征工程模块 - 将 riichienv Observation 转换为模型输入

设计思路:
- 37 种牌类型 (34 标准 + 3 aka5)
- 特征分为两部分:
  1. 牌类型平面: 手牌、副露、各家舍牌、宝牌
  2. 标量特征: 向听数、分数差、副露次数、巡目等

输出形状:
- 牌类型平面: (37, C) float32
- 标量特征: (S,) float32
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from riichienv import Observation, HandEvaluator, calculate_shanten, parse_hand
import riichienv.convert as cvt


# ============================================================================
# 常量定义
# ============================================================================

# 37 牌类型 (34 标准 + 3 aka5)
N_TILE_TYPES = 37

# aka5 的牌类型索引 (34, 35, 36)
AKA5_TYPES = {
    34: '5m',  # aka5 5万
    35: '5p',  # aka5 5筒
    36: '5s',  # aka5 5索
}

# 默认特征维度
DEFAULT_TILE_PLANE_DIM = 32
DEFAULT_SCALAR_DIM = 24


# ============================================================================
# 特征编码器
# ============================================================================

class FeatureEncoder:
    """
    特征编码器 - 将 riichienv Observation 转换为神经网络输入

    输出形状:
    - 牌类型平面: (37, tile_plane_dim)
    - 标量特征: (scalar_dim,)
    """

    def __init__(self, tile_plane_dim: int = 32, scalar_dim: int = 24):
        """
        Args:
            tile_plane_dim: 牌类型平面的特征维度，默认 32
            scalar_dim: 标量特征的维度，默认 24
        """
        self.tile_plane_dim = tile_plane_dim
        self.scalar_dim = scalar_dim
        self.n_tile_types = N_TILE_TYPES

    def tile_to_type(self, tile_id: int) -> int:
        """
        将 tile id (0-135) 映射到牌类型索引 (0-36)

        37 牌类型:
        - 0-8:   1-9m (万子)
        - 9-17:  1-9p (筒子)
        - 18-26: 1-9s (索子)
        - 27-33: E/S/W/N/P/F/C (字牌)
        - 34:    aka5 5m
        - 35:    aka5 5p
        - 36:    aka5 5s

        riichienv tile id 分布:
        - 万子: 0-35 (1-9m * 4)
        - 筒子: 36-71 (1-9p * 4)
        - 索子: 72-107 (1-9s * 4)
        - 字牌: 108-135 (7种字牌 * 4)

        aka5 tile id: 17(5m), 53(5p), 89(5s)
        """
        # aka5 先检查
        if tile_id == 17:
            return 34  # aka5 5m
        if tile_id == 53:
            return 35  # aka5 5p
        if tile_id == 89:
            return 36  # aka5 5s

        # 标准牌: tile_type = tile_id // 4
        return tile_id // 4

    def encode(self, obs: Observation) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 Observation 编码为特征

        Returns:
            tile_features: shape (37, tile_plane_dim) - 牌类型平面特征
            scalar_features: shape (scalar_dim,) - 标量特征
        """
        tile_features = np.zeros((37, self.tile_plane_dim), dtype=np.float32)
        scalar_features = np.zeros(self.scalar_dim, dtype=np.float32)

        # =====================================================================
        # 牌类型平面特征
        # =====================================================================

        # 通道 0: 自己的手牌 (37)
        for tile_id in obs.hand:
            tile_type = self.tile_to_type(tile_id)
            if tile_type < 37:
                tile_features[tile_type, 0] = 1.0

        # 通道 1: 宝牌指示器 (37)
        if hasattr(obs, 'dora_indicators') and obs.dora_indicators:
            for tile_id in obs.dora_indicators:
                tile_type = self.tile_to_type(tile_id)
                if tile_type < 37:
                    tile_features[tile_type, 1] = 1.0

        # 通道 2-5: 各玩家舍牌 (4 * 37 -> 简化为归一化计数)
        # 每位玩家一个通道，记录该玩家舍牌的牌类型
        if hasattr(obs, 'kawa') and obs.kawa:
            for pid, discards in enumerate(obs.kawa):
                if pid < 4:
                    channel = 2 + pid
                    tile_type_counts = np.zeros(37)
                    for tile_id in discards:
                        tile_type = self.tile_to_type(tile_id)
                        if tile_type < 37:
                            tile_type_counts[tile_type] += 1
                    # 归一化到 [0, 1]
                    if tile_type_counts.max() > 0:
                        tile_type_counts /= tile_type_counts.max()
                    tile_features[:, channel] = tile_type_counts

        # =====================================================================
        # 标量特征
        # =====================================================================

        scalar_idx = 0

        # 0: 亲家标志 (是否为亲)
        oya = obs.oya if hasattr(obs, 'oya') else 0
        scalar_features[scalar_idx] = 1.0 if oya == 0 else 0.0
        scalar_idx += 1

        # 1: 场风 (0-3)
        round_wind = obs.round_wind if hasattr(obs, 'round_wind') else 0
        scalar_features[scalar_idx] = round_wind / 3.0
        scalar_idx += 1

        # 2: 本场
        honba = obs.honba if hasattr(obs, 'honba') else 0
        scalar_features[scalar_idx] = min(honba / 10.0, 1.0)
        scalar_idx += 1

        # 3: 立直棒
        riichi_sticks = obs.riichi_sticks if hasattr(obs, 'riichi_sticks') else 0
        scalar_features[scalar_idx] = min(riichi_sticks / 5.0, 1.0)
        scalar_idx += 1

        # 4-7: 各玩家副露次数 (0=门清)
        if hasattr(obs, 'fuuro_counts'):
            for pid in range(4):
                count = obs.fuuro_counts[pid] if pid < len(obs.fuuro_counts) else 0
                scalar_features[scalar_idx + pid] = min(count / 4.0, 1.0)
        else:
            # 从 fuuro 计算副露次数
            for pid in range(4):
                count = 0
                if hasattr(obs, 'fuuro') and obs.fuuro:
                    for meld in obs.fuuro:
                        if isinstance(meld, tuple) and len(meld) >= 2:
                            actor = meld[0] if isinstance(meld[0], int) else 0
                            if actor == pid:
                                count += 1
                scalar_features[scalar_idx + pid] = min(count / 4.0, 1.0)
        scalar_idx += 4

        # 8-11: 各玩家巡目 (相对巡目)
        # 巡目 = 自己的回合数，可以通过累计舍牌数估算
        if hasattr(obs, 'kawa') and obs.kawa:
            for pid in range(4):
                discards = obs.kawa[pid] if pid < len(obs.kawa) else []
                # 粗略估计巡目 = 舍牌数
                # 实际巡目需要考虑副露等因素
                n_discards = len(discards)
                scalar_features[scalar_idx + pid] = min(n_discards / 20.0, 1.0)
        scalar_idx += 4

        # 12: 与其他三家分数差 (取最大差值)
        my_score = obs.scores[0]  # 当前玩家的分数
        max_diff = 0
        for score in obs.scores[1:]:
            diff = abs(score - my_score) / 50000.0
            max_diff = max(max_diff, diff)
        scalar_features[scalar_idx] = min(max_diff, 1.0)
        scalar_idx += 1

        # =====================================================================
        # 手牌分析
        # =====================================================================

        try:
            he = HandEvaluator(obs.hand, None)
            is_tenpai = he.is_tenpai()
            waits = he.get_waits() if is_tenpai else []
            n_waits = len(waits)

            # 13: 向听数 (0=已听牌)
            shanten = self._calc_shanten(he)
            scalar_features[scalar_idx] = shanten / 7.0
            scalar_idx += 1

            # 14: 是否听牌
            scalar_features[scalar_idx] = 1.0 if is_tenpai else 0.0
            scalar_idx += 1

            # 15: 有效进张数
            scalar_features[scalar_idx] = n_waits / 34.0
            scalar_idx += 1

            # 16: 预计胡牌点数
            if is_tenpai and n_waits > 0:
                dora = obs.dora_indicators if hasattr(obs, 'dora_indicators') else []
                wait_tile_id = waits[0] * 4
                result = he.calc(wait_tile_id, dora, [], None)
                if result:
                    agari = max(result.ron_agari, result.tsumo_agari_oya, result.tsumo_agari_ko)
                    scalar_features[scalar_idx] = min(agari / 30000.0, 1.0)
            scalar_idx += 1

            # 17: 胡牌概率
            win_prob = self._calc_win_probability(shanten, n_waits, is_tenpai)
            scalar_features[scalar_idx] = win_prob
            scalar_idx += 1

            # 18: 有役/无役判断
            # 简单判断: 有进张则可能有役
            has_yaku = 1.0 if is_tenpai or shanten <= 1 else 0.0
            scalar_features[scalar_idx] = has_yaku
            scalar_idx += 1

            # 19: 副露倾向
            # 基于当前副露次数和巡目计算
            my_fuuro = scalar_features[4]  # 自己的副露次数
            my_round = scalar_features[8]  # 自己的巡目
            # 巡目后期副露倾向降低
            fuuro_tendency = my_fuuro * (1.0 - my_round * 0.5)
            scalar_features[scalar_idx] = min(fuuro_tendency, 0.1)
            scalar_idx += 1

        except Exception as e:
            # 分析失败时使用默认值
            scalar_features[scalar_idx:scalar_idx + 7] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            scalar_idx += 7

        return tile_features, scalar_features

    def _calc_shanten(self, he: HandEvaluator) -> int:
        """
        计算向听数 (距离听牌的距离)
        """
        if he.is_tenpai():
            return 0
        # HandEvaluator 没有直接提供向听数
        waits = he.get_waits()
        if len(waits) > 0:
            return 1
        return 4  # 粗略估计

    def _calc_win_probability(self, shanten: int, n_waits: int, is_tenpai: bool) -> float:
        """
        计算胡牌概率

        公式:
        - 听牌时: n_waits / 34
        - 1向听时: (n_waits / 34) * 0.3
        - 2向听时: (n_waits / 34) * 0.1
        - 3+向听: 极低
        """
        if is_tenpai:
            return n_waits / 34.0

        base_rates = {0: 1.0, 1: 0.3, 2: 0.1, 3: 0.02, 4: 0.005}
        base_rate = base_rates.get(shanten, 0.001)
        return (n_waits / 34.0) * base_rate

    def get_feature_shape(self) -> Tuple[Tuple[int, int], Tuple[int]]:
        """返回特征形状"""
        return (37, self.tile_plane_dim), (self.scalar_dim,)

    def get_total_dim(self) -> int:
        """返回总特征维度"""
        return 37 * self.tile_plane_dim + self.scalar_dim


def test_feature_encoder():
    """测试特征编码器"""
    from riichienv import RiichiEnv

    env = RiichiEnv(game_mode='4p-red-half')
    obs_dict = env.reset()
    obs = list(obs_dict.values())[0]

    encoder = FeatureEncoder(tile_plane_dim=32, scalar_dim=24)

    # 测试编码
    tile_feat, scalar_feat = encoder.encode(obs)

    print(f"=== 特征形状 ===")
    print(f"牌类型平面: {tile_feat.shape}")
    print(f"标量特征: {scalar_feat.shape}")
    print(f"总维度: {encoder.get_total_dim()}")

    print(f"\n=== 牌类型平面 ===")
    print(f"手牌通道 (ch=0) 非零: {np.where(tile_feat[:, 0] > 0)[0].tolist()}")
    print(f"宝牌通道 (ch=1) 非零: {np.where(tile_feat[:, 1] > 0)[0].tolist()}")

    print(f"\n=== 标量特征 ===")
    print(f"0: 亲家: {scalar_feat[0]:.2f}")
    print(f"1: 场风: {scalar_feat[1]:.2f}")
    print(f"2: 本场: {scalar_feat[2]:.2f}")
    print(f"3: 立直棒: {scalar_feat[3]:.2f}")
    print(f"4-7: 副露次数: {scalar_feat[4:8]}")
    print(f"8-11: 巡目: {scalar_feat[8:12]}")
    print(f"12: 分数差: {scalar_feat[12]:.2f}")
    print(f"13: 向听数: {scalar_feat[13]:.2f}")
    print(f"14: 是否听牌: {scalar_feat[14]:.2f}")
    print(f"15: 有效进张: {scalar_feat[15]:.2f}")
    print(f"16: 预计胡牌点: {scalar_feat[16]:.2f}")
    print(f"17: 胡牌概率: {scalar_feat[17]:.4f}")
    print(f"18: 有役/无役: {scalar_feat[18]:.2f}")
    print(f"19: 副露倾向: {scalar_feat[19]:.2f}")

    # 测试 tile_to_type
    print(f"\n=== tile_to_type 测试 ===")
    print(f"5m (tile_id=17) -> type {encoder.tile_to_type(17)} (应该是 34 aka5)")
    print(f"5p (tile_id=53) -> type {encoder.tile_to_type(53)} (应该是 35 aka5)")
    print(f"5s (tile_id=89) -> type {encoder.tile_to_type(89)} (应该是 36 aka5)")
    print(f"1m (tile_id=0) -> type {encoder.tile_to_type(0)} (应该是 0)")
    print(f"1p (tile_id=36) -> type {encoder.tile_to_type(36)} (应该是 9)")


if __name__ == "__main__":
    test_feature_encoder()
