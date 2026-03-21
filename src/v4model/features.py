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
            # 使用 riichienv.calculate_shanten 计算真正的向听数
            shanten = self._calc_shanten(obs.hand)
            scalar_features[scalar_idx] = shanten / 10.0
            scalar_idx += 1

            # 14: 是否听牌
            is_tenpai = (shanten <= 0)
            scalar_features[scalar_idx] = 1.0 if is_tenpai else 0.0
            scalar_idx += 1

            # 15: 有效进张数 (Ukeire)
            # 按向听数归一化
            ukeire = self._calc_ukeire(obs.hand)
            # 不同向听数的进张数范围不同，按向听数分别归一化
            if shanten == 1:
                # 1向听: 最大进张数约20
                normalized_ukeire = min(ukeire / 25.0, 1.0)
            elif shanten == 2:
                # 2向听: 最大进张数约30
                normalized_ukeire = min(ukeire / 100.0, 1.0)
            elif shanten >= 3:
                # 3+向听: 最大进张数约50
                normalized_ukeire = min(ukeire / 250.0, 1.0)
            else:
                normalized_ukeire = 0.0
            scalar_features[scalar_idx] = normalized_ukeire
            scalar_idx += 1

            # 16: 非役字牌倾向 (无人立直时)
            # 无人立直时才考虑打出字牌

            # 检查其他家是否有人立直
            other_riichi = any(obs.riichi_declared[i] for i in range(4) if i != obs.player_id)

            # 估算巡目 (通过累计舍牌数)
            total_discards = sum(len(d) for d in obs.discards) if hasattr(obs, 'discards') else 0
            is_early = total_discards < 6  # 早巡
            is_late = total_discards >= 12  # 尾盘

            # 宝牌指示器的牌类型
            dora_types = set()
            if hasattr(obs, 'dora_indicators'):
                for tid in obs.dora_indicators:
                    dora_types.add(tid // 4)

            if not other_riichi:
                # 无人立直，计算打出字牌倾向
                # 统计手里有多少字牌
                honor_tiles = [t for t in obs.hand if t >= 108]  # 字牌 tile_id >= 108
                n_honor = len(honor_tiles)

                # 检查打出字牌是否进向听
                # 如果字牌不是宝牌，打出通常不进向听
                if n_honor > 0:
                    # 检查手里字牌有多少是宝牌
                    honor_dora_count = sum(1 for t in honor_tiles if t // 4 in dora_types)

                    if is_early and honor_dora_count == 0:
                        # 早巡 + 字牌非宝牌 + 打出不进向听 = 极高概率
                        honor_tendency = 0.95
                    elif is_early and honor_dora_count > 0:
                        # 早巡 + 字牌是宝牌 + 打出进向听 = 极低概率
                        honor_tendency = 0.05
                    elif is_late and n_honor >= 2:
                        # 尾盘 + >=2张字牌 = 极低概率
                        honor_tendency = 0.1
                    elif shanten <= 2:
                        # 其他巡目 + 向听<=2 = 极高概率
                        honor_tendency = 0.9
                    else:
                        honor_tendency = 0.5
                else:
                    honor_tendency = 0.0
            else:
                honor_tendency = 0.0  # 有人立直时设为0
            scalar_features[scalar_idx] = honor_tendency
            scalar_idx += 1

            # 17: 胡牌概率 (暂时保留占位)
            scalar_features[scalar_idx] = 0.0
            scalar_idx += 1

            # 18: Speed Reference (速度参考)
            # 仅在 1-向听和 2-向听时有意义
            speed_ref = self._calc_speed_ref(obs.hand, shanten)
            scalar_features[scalar_idx] = speed_ref / 100.0  # 归一化到 [0, 1] (原始是百分比)
            scalar_idx += 1

            # 19: 行动模式 (0=平衡, 0.5=进攻, 1=防守)
            # 防守模式：其他家立直 且 自己向听>2，或者其他家fuuro宝牌>=3
            # 进攻模式：早巡 或 自己>=2张宝牌
            # 其他：平衡模式

            # 检查其他家fuuro中的宝牌数
            fuuro_dora_count = 0
            if hasattr(obs, 'melds'):
                for pid, meld_list in enumerate(obs.melds):
                    if pid == obs.player_id:
                        continue
                    for meld in meld_list:
                        # meld 可能是 list 或其他格式
                        if isinstance(meld, list):
                            for tile_id in meld:
                                if tile_id // 4 in dora_types:
                                    fuuro_dora_count += 1

            # 统计自己手里的宝牌数
            my_dora_count = sum(1 for t in obs.hand if t // 4 in dora_types)

            # 判断模式
            if other_riichi and shanten > 2:
                # 防守：有人立直 + 自己向听>2
                action_mode = 1.0
            elif fuuro_dora_count >= 3:
                # 防守：其他家fuuro宝牌>=3
                action_mode = 1.0
            elif is_early or my_dora_count >= 2:
                # 进攻：早巡 或 自己>=2张宝牌
                action_mode = 0.5
            else:
                # 平衡
                action_mode = 0.0

            scalar_features[scalar_idx] = action_mode
            scalar_idx += 1

        except Exception as e:
            # 分析失败时使用默认值
            scalar_features[scalar_idx:scalar_idx + 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            scalar_idx += 6

        return tile_features, scalar_features

    def _calc_shanten(self, hand_tile_ids: List[int]) -> int:
        """
        计算向听数 (距离听牌的距离)

        使用 riichienv.calculate_shanten 计算真正的向听数

        Args:
            hand_tile_ids: 手牌的 tile id 列表 (0-135)

        Returns:
            向听数 (-1=已完成, 0=听牌, >0=距离听牌)
        """
        try:
            # calculate_shanten 期望的是 [0,4,8,...] 格式的 tile id 列表
            return calculate_shanten(hand_tile_ids)
        except Exception:
            # 如果计算失败，返回一个保守的估计
            return 4

    def _calc_ukeire(self, hand_tile_ids: List[int]) -> int:
        """
        计算最佳进张数 (Ukeire / Waiting)

        在打牌前计算：
        - 如果是14张手牌，计算打每张牌后的最大进张数
        - 如果是13张手牌，直接计算进张数

        Args:
            hand_tile_ids: 手牌的 tile id 列表 (0-135)

        Returns:
            最大有效进张数
        """
        original_shanten = self._calc_shanten(hand_tile_ids)

        # 如果已经听牌或完成，不需要进张
        if original_shanten <= 0:
            return 0

        # 如果是13张手牌，直接计算进张数
        if len(hand_tile_ids) == 13:
            return self._calc_ukeire_after_discard(hand_tile_ids)

        # 如果是14张手牌，计算打每张牌后的最大进张数
        if len(hand_tile_ids) == 14:
            return self._calc_best_ukeire_for_14tile(hand_tile_ids)

        return 0

    def _calc_ukeire_after_discard(self, hand_13: List[int]) -> int:
        """
        计算打牌后的进张数

        Args:
            hand_13: 打牌后的13张手牌

        Returns:
            进张数 (摸入能听牌的牌数)
        """
        original_shanten = self._calc_shanten(hand_13)

        # 如果已经听牌或完成，不需要进张
        if original_shanten <= 0:
            return 0

        # 统计每种牌type的数量
        tile_type_counts = [0] * 34
        for tid in hand_13:
            if 0 <= tid < 136:
                tile_type = tid // 4
                tile_type_counts[tile_type] += 1

        total_ukeire = 0

        # 遍历所有34种牌，检查摸入是否能降低向听
        for tile_type in range(34):
            remaining = 4 - tile_type_counts[tile_type]
            if remaining <= 0:
                continue

            # 模拟摸入这张牌 (13张 -> 14张)
            test_hand = hand_13 + [tile_type * 4]
            new_shanten = self._calc_shanten(test_hand)

            if new_shanten < original_shanten:
                total_ukeire += remaining

        return total_ukeire

    def _calc_best_ukeire_for_14tile(self, hand_14: List[int]) -> int:
        """
        计算14张手牌的最佳进张数

        遍历所有可能的打牌选择，返回最佳选择的最大进张数

        Args:
            hand_14: 14张手牌

        Returns:
            最大有效进张数
        """
        original_shanten = self._calc_shanten(hand_14)

        if original_shanten <= 0:
            return 0

        # 统计每种牌type的数量
        tile_type_counts = [0] * 34
        for tid in hand_14:
            if 0 <= tid < 136:
                tile_type = tid // 4
                tile_type_counts[tile_type] += 1

        max_ukeire = 0

        # 遍历所有可能的打牌选择
        for tile_type in range(34):
            if tile_type_counts[tile_type] == 0:
                continue

            # 模拟打出这张牌
            new_hand = []
            removed = False
            for tid in hand_14:
                if not removed and tid // 4 == tile_type:
                    removed = True
                else:
                    new_hand.append(tid)

            # 计算打这张牌后的进张数
            ukeire = self._calc_ukeire_after_discard(new_hand)
            max_ukeire = max(max_ukeire, ukeire)

        return max_ukeire

    def _calc_speed_ref(self, hand_tile_ids: List[int], shanten: int) -> float:
        """
        计算速度参考 (Speed Reference)

        仅在 1-向听和 2-向听时有意义

        公式:
        p2 = ukeire / 120
        p1 = avgNextUkeire / 120
        q1 = 1 - p1
        q2 = 1 - p2
        speedRef = (1 - (p2 * q1^n - p1 * q2^n) / (q1 - q2)) * 100

        Args:
            hand_tile_ids: 手牌的 tile id 列表
            shanten: 当前向听数

        Returns:
            Speed Reference (百分比，0-100)
        """
        if shanten < 1 or shanten > 2:
            return 0.0

        # 仅在14张手牌时计算
        if len(hand_tile_ids) != 14:
            return 0.0

        original_shanten = shanten
        ukeire = self._calc_ukeire(hand_tile_ids)

        if ukeire == 0:
            return 0.0

        # 计算 avgNextUkeire
        # 统计所有能进张的牌，以及进张后的最佳进张数
        tile_type_counts = [0] * 34
        for tid in hand_tile_ids:
            if 0 <= tid < 136:
                tile_type = tid // 4
                tile_type_counts[tile_type] += 1

        next_shanten_tiles = 0
        next_shanten_ukeire = 0

        # 首先收集所有能进张的牌
        for tile_type in range(34):
            remaining = 4 - tile_type_counts[tile_type]
            if remaining <= 0:
                continue

            test_hand = hand_tile_ids + [tile_type * 4]
            new_shanten = self._calc_shanten(test_hand)

            if new_shanten < original_shanten:
                # 这是能降低向听的进张牌
                next_shanten_tiles += remaining

                # 计算进张后的最佳进张 (打哪张最好)
                best_after_ukeire = self._calc_best_discard_ukeire(test_hand, original_shanten - 1)
                next_shanten_ukeire += remaining * best_after_ukeire

        if next_shanten_tiles == 0:
            return 0.0

        avg_next_ukeire = next_shanten_ukeire / next_shanten_tiles

        # 计算 Speed Reference
        left_count = 120
        left_turns = 10 if shanten == 1 else 3

        p2 = ukeire / left_count
        p1 = avg_next_ukeire / left_count
        q1 = 1 - p1
        q2 = 1 - p2

        if abs(q1 - q2) < 1e-9:
            return 0.0

        # speedRef = (1 - (p2 * q1^n - p1 * q2^n) / (q1 - q2)) * 100
        q1_power_n = q1 ** left_turns
        q2_power_n = q2 ** left_turns
        speed_ref = (1 - (p2 * q1_power_n - p1 * q2_power_n) / (q1 - q2)) * 100

        return max(0.0, min(100.0, speed_ref))

    def _calc_best_discard_ukeire(self, hand_tile_ids: List[int], target_shanten: int) -> int:
        """
        计算打某张牌后的最佳进张数

        遍历手牌中每种牌，计算打出后的最大进张数

        Args:
            hand_tile_ids: 手牌 (包含刚摸入的牌)
            target_shanten: 目标向听数 (应该是 original_shanten - 1)

        Returns:
            最佳进张数
        """
        # 统计每种牌type的数量
        tile_type_counts = [0] * 34
        for tid in hand_tile_ids:
            if 0 <= tid < 136:
                tile_type = tid // 4
                tile_type_counts[tile_type] += 1

        best_ukeire = 0

        # 尝试打每种牌 (只考虑手牌中有的)
        for tile_type in range(34):
            if tile_type_counts[tile_type] <= 0:
                continue

            # 模拟打出这张牌 (只移除一张)
            new_hand = []
            removed = False
            for tid in hand_tile_ids:
                if not removed and tid // 4 == tile_type:
                    removed = True  # 只移除一张
                else:
                    new_hand.append(tid)

            new_shanten = self._calc_shanten(new_hand)

            if new_shanten == target_shanten:
                # 计算这个状态的进张数
                ukeire = self._calc_ukeire(new_hand)
                best_ukeire = max(best_ukeire, ukeire)

        return best_ukeire

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
