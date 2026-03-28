from __future__ import annotations

from collections import Counter
from typing import List

from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES, tile_without_aka, tile_to_34 as _tile_to_34
from mahjong_env.types import Action


def _chi_patterns(tile: str) -> List[List[str]]:
    if tile in AKA_DORA_TILES:
        tile = normalize_tile(tile)
    if len(tile) != 2:
        return []
    num = int(tile[0])
    suit = tile[1]
    if suit not in ("m", "p", "s"):
        return []
    patterns: List[List[str]] = []
    for seq in ((num - 2, num - 1), (num - 1, num + 1), (num + 1, num + 2)):
        a, b = seq
        if 1 <= a <= 9 and 1 <= b <= 9:
            patterns.append([f"{a}{suit}", f"{b}{suit}"])
    return patterns


def _can_pon(hand: Counter, tile: str) -> bool:
    normalized = normalize_tile(tile)
    if not normalized or len(normalized) < 2:
        return False
    aka_tile = f"{normalized[0]}r{normalized[1]}"
    if aka_tile in AKA_DORA_TILES:
        return hand[normalized] + hand.get(aka_tile, 0) >= 2
    return hand[normalized] >= 2


def _can_daiminkan(hand: Counter, tile: str) -> bool:
    normalized = normalize_tile(tile)
    if not normalized or len(normalized) < 2:
        return False
    aka_tile = f"{normalized[0]}r{normalized[1]}"
    if aka_tile in AKA_DORA_TILES:
        return hand[normalized] + hand.get(aka_tile, 0) >= 3
    return hand[normalized] >= 3


def _hand_has_tile(hand: Counter, t: str) -> bool:
    """检查手牌中是否有牌 t，同时考虑赤宝牌（5m/5p/5s ↔ 5mr/5pr/5sr）。"""
    if hand[t] >= 1:
        return True
    if len(t) == 2 and t[0] == "5" and t[1] in ("m", "p", "s"):
        return hand[t + "r"] >= 1
    return False


def _pick_chi_tile(hand: Counter, t: str) -> str:
    """从手牌中取牌 t，优先返回赤宝牌版本。"""
    if len(t) == 2 and t[0] == "5" and t[1] in ("m", "p", "s") and hand[t + "r"] >= 1:
        return t + "r"
    return t


def _pick_consumed(hand: Counter, normalized: str, n: int) -> List[str]:
    """从手牌中选出 n 张 normalized 牌，优先用赤宝牌版本。"""
    aka = normalized[0] + normalized[1] + "r" if len(normalized) == 2 and normalized[0] == "5" and normalized[1] in ("m", "p", "s") else None
    result: List[str] = []
    # 先取赤宝牌
    if aka and aka in AKA_DORA_TILES:
        for _ in range(min(hand.get(aka, 0), n - len(result))):
            result.append(aka)
    # 再补普通牌
    while len(result) < n:
        result.append(normalized)
    return result


def _ankan_candidates(hand: Counter) -> List[str]:
    result = []
    for t, n in hand.items():
        if n >= 4 and t not in AKA_DORA_TILES:
            result.append(t)
    return result


def enumerate_legal_actions(state_snapshot: Dict, actor: int) -> List[Action]:
    legal: List[Action] = []
    hand = Counter(state_snapshot["hand"])
    last_discard = state_snapshot["last_discard"]
    actor_to_move = state_snapshot["actor_to_move"]
    last_tsumo = state_snapshot.get("last_tsumo", [None, None, None, None])[actor]
    last_tsumo_raw = state_snapshot.get("last_tsumo_raw", [None, None, None, None])[actor]
    reached = state_snapshot["reached"][actor]

    # If there is a pending `last_discard` from someone else, the player is in
    # a "call reaction" stage. Even if `actor_to_move` was advanced
    # simplistically, we must not generate `dahai` here.
    if last_discard and last_discard.get("actor") != actor:
        if not last_discard:
            legal.append(Action(type="none", actor=actor))
            return legal
        discarder = last_discard["actor"]
        tile_norm = last_discard["pai"]
        tile_raw = last_discard.get("pai_raw", tile_norm)
        if discarder == actor:
            legal.append(Action(type="none", actor=actor))
            return legal
        # 荣和：检查 waits_tiles（由 bot.py/replay.py 注入）
        waits_tiles = state_snapshot.get("waits_tiles")
        if waits_tiles is not None:
            tile34_idx = _tile_to_34(normalize_tile(tile_raw))
            if tile34_idx >= 0 and tile34_idx < len(waits_tiles) and waits_tiles[tile34_idx]:
                legal.append(Action(type="hora", actor=actor, target=discarder, pai=tile_raw))
        if _can_pon(hand, tile_raw):
            consumed_pon = _pick_consumed(hand, tile_norm, 2)
            legal.append(Action(type="pon", actor=actor, target=discarder, pai=tile_raw, consumed=consumed_pon))
        if _can_daiminkan(hand, tile_raw):
            consumed_kan = _pick_consumed(hand, tile_norm, 3)
            legal.append(Action(type="daiminkan", actor=actor, target=discarder, pai=tile_raw, consumed=consumed_kan))

        next_player = (discarder + 1) % 4
        if actor == next_player:
            for c in _chi_patterns(tile_raw):
                if _hand_has_tile(hand, c[0]) and _hand_has_tile(hand, c[1]):
                    actual_c = [_pick_chi_tile(hand, c[0]), _pick_chi_tile(hand, c[1])]
                    legal.append(Action(type="chi", actor=actor, target=discarder, pai=tile_raw, consumed=actual_c))

        legal.append(Action(type="none", actor=actor))
        return legal

    if actor_to_move == actor:
        pending_reach = state_snapshot.get("pending_reach", [False, False, False, False])[actor]

        if reached:
            # 立直已被接受：只能摸切（tsumogiri），或暗杠。
            if last_tsumo_raw is not None:
                legal.append(Action(type="dahai", actor=actor, pai=last_tsumo_raw, tsumogiri=True))
            elif last_tsumo is not None:
                legal.append(Action(type="dahai", actor=actor, pai=last_tsumo, tsumogiri=True))
            for tile in _ankan_candidates(hand):
                legal.append(Action(type="ankan", actor=actor, pai=tile, consumed=[tile] * 4))
            return legal

        if pending_reach:
            # 已宣告立直、等待打出立直宣言牌：只能打出让手牌保持听牌的牌。
            # 对每张候选牌，临时移除后检验 shanten==0（听牌）。
            from mahjong_env.tiles import tile_to_136 as _to_136
            from mahjong.shanten import Shanten
            from mahjong.tile import TilesConverter
            _shanten_calc = Shanten()
            for tile in list(hand.keys()):
                pai_out = tile
                if last_tsumo == tile and last_tsumo_raw is not None:
                    pai_out = last_tsumo_raw
                test_hand = dict(hand)
                test_hand[tile] -= 1
                if test_hand[tile] == 0:
                    del test_hand[tile]
                hand136 = []
                for t, cnt in test_hand.items():
                    idx136 = _to_136(t)
                    if idx136 >= 0:
                        hand136.extend([idx136] * cnt)
                if not hand136:
                    continue
                tiles34 = TilesConverter.to_34_array(hand136)
                if _shanten_calc.calculate_shanten(tiles34) == 0:
                    legal.append(Action(type="dahai", actor=actor, pai=pai_out, tsumogiri=(last_tsumo == tile)))
            if not legal:
                # fallback：shanten 计算失败时退化为全打法
                for tile in hand.keys():
                    pai_out = tile
                    if last_tsumo == tile and last_tsumo_raw is not None:
                        pai_out = last_tsumo_raw
                    legal.append(Action(type="dahai", actor=actor, pai=pai_out, tsumogiri=(last_tsumo == tile)))
            return legal

        # 普通打牌阶段
        # Note: mjai 会用 last_self_tsumo 的"牌面字符串"来判断 tsumogiri。
        # 我们的简化状态同样用 tile 字符串进行匹配。
        for tile in hand.keys():
            pai_out = tile
            if last_tsumo == tile and last_tsumo_raw is not None:
                pai_out = last_tsumo_raw
            legal.append(
                Action(
                    type="dahai",
                    actor=actor,
                    pai=pai_out,
                    tsumogiri=(last_tsumo == tile),
                )
            )
        # Reach 只能在"打牌选择"阶段宣告，此时手牌应为14张（已含自摸牌）且必须听牌。
        # 这里用更严格的条件，减少reach动作在不合时机时被加入legal set。
        # shanten 信息由 libriichi 在 mjai_bot.py 的 react() 中计算并添加到 snapshot。
        # 只有 shanten == 0 (听牌) 时才允许立直，防止模型错误选择导致 mjai_simulator 报错。
        shanten = state_snapshot.get("shanten", None)
        if not reached and sum(hand.values()) == 14 and shanten == 0:
            legal.append(Action(type="reach", actor=actor))
        for tile in _ankan_candidates(hand):
            legal.append(Action(type="ankan", actor=actor, pai=tile, consumed=[tile] * 4))
        # kakan（对已碰/明杠再杠）在"打牌选择"阶段可能出现。
        # 我们基于已有的 pon meld 来枚举升级可能性（不处理红牌aka精确选择）。
        for meld in state_snapshot.get("melds", [[]])[actor]:
            if meld.get("type") != "pon":
                continue
            meld_pai = meld.get("pai")
            if not meld_pai:
                continue
            # 需要手里还剩1张"杠牌"(考虑赤宝牌等价)。
            meld_norm = normalize_tile(meld_pai)
            if _hand_has_tile(hand, meld_norm):
                kakan_pai = _pick_chi_tile(hand, meld_norm)
                legal.append(
                    Action(
                        type="kakan",
                        actor=actor,
                        pai=kakan_pai,
                        consumed=[meld_pai] * 3,
                    )
                )
        return legal

    if not last_discard:
        legal.append(Action(type="none", actor=actor))
        return legal

    discarder = last_discard["actor"]
    tile = last_discard["pai"]
    if discarder == actor:
        legal.append(Action(type="none", actor=actor))
        return legal

    if _can_pon(hand, tile):
        legal.append(Action(type="pon", actor=actor, target=discarder, pai=tile, consumed=[tile, tile]))
    if _can_daiminkan(hand, tile):
        legal.append(Action(type="daiminkan", actor=actor, target=discarder, pai=tile, consumed=[tile, tile, tile]))

    next_player = (discarder + 1) % 4
    if actor == next_player:
        for c in _chi_patterns(tile):
            if hand[c[0]] >= 1 and hand[c[1]] >= 1:
                legal.append(Action(type="chi", actor=actor, target=discarder, pai=tile, consumed=c))

    legal.append(Action(type="none", actor=actor))
    return legal

