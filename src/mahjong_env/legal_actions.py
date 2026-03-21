from __future__ import annotations

from collections import Counter
from typing import Dict, List

from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES, tile_without_aka
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
        if _can_pon(hand, tile_raw):
            legal.append(Action(type="pon", actor=actor, target=discarder, pai=tile_raw, consumed=[tile_norm, tile_norm]))
        if _can_daiminkan(hand, tile_raw):
            legal.append(Action(type="daiminkan", actor=actor, target=discarder, pai=tile_raw, consumed=[tile_norm, tile_norm, tile_norm]))

        next_player = (discarder + 1) % 4
        if actor == next_player:
            for c in _chi_patterns(tile_raw):
                if hand[c[0]] >= 1 and hand[c[1]] >= 1:
                    legal.append(Action(type="chi", actor=actor, target=discarder, pai=tile_raw, consumed=c))

        legal.append(Action(type="none", actor=actor))
        return legal

    if actor_to_move == actor:
        # Note: mjai 会用 last_self_tsumo 的“牌面字符串”来判断 tsumogiri。
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
        if not reached and sum(hand.values()) == 14 and (shanten is None or shanten == 0):
            legal.append(Action(type="reach", actor=actor))
        for tile in _ankan_candidates(hand):
            legal.append(Action(type="ankan", actor=actor, pai=tile, consumed=[tile] * 4))
        # kakan（对已碰/明杠再杠）在“打牌选择”阶段可能出现。
        # 我们基于已有的 pon meld 来枚举升级可能性（不处理红牌aka精确选择）。
        for meld in state_snapshot.get("melds", [[]])[actor]:
            if meld.get("type") != "pon":
                continue
            meld_pai = meld.get("pai")
            if not meld_pai:
                continue
            # 需要手里还剩1张“杠牌”。
            if hand.get(meld_pai, 0) >= 1:
                legal.append(
                    Action(
                        type="kakan",
                        actor=actor,
                        pai=meld_pai,
                        consumed=[meld_pai] * 3,
                    )
                )
        legal.append(Action(type="none", actor=actor))
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

