from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mahjong_env.legal_actions import enumerate_legal_action_specs
from mahjong_env.scoring import score_hora
from mahjong_env.state import GameState, PlayerState, TileStateError, apply_event
from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES
from mahjong_env.types import action_dict_to_spec, action_specs_match

AKA5 = ("5mr", "5pr", "5sr")


def _normalize_or_keep_aka(tile: str) -> str:
    """规范化 tile，但保留赤宝牌的原样"""
    if tile in AKA5:
        return tile
    return normalize_tile(tile)


def _build_wall_tiles() -> List[str]:
    """构造一副 136 张牌：每个数牌花色 4 张，其中 1 张 5 替换成赤宝牌。"""

    wall: List[str] = []

    for suit, aka_tile in (("m", "5mr"), ("p", "5pr"), ("s", "5sr")):
        suit_tiles = [f"{n}{suit}" for n in range(1, 10) for _ in range(4)]
        first_five = suit_tiles.index(f"5{suit}")
        suit_tiles[first_five] = aka_tile
        wall.extend(suit_tiles)

    for honor in ("E", "S", "W", "N", "P", "F", "C"):
        wall.extend([honor] * 4)

    assert len(wall) == 136
    assert wall.count("5mr") == 1
    assert wall.count("5pr") == 1
    assert wall.count("5sr") == 1
    return wall


WALL_TILES = _build_wall_tiles()


def _shuffle_wall(seed: Optional[int] = None) -> List[str]:
    wall = WALL_TILES[:]
    if seed is not None:
        random.seed(seed)
    random.shuffle(wall)
    return wall


@dataclass
class BattleConfig:
    player_count: int = 4
    players: List[Dict] = field(default_factory=list)  # [{id, name, type}]
    target_score: int = 30000
    allow_west_round: bool = True
    allow_agari_yame: bool = True


@dataclass
class BattleRoom:
    game_id: str
    config: BattleConfig
    state: GameState
    wall: List[str] = field(default_factory=list)
    wall_index: int = 0
    dead_wall: List[str] = field(default_factory=list)
    rinshan_tiles: List[str] = field(default_factory=list)
    rinshan_index: int = 0
    dora_indicator_tiles: List[str] = field(default_factory=list)
    ura_indicator_tiles: List[str] = field(default_factory=list)
    phase: str = "waiting"  # waiting | playing | ended
    winner: Optional[int] = None
    human_player_id: int = 0  # 人类玩家所在的座位，-1 表示全Bot模式
    events: List[Dict] = field(default_factory=list)  # mjai 格式事件日志
    pending_rinshan: bool = False  # 是否需要摸岭上牌（大明杠/暗杠/加杠后）
    pending_kakan: Optional[Dict] = None
    replay_draw_actor: Optional[int] = None  # 当前展示中的“摸到第14张”状态
    last_heartbeat: float = field(
        default_factory=lambda: __import__("time").time()
    )  # 最后心跳时间
    disconnected: bool = False  # 人类玩家是否已断线

    def draw_tile(self) -> Optional[str]:
        if self.wall_index >= len(self.wall):
            return None
        tile = self.wall[self.wall_index]
        self.wall_index += 1
        return tile

    def draw_rinshan_tile(self) -> Optional[str]:
        if self.rinshan_index >= len(self.rinshan_tiles):
            return None
        tile = self.rinshan_tiles[self.rinshan_index]
        self.rinshan_index += 1
        return tile

    def remaining_wall(self) -> int:
        return len(self.wall) - self.wall_index


class BattleManager:
    MAX_ROOMS = 10

    def __init__(self):
        self.rooms: Dict[str, BattleRoom] = {}
        self.room_order: List[str] = []

    def create_room(
        self, config: BattleConfig, seed: Optional[int] = None
    ) -> BattleRoom:
        game_id = str(uuid.uuid4())[:8]
        room = BattleRoom(
            game_id=game_id,
            config=config,
            state=GameState(),
        )
        self.rooms[game_id] = room
        self.room_order.append(game_id)
        while len(self.room_order) > self.MAX_ROOMS:
            old_id = self.room_order.pop(0)
            self.rooms.pop(old_id, None)
        return room

    def get_room(self, game_id: str) -> Optional[BattleRoom]:
        return self.rooms.get(game_id)

    def close_room(self, game_id: str) -> None:
        """主动关闭并清理房间"""
        if game_id in self.room_order:
            self.room_order.remove(game_id)
        self.rooms.pop(game_id, None)

    def is_game_ended(self, room: BattleRoom) -> bool:
        """检查游戏是否应该结束。"""
        if min(room.state.scores) < 0:
            return True
        if room.state.bakaze not in ("S", "W"):
            return False

        is_all_last = room.state.kyoku == 4
        if not is_all_last:
            return False

        target_score = room.config.target_score
        max_score = max(room.state.scores)
        current_oya = room.state.oya
        top_score = max_score
        top_players = [pid for pid, score in enumerate(room.state.scores) if score == top_score]
        dealer_is_top = current_oya in top_players
        renchan = self._is_renchan(room)

        if room.state.bakaze == "S":
            if renchan:
                if (
                    room.config.allow_agari_yame
                    and dealer_is_top
                    and room.state.scores[current_oya] >= target_score
                ):
                    return True
                return False
            return max_score >= target_score and (room.state.bakaze != "W")

        # West round: stop once target reached on all-last or once renchan ends there.
        if renchan:
            if (
                room.config.allow_agari_yame
                and dealer_is_top
                and room.state.scores[current_oya] >= target_score
            ):
                return True
            return False
        return True

    def _expected_oya_for_kyoku(self, room: BattleRoom) -> int:
        return (room.state.kyoku - 1) % 4

    def _is_renchan(self, room: BattleRoom) -> bool:
        return room.state.oya == self._expected_oya_for_kyoku(room)

    def next_kyoku(self, room: BattleRoom) -> bool:
        """准备下一局，返回False表示游戏应该结束"""
        if self.is_game_ended(room):
            return False

        if self._is_renchan(room):
            return True

        room.state.kyoku += 1
        if room.state.kyoku > 4:
            room.state.kyoku = 1
            BAKAZE_ORDER = ["E", "S", "W"] if room.config.allow_west_round else ["E", "S"]
            current_idx = (
                BAKAZE_ORDER.index(room.state.bakaze)
                if room.state.bakaze in BAKAZE_ORDER
                else 0
            )
            next_idx = current_idx + 1
            if next_idx >= len(BAKAZE_ORDER):
                return False
            room.state.bakaze = BAKAZE_ORDER[next_idx]

        return True

    def start_kyoku(self, room: BattleRoom, seed: Optional[int] = None) -> None:
        wall = _shuffle_wall(seed)
        room.dead_wall = wall[-14:]
        room.rinshan_tiles = room.dead_wall[:4]
        room.rinshan_index = 0
        room.dora_indicator_tiles = room.dead_wall[4::2][:5]
        room.ura_indicator_tiles = room.dead_wall[5::2][:5]
        room.wall = wall[:-14]
        room.wall_index = 0

        bakaze = room.state.bakaze
        kyoku = room.state.kyoku
        honba = room.state.honba
        oya = room.state.oya

        tehais: List[List[str]] = []
        for pid in range(4):
            hand = []
            for _ in range(13):
                tile = room.draw_tile()
                if tile:
                    hand.append(tile)
            tehais.append(hand)

        dora_marker = room.dora_indicator_tiles[0] if room.dora_indicator_tiles else None

        room.state.in_game = True
        room.state.bakaze = bakaze
        room.state.kyoku = kyoku
        room.state.honba = honba
        room.state.oya = oya
        room.state.dora_markers = [dora_marker] if dora_marker else []
        room.state.ura_dora_markers = []
        room.state.actor_to_move = oya
        room.state.last_discard = None
        room.state.last_tsumo = [None, None, None, None]
        room.state.last_tsumo_raw = [None, None, None, None]
        room.state.remaining_wall = room.remaining_wall()
        room.replay_draw_actor = None

        for pid in range(4):
            room.state.players[pid] = PlayerState()
            for tile in tehais[pid]:
                room.state.players[pid].hand[tile] += 1

        room.phase = "playing"

        if not room.events:
            room.events.append(
                {
                    "type": "start_game",
                    "names": [p["name"] for p in room.config.players],
                    "kyoku_first": 0,
                    "aka_flag": True,
                }
            )

        room.events.append(
            {
                "type": "start_kyoku",
                "bakaze": bakaze,
                "dora_marker": dora_marker,
                "kyoku": kyoku,
                "honba": honba,
                "kyotaku": room.state.kyotaku,
                "oya": oya,
                "scores": room.state.scores[:],
                "tehais": tehais,
            }
        )

    def draw(self, room: BattleRoom, actor: int) -> Optional[str]:
        was_rinshan = room.pending_rinshan
        tile = room.draw_rinshan_tile() if was_rinshan else room.draw_tile()
        if tile:
            room.state.players[actor].hand[tile] += 1
            room.state.last_tsumo[actor] = tile
            room.state.last_tsumo_raw[actor] = tile
            room.state.players[actor].rinshan_tsumo = was_rinshan
            room.state.remaining_wall = room.remaining_wall()
            room.events.append(
                {"type": "tsumo", "actor": actor, "pai": tile, "rinshan": was_rinshan}
            )
        else:
            room.state.last_tsumo[actor] = None
            room.state.last_tsumo_raw[actor] = None
            room.state.players[actor].rinshan_tsumo = False
        room.state.actor_to_move = actor
        room.state.last_discard = None
        room.state.players[actor].doujun_furiten = False
        room.pending_rinshan = False
        room.replay_draw_actor = actor
        return tile

    def discard(
        self, room: BattleRoom, actor: int, pai: str, tsumogiri: bool = False
    ) -> bool:
        tile_key = _normalize_or_keep_aka(pai)
        hand = room.state.players[actor].hand
        # 优先用规范化后的 key
        if hand.get(tile_key, 0) > 0:
            hand[tile_key] -= 1
            if hand[tile_key] == 0:
                del hand[tile_key]
        # fallback：原始 pai
        elif hand.get(pai, 0) > 0:
            hand[pai] -= 1
            if hand[pai] == 0:
                del hand[pai]
            tile_key = pai
        # aka fallback：普通 5 被预测但手牌里有赤宝牌（5mr/5pr/5sr）
        elif pai[0] == "5" and pai[1] in "mps" and len(pai) == 2:
            aka_key = pai + "r"
            if hand.get(aka_key, 0) > 0:
                hand[aka_key] -= 1
                if hand[aka_key] == 0:
                    del hand[aka_key]
            else:
                raise TileStateError(
                    f"battle discard actor={actor}: missing tile {pai} "
                    f"(aka {aka_key}), hand={dict(hand)}"
                )
        else:
            raise TileStateError(
                f"battle discard actor={actor}: missing tile {pai}, hand={dict(hand)}"
            )

        reach_declared = room.state.players[actor].pending_reach
        room.state.players[actor].discards.append(
            {"pai": tile_key, "tsumogiri": tsumogiri, "reach_declared": reach_declared}
        )
        room.state.last_discard = {"actor": actor, "pai": tile_key, "pai_raw": pai}
        room.state.last_tsumo[actor] = None
        room.state.last_tsumo_raw[actor] = None
        room.state.players[actor].ippatsu_eligible = False
        room.state.players[actor].rinshan_tsumo = False
        room.replay_draw_actor = None

        room.state.actor_to_move = (actor + 1) % 4

        room.events.append(
            {
                "type": "dahai",
                "actor": actor,
                "pai": tile_key,
                "tsumogiri": tsumogiri,
            }
        )

        if reach_declared:
            room.state.scores[actor] -= 1000
            room.state.kyotaku += 1
            room.state.players[actor].reached = True
            room.state.players[actor].pending_reach = False
            room.events.append(
                {
                    "type": "reach_accepted",
                    "actor": actor,
                    "scores": room.state.scores[:],
                    "kyotaku": room.state.kyotaku,
                }
            )
        return True

    def handle_meld(
        self,
        room: BattleRoom,
        meld_type: str,
        actor: int,
        pai: str,
        consumed: List[str],
        target: Optional[int] = None,
    ) -> bool:
        for p in room.state.players:
            p.ippatsu_eligible = False
        consumed_keys = [_normalize_or_keep_aka(t) for t in consumed]
        hand = room.state.players[actor].hand
        if meld_type == "kakan":
            added_tile = _normalize_or_keep_aka(pai)
            if hand.get(added_tile, 0) <= 0:
                for t in consumed_keys:
                    if hand.get(t, 0) > 0:
                        added_tile = t
                        break
            if hand.get(added_tile, 0) <= 0:
                raise TileStateError(
                    f"battle handle_meld actor={actor}: missing tile {added_tile}, hand={dict(hand)}"
                )
            room.state.last_discard = None
            room.state.last_kakan = {
                "actor": actor,
                "pai": _normalize_or_keep_aka(pai),
                "pai_raw": _normalize_or_keep_aka(pai),
                "consumed": consumed_keys,
                "target": target,
            }
            room.state.last_tsumo[actor] = None
            room.state.last_tsumo_raw[actor] = None
            room.state.actor_to_move = actor
            room.pending_kakan = {
                "actor": actor,
                "pai": _normalize_or_keep_aka(pai),
                "consumed": consumed_keys,
                "target": target,
                "added_tile": added_tile,
            }
            room.events.append(
                {
                    "type": "kakan",
                    "actor": actor,
                    "pai": _normalize_or_keep_aka(pai),
                    "pai_raw": _normalize_or_keep_aka(pai),
                    "consumed": consumed_keys,
                }
            )
            return True
        tiles_to_remove = consumed_keys

        for t in tiles_to_remove:
            if hand.get(t, 0) <= 0:
                # aka fallback：预测普通5但手牌实际是赤宝牌
                if t[0] == "5" and t[1] in "mps" and len(t) == 2:
                    aka_key = t + "r"
                    if hand.get(aka_key, 0) > 0:
                        hand[aka_key] -= 1
                        if hand[aka_key] == 0:
                            del hand[aka_key]
                        continue
                raise TileStateError(
                    f"battle handle_meld actor={actor}: missing tile {t}, hand={dict(hand)}"
                )
            hand[t] -= 1
            if hand[t] == 0:
                del hand[t]

        if meld_type == "pon":
            room.state.players[actor].melds.append(
                {
                    "type": "pon",
                    "pai": normalize_tile(pai),
                    "pai_raw": _normalize_or_keep_aka(pai),
                    "consumed": consumed_keys,
                    "target": target,
                }
            )
        elif meld_type == "chi":
            room.state.players[actor].melds.append(
                {
                    "type": "chi",
                    "pai": normalize_tile(pai),
                    "pai_raw": _normalize_or_keep_aka(pai),
                    "consumed": consumed_keys,
                    "target": target,
                }
            )
        elif meld_type == "daiminkan":
            room.state.players[actor].melds.append(
                {
                    "type": "daiminkan",
                    "pai": normalize_tile(pai),
                    "pai_raw": _normalize_or_keep_aka(pai),
                    "consumed": consumed_keys,
                    "target": target,
                }
            )
            self._reveal_dora(room)
        elif meld_type == "ankan":
            consumed0 = consumed[0] if consumed else pai
            room.state.players[actor].melds.append(
                {
                    "type": "ankan",
                    "pai": normalize_tile(consumed0),
                    "pai_raw": _normalize_or_keep_aka(consumed0),
                    "consumed": consumed_keys,
                    "target": actor,
                }
            )
        room.state.last_discard = None
        room.state.last_kakan = None
        room.state.last_tsumo[actor] = None
        room.state.last_tsumo_raw[actor] = None
        room.state.actor_to_move = actor
        room.pending_rinshan = meld_type in ("daiminkan", "ankan")
        room.replay_draw_actor = actor if meld_type in ("chi", "pon") else None

        event = {
            "type": meld_type,
            "actor": actor,
            "pai": _normalize_or_keep_aka(pai),
            "consumed": consumed_keys,
        }
        if target is not None:
            event["target"] = target
        room.events.append(event)
        return True

    def accept_kakan(self, room: BattleRoom) -> List[Dict]:
        pending = room.pending_kakan
        if not pending:
            return []
        actor = pending["actor"]
        pai = pending["pai"]
        consumed_keys = list(pending.get("consumed", []))
        added_tile = pending["added_tile"]
        hand = room.state.players[actor].hand
        if hand.get(added_tile, 0) <= 0:
            raise TileStateError(
                f"battle accept_kakan actor={actor}: missing tile {added_tile}, hand={dict(hand)}"
            )
        hand[added_tile] -= 1
        if hand[added_tile] == 0:
            del hand[added_tile]

        upgraded = False
        for meld in room.state.players[actor].melds:
            if meld.get("type") != "pon":
                continue
            if normalize_tile(meld.get("pai", "")) != normalize_tile(pai):
                continue
            base_tiles = list(meld.get("consumed", []))
            called_tile = normalize_tile(meld.get("pai", pai))
            meld["type"] = "kakan"
            meld["pai"] = normalize_tile(pai)
            meld["pai_raw"] = _normalize_or_keep_aka(pai)
            meld["consumed"] = base_tiles + [meld.get("pai_raw", called_tile), added_tile]
            upgraded = True
            break
        if not upgraded:
            room.state.players[actor].melds.append(
                {
                    "type": "kakan",
                    "pai": normalize_tile(pai),
                    "pai_raw": _normalize_or_keep_aka(pai),
                    "consumed": consumed_keys + [added_tile],
                    "target": actor,
                }
            )

        room.state.last_kakan = None
        room.state.last_discard = None
        room.state.last_tsumo[actor] = None
        room.state.last_tsumo_raw[actor] = None
        room.state.actor_to_move = actor
        room.pending_kakan = None
        room.pending_rinshan = True
        room.replay_draw_actor = None

        start_idx = len(room.events)
        room.events.append(
            {
                "type": "kakan_accepted",
                "actor": actor,
                "pai": _normalize_or_keep_aka(pai),
                "pai_raw": _normalize_or_keep_aka(pai),
                "consumed": consumed_keys,
            }
        )
        self._reveal_dora(room)
        return room.events[start_idx:]

    def cancel_kakan(self, room: BattleRoom) -> None:
        room.pending_kakan = None
        room.state.last_kakan = None
        room.replay_draw_actor = None

    def reach(self, room: BattleRoom, actor: int) -> None:
        room.state.players[actor].pending_reach = True
        room.events.append({"type": "reach", "actor": actor})

    def _reveal_dora(self, room: BattleRoom) -> None:
        """从死牌山揭示下一张宝牌指示牌。"""
        next_index = len(room.state.dora_markers)
        if next_index >= len(room.dora_indicator_tiles):
            return
        dora = room.dora_indicator_tiles[next_index]
        room.state.dora_markers.append(dora)
        room.events.append({"type": "dora", "dora_marker": dora})

    def ryukyoku(self, room: BattleRoom, tenpai: List[int] = None) -> None:
        """流局处理
        tenpai: 哪些玩家听牌，默认None表示无人听牌
        """
        room.phase = "ended"
        room.replay_draw_actor = None

        if tenpai is None:
            tenpai = []

        oya = room.state.oya
        is_oya_tenpai = oya in tenpai

        room.state.honba += 1

        deltas = [0, 0, 0, 0]
        tenpai = sorted(set(tenpai))
        noten = [i for i in range(4) if i not in tenpai]
        if 0 < len(tenpai) < 4:
            reward = 3000 // len(tenpai)
            penalty = 3000 // len(noten)
            for i in tenpai:
                deltas[i] += reward
            for i in noten:
                deltas[i] -= penalty

        for i in range(4):
            room.state.scores[i] += deltas[i]

        if not is_oya_tenpai:
            room.state.oya = (oya + 1) % 4

        room.events.append(
            {
                "type": "ryukyoku",
                "deltas": deltas,
                "scores": room.state.scores[:],
                "honba": room.state.honba,
                "kyotaku": room.state.kyotaku,
                "tenpai_players": tenpai[:],
            }
        )
        room.events.append({"type": "end_kyoku"})

    def hora(
        self,
        room: BattleRoom,
        actor: int,
        target: int,
        pai: str,
        is_tsumo: bool = False,
    ) -> Dict:
        room.phase = "ended"
        room.winner = actor
        room.replay_draw_actor = None
        oya = room.state.oya
        is_oya_winner = actor == oya
        scoring = score_hora(
            room.state,
            actor=actor,
            target=target,
            pai=pai,
            is_tsumo=is_tsumo,
            ura_dora_markers=self._active_ura_markers(room, actor),
            is_rinshan=is_tsumo and room.state.players[actor].rinshan_tsumo,
            is_haitei=room.remaining_wall() == 0 and is_tsumo,
            is_houtei=room.remaining_wall() == 0 and not is_tsumo,
        )
        deltas = scoring.deltas
        hora_honba = room.state.honba
        hora_kyotaku = room.state.kyotaku
        ura_dora_markers = self._active_ura_markers(room, actor)

        if is_oya_winner:
            room.state.honba += 1
        else:
            room.state.honba = 0
        room.state.kyotaku = 0

        for i in range(4):
            room.state.scores[i] += deltas[i]

        if not is_oya_winner:
            room.state.oya = (oya + 1) % 4

        room.events.append(
            {
                "type": "hora",
                "actor": actor,
                "target": target,
                "pai": pai,
                "is_tsumo": is_tsumo,
                "deltas": deltas,
                "scores": room.state.scores[:],
                "han": scoring.han,
                "fu": scoring.fu,
                "yaku": scoring.yaku,
                "yaku_details": scoring.yaku_details,
                "cost": scoring.cost,
                "honba": hora_honba,
                "kyotaku": hora_kyotaku,
                "ura_dora_markers": ura_dora_markers,
            }
        )
        room.events.append({"type": "end_kyoku"})

        return {
            "type": "hora",
            "actor": actor,
            "target": target,
            "pai": pai,
            "is_tsumo": is_tsumo,
            "han": scoring.han,
            "fu": scoring.fu,
            "yaku": scoring.yaku,
            "yaku_details": scoring.yaku_details,
            "deltas": deltas,
            "cost": scoring.cost,
            "honba": hora_honba,
            "kyotaku": hora_kyotaku,
            "ura_dora_markers": ura_dora_markers,
        }

    def _active_ura_markers(self, room: BattleRoom, actor: int) -> List[str]:
        if not room.state.players[actor].reached:
            return []
        return room.ura_indicator_tiles[: len(room.state.dora_markers)]

    def get_snap_with_shanten(self, room: BattleRoom, actor: int) -> Dict:
        """返回注入了 shanten/waits 的 snapshot，供 legal_actions 判断立直/荣和用。"""
        from mahjong_env.replay import _calc_shanten_waits

        snap = room.state.snapshot(actor)
        try:
            hand_raw = snap.get("hand", [])
            hand_list = (
                hand_raw
                if isinstance(hand_raw, list)
                else [t for t, cnt in hand_raw.items() for _ in range(cnt)]
            )
            melds_list = (snap.get("melds") or [[], [], [], []])[actor]
            shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(
                hand_list, melds_list
            )
            snap["shanten"] = shanten
            snap["waits_count"] = waits_cnt
            snap["waits_tiles"] = waits_tiles
        except Exception as e:
            print(f"[WARN] get_snap_with_shanten failed for actor {actor}: {e}")
            snap["shanten"] = 8
            snap["waits_count"] = 0
            snap["waits_tiles"] = [False] * 34
        return snap

    def prepare_turn(self, room: BattleRoom, actor: int) -> Optional[str]:
        """为 actor 准备回合：如需摸牌则摸牌，返回摸到的牌（或 None）。
        适用于 bot 和人类，流局时返回 None 并设置 room.phase=ended。"""
        last_discard = room.state.last_discard
        is_response = last_discard and last_discard.get("actor") != actor
        is_kakan_response = room.state.last_kakan and room.state.last_kakan.get("actor") != actor
        if is_response:
            return None  # 响应他家弃牌，不需摸牌
        if is_kakan_response:
            return None

        if room.pending_rinshan:
            if room.remaining_wall() < 1:
                self.ryukyoku(room)
                return None
            return self.draw(room, actor)

        if room.state.last_tsumo[actor] is not None:
            return room.state.last_tsumo[actor]  # 已有摸牌

        hand_size = sum(room.state.players[actor].hand.values())
        if hand_size in (11, 12):  # 碰/吃后直接打牌
            return None

        if room.remaining_wall() < 1:
            self.ryukyoku(room)
            return None
        return self.draw(room, actor)

    def apply_action(self, room: BattleRoom, actor: int, action: Dict) -> bool:
        """应用一个动作（dahai/pon/chi/kan/hora/reach/none），返回是否成功。
        适用于 bot 和人类，统一入口。"""
        action_type = action.get("type")
        pai = action.get("pai", "")
        consumed = action.get("consumed", [])
        target = action.get("target", actor)
        tsumogiri = action.get("tsumogiri", False)
        last_discard = room.state.last_discard
        is_response = last_discard and last_discard.get("actor") != actor
        last_kakan = room.state.last_kakan
        is_kakan_response = last_kakan and last_kakan.get("actor") != actor

        if action_type == "dahai":
            self.discard(room, actor, pai, tsumogiri=tsumogiri)
        elif action_type in ("pon", "chi", "daiminkan"):
            self.handle_meld(room, action_type, actor, pai, consumed, target=target)
        elif action_type == "ankan":
            self.handle_meld(room, "ankan", actor, pai, consumed or [pai] * 4)
        elif action_type == "kakan":
            self.handle_meld(room, "kakan", actor, pai, consumed)
        elif action_type == "reach":
            self.reach(room, actor)
        elif action_type == "hora":
            is_tsumo = target == actor
            self.hora(room, actor, target, pai, is_tsumo=is_tsumo)
        elif action_type == "none":
            if is_response:
                discarder = last_discard["actor"]
                next_actor = (actor + 1) % 4
                if next_actor == discarder:
                    room.state.last_discard = None
                    room.state.actor_to_move = (discarder + 1) % 4
                else:
                    room.state.actor_to_move = next_actor
            elif is_kakan_response:
                room.state.actor_to_move = last_kakan["actor"]
            else:
                # 摸牌回合 none → 摸切
                tsumo = room.state.last_tsumo_raw[actor] or room.state.last_tsumo[actor]
                if tsumo:
                    self.discard(room, actor, tsumo, tsumogiri=True)
        else:
            print(f"[WARN] apply_action: unknown action type {action_type}")
            return False
        return True

    def process_human_action(
        self, room: BattleRoom, actor: int, action: Dict
    ) -> Optional[str]:
        """处理人类玩家的动作。返回 None 表示成功，返回错误字符串表示失败。"""
        is_response = bool(
            room.state.last_discard and room.state.last_discard.get("actor") != actor
        )
        is_kakan_response = bool(
            room.state.last_kakan and room.state.last_kakan.get("actor") != actor
        )
        if room.state.actor_to_move != actor and not is_response and not is_kakan_response:
            return "Not your turn"

        # 准备回合（摸牌/流局判断）
        self.prepare_turn(room, actor)
        if room.phase == "ended":
            return None

        # 枚举合法动作并验证
        snap = self.get_snap_with_shanten(room, actor)
        legal_specs = enumerate_legal_action_specs(snap, actor)
        requested_spec = action_dict_to_spec(action, actor_hint=actor)
        matched_spec = next((spec for spec in legal_specs if action_specs_match(spec, requested_spec)), None)
        if matched_spec is None:
            return "Illegal action"

        # 应用动作
        self.apply_action(room, actor, matched_spec.to_mjai())
        return None

    def get_state_for_player(self, room: BattleRoom, player_id: int) -> Dict:
        snap = self.get_snap_with_shanten(room, player_id)

        legal_specs = enumerate_legal_action_specs(snap, player_id)
        # 响应弃牌时（last_discard 来自他家），保留 none（skip）
        # 自己摸牌打牌回合，none 无意义，过滤掉
        is_response = (
            snap.get("last_discard") and snap["last_discard"].get("actor") != player_id
        )
        is_kakan_response = (
            snap.get("last_kakan") and snap["last_kakan"].get("actor") != player_id
        )
        legal_actions = []
        for spec in legal_specs:
            if spec.type == "none" and not is_response and not is_kakan_response:
                continue
            action_dict = spec.to_mjai()
            if spec.type != "none" and "actor" not in action_dict:
                action_dict["actor"] = player_id if spec.actor is None else spec.actor
            legal_actions.append(action_dict)

        return {
            "game_id": room.game_id,
            "phase": room.phase,
            "winner": room.winner,
            "bakaze": room.state.bakaze,
            "kyoku": room.state.kyoku,
            "honba": room.state.honba,
            "kyotaku": room.state.kyotaku,
            "oya": room.state.oya,
            "scores": room.state.scores[:],
            "dora_markers": room.state.dora_markers[:],
            "actor_to_move": room.state.actor_to_move,
            "last_discard": snap["last_discard"],
            "last_kakan": snap.get("last_kakan"),
            "hand": snap["hand"],
            "tsumo_pai": room.state.last_tsumo[player_id],
            "discards": [p.discards[:] for p in room.state.players],
            "melds": [p.melds[:] for p in room.state.players],
            "reached": [p.reached for p in room.state.players],
            "pending_reach": [p.pending_reach for p in room.state.players],
            "replay_draw_actor": room.replay_draw_actor,
            "legal_actions": legal_actions,
            "remaining_wall": room.remaining_wall(),
            "human_player_id": room.human_player_id,
            "player_info": [
                {
                    "player_id": pid,
                    "name": room.config.players[pid]["name"]
                    if pid < len(room.config.players)
                    else f"Player{pid}",
                    "type": room.config.players[pid]["type"]
                    if pid < len(room.config.players)
                    else "bot",
                }
                for pid in range(4)
            ],
        }


_manager: Optional[BattleManager] = None


def get_manager() -> BattleManager:
    global _manager
    if _manager is None:
        _manager = BattleManager()
    return _manager
