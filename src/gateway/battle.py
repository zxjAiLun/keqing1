from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState, PlayerState, apply_event
from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES

AKA5 = ("5mr", "5pr", "5sr")


def _normalize_or_keep_aka(tile: str) -> str:
    """规范化 tile，但保留赤宝牌的原样"""
    if tile in AKA5:
        return tile
    return normalize_tile(tile)


# 牌山：一副136张
WALL_TILES = (
    [f"{n}m" for n in range(1, 10)] * 4
    + [f"{n}p" for n in range(1, 10)] * 4
    + [f"{n}s" for n in range(1, 10)] * 4
    + ["E", "S", "W", "N"] * 4
    + ["P", "F", "C"] * 4
)


def _shuffle_wall(seed: Optional[int] = None) -> List[str]:
    wall = WALL_TILES[:]
    if seed is not None:
        random.seed(seed)
    random.shuffle(wall)
    return wall


@dataclass
class SeatState:
    player_id: int
    name: str
    player_type: str  # "human" | "bot"
    hand: List[str] = field(default_factory=list)
    discards: List[Dict] = field(default_factory=list)
    melds: List[Dict] = field(default_factory=list)
    reached: bool = False
    pending_reach: bool = False


@dataclass
class BattleConfig:
    player_count: int = 4
    players: List[Dict] = field(default_factory=list)  # [{id, name, type}]


@dataclass
class BattleRoom:
    game_id: str
    config: BattleConfig
    state: GameState
    wall: List[str] = field(default_factory=list)
    wall_index: int = 0
    phase: str = "waiting"  # waiting | playing | ended
    winner: Optional[int] = None
    human_player_id: int = 0  # 人类玩家所在的座位，-1 表示全Bot模式
    events: List[Dict] = field(default_factory=list)  # mjai 格式事件日志
    pending_rinshan: bool = False  # 是否需要摸岭上牌（大明杠/暗杠/加杠后）

    def draw_tile(self) -> Optional[str]:
        if self.wall_index >= len(self.wall):
            return None
        tile = self.wall[self.wall_index]
        self.wall_index += 1
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
        """检查游戏是否应该结束"""
        max_score = max(room.state.scores)
        if max_score >= 30000:
            return True
        return False

    def next_kyoku(self, room: BattleRoom) -> bool:
        """准备下一局，返回False表示游戏应该结束"""
        if self.is_game_ended(room):
            return False

        room.state.kyoku += 1
        if room.state.kyoku > 4:
            room.state.kyoku = 1
            BAKAZE_ORDER = ["E", "S", "W", "N"]
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
        room.wall = wall
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

        dora_marker = room.draw_tile()

        room.state.in_game = True
        room.state.bakaze = bakaze
        room.state.kyoku = kyoku
        room.state.honba = honba
        room.state.oya = oya
        room.state.dora_markers = [dora_marker] if dora_marker else []
        room.state.actor_to_move = oya
        room.state.last_discard = None
        room.state.last_tsumo = [None, None, None, None]
        room.state.last_tsumo_raw = [None, None, None, None]

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
        tile = room.draw_tile()
        if tile:
            room.state.players[actor].hand[tile] += 1
            room.state.last_tsumo[actor] = tile
            room.state.last_tsumo_raw[actor] = tile
            room.events.append({"type": "tsumo", "actor": actor, "pai": tile})
        else:
            room.state.last_tsumo[actor] = None
            room.state.last_tsumo_raw[actor] = None
        room.state.actor_to_move = actor
        room.state.last_discard = None
        room.state.players[actor].doujun_furiten = False
        room.pending_rinshan = False
        return tile

    def discard(
        self, room: BattleRoom, actor: int, pai: str, tsumogiri: bool = False
    ) -> bool:
        tile_key = _normalize_or_keep_aka(pai)
        if room.state.players[actor].hand.get(tile_key, 0) > 0:
            room.state.players[actor].hand[tile_key] -= 1
            if room.state.players[actor].hand[tile_key] == 0:
                del room.state.players[actor].hand[tile_key]
        elif room.state.players[actor].hand.get(pai, 0) > 0:
            room.state.players[actor].hand[pai] -= 1
            if room.state.players[actor].hand[pai] == 0:
                del room.state.players[actor].hand[pai]
            tile_key = pai
        else:
            print(
                f"[WARN] discard: actor {actor} missing tile {pai}, hand={dict(room.state.players[actor].hand)}"
            )
            return False

        reach_declared = room.state.players[actor].pending_reach
        room.state.players[actor].discards.append(
            {"pai": tile_key, "tsumogiri": tsumogiri, "reach_declared": reach_declared}
        )
        room.state.last_discard = {"actor": actor, "pai": tile_key, "pai_raw": pai}
        room.state.last_tsumo[actor] = None
        room.state.last_tsumo_raw[actor] = None

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
            room.state.players[actor].reached = True
            room.state.players[actor].pending_reach = False
            room.events.append({"type": "reach_accepted", "actor": actor})
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
        consumed_keys = [_normalize_or_keep_aka(t) for t in consumed]
        for t in consumed_keys:
            if room.state.players[actor].hand.get(t, 0) <= 0:
                print(
                    f"[WARN] handle_meld: actor {actor} missing tile {t}, hand={dict(room.state.players[actor].hand)}"
                )
                return False
            room.state.players[actor].hand[t] -= 1
            if room.state.players[actor].hand[t] == 0:
                del room.state.players[actor].hand[t]

        if meld_type == "pon":
            room.state.players[actor].melds.append(
                {
                    "type": "pon",
                    "pai": normalize_tile(pai),
                    "consumed": consumed_keys,
                    "target": target,
                }
            )
        elif meld_type == "chi":
            room.state.players[actor].melds.append(
                {
                    "type": "chi",
                    "pai": normalize_tile(pai),
                    "consumed": consumed_keys,
                    "target": target,
                }
            )
        elif meld_type == "daiminkan":
            room.state.players[actor].melds.append(
                {
                    "type": "daiminkan",
                    "pai": normalize_tile(pai),
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
                    "consumed": consumed_keys,
                    "target": actor,
                }
            )
        elif meld_type == "kakan":
            room.state.players[actor].melds.append(
                {
                    "type": "kakan",
                    "pai": normalize_tile(pai),
                    "consumed": consumed_keys + [normalize_tile(pai)],
                    "target": actor,
                }
            )
            self._reveal_dora(room)

        room.state.last_discard = None
        room.state.last_tsumo[actor] = None
        room.state.last_tsumo_raw[actor] = None
        room.state.actor_to_move = actor
        room.pending_rinshan = meld_type in ("daiminkan", "ankan", "kakan")

        event = {
            "type": meld_type,
            "actor": actor,
            "pai": _normalize_or_keep_aka(pai),
            "consumed": consumed_keys,
        }
        if target is not None:
            event["target"] = target
        room.events.append(event)

    def reach(self, room: BattleRoom, actor: int) -> None:
        room.state.players[actor].pending_reach = True
        room.events.append({"type": "reach", "actor": actor})

    def _reveal_dora(self, room: BattleRoom) -> None:
        """从牌山揭示宝牌指示牌"""
        dora = room.draw_tile()
        if dora:
            room.state.dora_markers.append(dora)
            room.events.append({"type": "dora", "dora_marker": dora})

    def ryukyoku(self, room: BattleRoom, tenpai: List[int] = None) -> None:
        """流局处理
        tenpai: 哪些玩家听牌，默认None表示无人听牌
        """
        room.phase = "ended"

        if tenpai is None:
            tenpai = []

        oya = room.state.oya
        is_oya_tenpai = oya in tenpai

        room.state.honba += 1

        deltas = [0, 0, 0, 0]

        for i in range(4):
            if i not in tenpai:
                penalty = 1000 if is_oya_tenpai and i == oya else 500
                deltas[i] = -penalty
                room.state.kyotaku += penalty

        for i in range(4):
            room.state.scores[i] += deltas[i]

        if not is_oya_tenpai:
            room.state.oya = (oya + 1) % 4

        room.events.append(
            {
                "type": "ryukyoku",
                "deltas": deltas,
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
        score_delta: int = 3000,
    ) -> Dict:
        room.phase = "ended"
        room.winner = actor

        yakuman = 1
        han = 3 * yakuman

        oya = room.state.oya
        is_oya_winner = actor == oya

        deltas = [0, 0, 0, 0]

        if is_tsumo:
            for i in range(4):
                if i == actor:
                    continue
                payment = score_delta if is_oya_winner else score_delta // 2
                deltas[i] = -payment
                deltas[actor] += payment
        else:
            deltas[actor] = score_delta
            deltas[target] = -score_delta

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
            }
        )
        room.events.append({"type": "end_kyoku"})

        return {
            "type": "hora",
            "actor": actor,
            "target": target,
            "pai": pai,
            "is_tsumo": is_tsumo,
            "han": han,
            "fu": 30,
        }

    def get_state_for_player(self, room: BattleRoom, player_id: int) -> Dict:
        snap = room.state.snapshot(player_id)

        print(
            f"[DEBUG] get_state_for_player: player_id={player_id}, actor_to_move={room.state.actor_to_move}, hand={snap.get('hand', [])}, last_tsumo={room.state.last_tsumo[player_id]}"
        )

        legal = enumerate_legal_actions(snap, player_id)
        print(
            f"[DEBUG] enumerate_legal_actions returned: {[(a.type, a.pai) for a in legal]}"
        )
        legal_actions = []
        for a in legal:
            if a.type == "none":
                continue
            action_dict = {
                "type": a.type,
                "actor": a.actor,
            }
            if a.pai:
                action_dict["pai"] = a.pai
            if a.target is not None:
                action_dict["target"] = a.target
            if a.consumed:
                action_dict["consumed"] = a.consumed
            if a.tsumogiri:
                action_dict["tsumogiri"] = a.tsumogiri
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
            "hand": snap["hand"],
            "tsumo_pai": room.state.last_tsumo[player_id],
            "discards": [p.discards[:] for p in room.state.players],
            "melds": [p.melds[:] for p in room.state.players],
            "reached": [p.reached for p in room.state.players],
            "pending_reach": [p.pending_reach for p in room.state.players],
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
