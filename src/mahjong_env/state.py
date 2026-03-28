from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES
from mahjong_env.types import MjaiEvent


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    return normalize_tile(tile)


@dataclass
class PlayerState:
    hand: Counter = field(default_factory=Counter)
    discards: List[str] = field(default_factory=list)
    melds: List[dict] = field(default_factory=list)
    reached: bool = False
    pending_reach: bool = False


@dataclass
class GameState:
    bakaze: str = "E"
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    dora_markers: List[str] = field(default_factory=list)
    scores: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    players: List[PlayerState] = field(default_factory=lambda: [PlayerState() for _ in range(4)])
    actor_to_move: Optional[int] = None
    last_discard: Optional[dict] = None
    last_tsumo: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    last_tsumo_raw: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    in_game: bool = False

    def snapshot(self, actor: int) -> Dict:
        hand_counter = self.players[actor].hand
        hand_list: List[str] = []
        for tile, cnt in sorted(hand_counter.items()):
            hand_list.extend([tile] * cnt)
        return {
            "bakaze": self.bakaze,
            "kyoku": self.kyoku,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "oya": self.oya,
            "scores": self.scores[:],
            "dora_markers": self.dora_markers[:],
            "actor": actor,
            "hand": hand_list,
            "discards": [p.discards[:] for p in self.players],
            "melds": [p.melds[:] for p in self.players],
            "reached": [p.reached for p in self.players],
            "actor_to_move": self.actor_to_move,
            "last_discard": self.last_discard.copy() if self.last_discard else None,
            "last_tsumo": self.last_tsumo[:],
            "last_tsumo_raw": self.last_tsumo_raw[:],
        }


def _remove_tile(counter: Counter, tile: str, n: int = 1) -> bool:
    if counter.get(tile, 0) < n:
        counter[tile] = n
    counter[tile] -= n
    if counter[tile] == 0:
        del counter[tile]
    return True


def apply_event(state: GameState, event: MjaiEvent) -> None:
    et = event["type"]
    if et == "start_game":
        state.in_game = True
        return
    if et == "start_kyoku":
        state.bakaze = event["bakaze"]
        state.kyoku = event["kyoku"]
        state.honba = event["honba"]
        state.kyotaku = event["kyotaku"]
        state.oya = event["oya"]
        state.scores = event["scores"][:]
        state.dora_markers = [_normalize_or_keep_aka(event["dora_marker"])]
        state.last_discard = None
        state.actor_to_move = state.oya
        state.last_tsumo = [None, None, None, None]
        state.last_tsumo_raw = [None, None, None, None]
        for pid in range(4):
            state.players[pid] = PlayerState()
            for tile in event["tehais"][pid]:
                if tile != "?":
                    state.players[pid].hand[_normalize_or_keep_aka(tile)] += 1
        return

    if et == "tsumo":
        actor = event["actor"]
        pai = event["pai"]
        if pai != "?":
            tile_key = _normalize_or_keep_aka(pai)
            state.players[actor].hand[tile_key] += 1
            state.last_tsumo[actor] = tile_key
            state.last_tsumo_raw[actor] = pai
        else:
            state.last_tsumo[actor] = None
            state.last_tsumo_raw[actor] = None
        state.actor_to_move = actor
        state.last_discard = None
        return

    if et == "dahai":
        actor = event["actor"]
        pai_raw = event["pai"]
        tile_key = _normalize_or_keep_aka(pai_raw)
        _remove_tile(state.players[actor].hand, tile_key, 1)
        state.players[actor].discards.append(tile_key)
        state.last_discard = {"actor": actor, "pai": tile_key, "pai_raw": pai_raw}
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        state.actor_to_move = (actor + 1) % 4
        if state.players[actor].pending_reach:
            state.players[actor].reached = True
            state.players[actor].pending_reach = False
        return

    if et in ("pon", "chi", "daiminkan"):
        actor = event["actor"]
        consumed = [_normalize_or_keep_aka(x) for x in event.get("consumed", [])]
        for t in consumed:
            _remove_tile(state.players[actor].hand, t, 1)
        state.players[actor].melds.append(
            {
                "type": et,
                "pai": _normalize_or_keep_aka(event["pai"]),
                "consumed": consumed,
                "target": event.get("target"),
            }
        )
        state.last_discard = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        state.actor_to_move = actor
        return

    if et == "ankan":
        actor = event["actor"]
        consumed = [_normalize_or_keep_aka(x) for x in event["consumed"]]
        for t in consumed:
            _remove_tile(state.players[actor].hand, t, 1)
        ankan_pai = _normalize_or_keep_aka(event["consumed"][0]) if "pai" not in event else _normalize_or_keep_aka(event["pai"])
        state.players[actor].melds.append(
            {"type": "ankan", "pai": ankan_pai, "consumed": consumed, "target": actor}
        )
        state.actor_to_move = actor
        state.last_discard = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        return

    if et == "kakan":
        actor = event["actor"]
        pai = _normalize_or_keep_aka(event["pai"])
        consumed = [_normalize_or_keep_aka(x) for x in event.get("consumed", [])]
        for t in consumed:
            _remove_tile(state.players[actor].hand, t, 1)
        _remove_tile(state.players[actor].hand, pai, 1)
        state.players[actor].melds.append(
            {"type": "kakan", "pai": pai, "consumed": consumed + [pai], "target": actor}
        )
        state.actor_to_move = actor
        state.last_discard = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        return

    if et == "reach":
        actor = event["actor"]
        state.players[actor].pending_reach = True
        return

    if et == "reach_accepted":
        return

    if et in ("hora", "ryukyoku", "end_kyoku", "end_game"):
        state.actor_to_move = None
        state.last_discard = None
        return

    if et == "dora":
        state.dora_markers.append(normalize_tile(event["dora_marker"]))
        return


def visible_tiles_for_actor(state: GameState, actor: int) -> Set[str]:
    visible: Set[str] = set(state.players[actor].hand.keys())
    for p in state.players:
        visible.update(p.discards)
        for meld in p.melds:
            visible.update(meld.get("consumed", []))
            if "pai" in meld:
                visible.add(meld["pai"])
    return visible

