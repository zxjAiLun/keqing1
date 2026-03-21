from __future__ import annotations

from typing import List

SUITED = ("m", "p", "s")
HONORS = ("E", "S", "W", "N", "P", "F", "C")

AKA_DORA_TILES = ("5mr", "5pr", "5sr")

NORMAL_SUITED_TILES = ("1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m",
                       "1p", "2p", "3p", "4p", "6p", "7p", "8p", "9p",
                       "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s")


def normalize_tile(tile: str) -> str:
    if tile.endswith("r"):
        return tile[:-1]
    return tile


def is_aka_dora(tile: str) -> bool:
    return tile in AKA_DORA_TILES


def all_discardable_tiles() -> List[str]:
    tiles: List[str] = []
    for suit in SUITED:
        for n in range(1, 10):
            tiles.append(f"{n}{suit}")
    tiles.extend(HONORS)
    return tiles


def all_discardable_tiles_with_aka() -> List[str]:
    tiles = []
    for suit in SUITED:
        for n in range(1, 10):
            if n == 5:
                tiles.append(f"5{suit}r")
            else:
                tiles.append(f"{n}{suit}")
    tiles.extend(HONORS)
    return tiles


def tile_without_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return normalize_tile(tile)
    return tile
