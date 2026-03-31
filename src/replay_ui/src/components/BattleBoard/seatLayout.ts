import type { MeldEntry } from "../../types/battle";

export type SeatPosition = "south" | "north" | "east" | "west";
export type LayoutAxis = "row" | "column";
export type RelativeCallSide = "left" | "across" | "right" | "self";

export interface SeatModel {
  position: SeatPosition;
  tileOrientation: 0 | 90 | 180 | 270;
  concealedAxis: LayoutAxis;
  concealedReverse: boolean;
  meldAxis: LayoutAxis;
  meldPlacement: "before" | "after";
  labelPlacement: "top" | "bottom" | "left" | "right";
}

export interface MeldDisplayTile {
  tile: string;
  rotated: boolean;
  stackedOn?: number;
  hidden?: boolean;
}

const SEAT_MODELS: Record<SeatPosition, SeatModel> = {
  south: {
    position: "south",
    tileOrientation: 0,
    concealedAxis: "row",
    concealedReverse: false,
    meldAxis: "row",
    meldPlacement: "after",
    labelPlacement: "bottom",
  },
  north: {
    position: "north",
    tileOrientation: 180,
    concealedAxis: "row",
    concealedReverse: true,
    meldAxis: "row",
    meldPlacement: "before",
    labelPlacement: "top",
  },
  east: {
    position: "east",
    tileOrientation: 270,
    concealedAxis: "column",
    concealedReverse: true,
    meldAxis: "column",
    meldPlacement: "after",
    labelPlacement: "right",
  },
  west: {
    position: "west",
    tileOrientation: 90,
    concealedAxis: "column",
    concealedReverse: false,
    meldAxis: "column",
    meldPlacement: "before",
    labelPlacement: "left",
  },
};

export function getSeatModel(position: SeatPosition): SeatModel {
  return SEAT_MODELS[position];
}

export function getRelativeCallSide(actor: number, target: number | null | undefined): RelativeCallSide {
  if (target == null || target === actor) return "self";
  const delta = (target - actor + 4) % 4;
  if (delta === 1) return "right";
  if (delta === 2) return "across";
  if (delta === 3) return "left";
  return "self";
}

function getClaimedTileIndex(tileCount: number, side: RelativeCallSide): number {
  if (tileCount <= 1) return 0;
  if (side === "left") return 0;
  if (side === "right") return tileCount - 1;
  if (side === "across") return Math.floor((tileCount - 1) / 2);
  return tileCount - 1;
}

function buildCalledMeldTiles(
  actor: number,
  target: number | null | undefined,
  handTiles: string[],
  calledTile: string,
): MeldDisplayTile[] {
  const claimedIndex = getClaimedTileIndex(handTiles.length + 1, getRelativeCallSide(actor, target));
  const displayTiles: MeldDisplayTile[] = [];
  let handIndex = 0;
  for (let idx = 0; idx < handTiles.length + 1; idx++) {
    if (idx === claimedIndex) {
      displayTiles.push({ tile: calledTile, rotated: true });
    } else {
      displayTiles.push({ tile: handTiles[handIndex], rotated: false });
      handIndex += 1;
    }
  }
  return displayTiles;
}

export function buildMeldDisplayTiles(actor: number, meld: MeldEntry): MeldDisplayTile[] {
  if (meld.type === "kakan" && meld.consumed.length >= 4) {
    const baseHandTiles = meld.consumed.slice(0, 2);
    const calledTile = meld.consumed[2] ?? meld.pai;
    const addedTile = meld.consumed[3];
    const baseDisplayTiles = buildCalledMeldTiles(actor, meld.target, baseHandTiles, calledTile);
    const claimedIndex = baseDisplayTiles.findIndex(tile => tile.rotated);
    return [...baseDisplayTiles, { tile: addedTile, rotated: false, stackedOn: claimedIndex }];
  }

  if (meld.type === "ankan") {
    const tiles = meld.consumed.slice(0, 4);
    return tiles.map((tile, idx) => ({
      tile,
      rotated: false,
      hidden: idx === 0 || idx === tiles.length - 1,
    }));
  }

  return buildCalledMeldTiles(actor, meld.target, [...meld.consumed], meld.pai);
}
