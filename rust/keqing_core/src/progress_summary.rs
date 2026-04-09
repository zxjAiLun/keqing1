//! Summary-oriented helpers for 3n+1 progress analysis.

use crate::counts::TILE_COUNT;
use crate::progress_delta::{calc_discard_deltas, calc_draw_deltas, calc_required_tiles};
use crate::shanten_table::{calc_shanten_all, calc_shanten_normal};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary3n1 {
    pub shanten: i8,
    pub waits_count: u8,
    pub waits_tiles: [bool; TILE_COUNT],
    pub tehai_count: u8,
    pub ukeire_type_count: u8,
    pub ukeire_live_count: u8,
    pub ukeire_tiles: [bool; TILE_COUNT],
}

fn is_suited_sequence_start(tile34: usize) -> bool {
    tile34 < 27 && (tile34 % 9) <= 6
}

fn is_complete_regular_counts(
    counts: &mut [u8; TILE_COUNT],
    melds_needed: u8,
    need_pair: bool,
) -> bool {
    let first = counts.iter().position(|&cnt| cnt > 0);
    let Some(first) = first else {
        return melds_needed == 0 && !need_pair;
    };

    if need_pair && counts[first] >= 2 {
        counts[first] -= 2;
        if is_complete_regular_counts(counts, melds_needed, false) {
            counts[first] += 2;
            return true;
        }
        counts[first] += 2;
    }

    if melds_needed > 0 && counts[first] >= 3 {
        counts[first] -= 3;
        if is_complete_regular_counts(counts, melds_needed - 1, need_pair) {
            counts[first] += 3;
            return true;
        }
        counts[first] += 3;
    }

    if melds_needed > 0
        && is_suited_sequence_start(first)
        && counts[first + 1] > 0
        && counts[first + 2] > 0
    {
        counts[first] -= 1;
        counts[first + 1] -= 1;
        counts[first + 2] -= 1;
        if is_complete_regular_counts(counts, melds_needed - 1, need_pair) {
            counts[first] += 1;
            counts[first + 1] += 1;
            counts[first + 2] += 1;
            return true;
        }
        counts[first] += 1;
        counts[first + 1] += 1;
        counts[first + 2] += 1;
    }

    false
}

fn find_regular_waits(counts34: &[u8; TILE_COUNT]) -> [bool; TILE_COUNT] {
    let mut waits = [false; TILE_COUNT];
    let tile_count: usize = counts34.iter().map(|&v| usize::from(v)).sum();
    if tile_count % 3 != 1 {
        return waits;
    }

    let melds_needed = (((tile_count + 1) - 2) / 3) as u8;
    for tile34 in 0..TILE_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        waits[tile34] = is_complete_regular_counts(&mut work, melds_needed, true);
    }
    waits
}

pub fn summarize_3n1(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> Summary3n1 {
    let tehai_count: u8 = counts34.iter().sum();
    if tehai_count == 0 {
        return Summary3n1 {
            shanten: 8,
            waits_count: 0,
            waits_tiles: [false; TILE_COUNT],
            tehai_count: 0,
            ukeire_type_count: 0,
            ukeire_live_count: 0,
            ukeire_tiles: [false; TILE_COUNT],
        };
    }

    let len_div3 = tehai_count / 3;
    let shanten = calc_shanten_all(counts34, len_div3);
    let regular_shanten = calc_shanten_normal(counts34, len_div3);
    let waits_tiles = if shanten <= 1 && regular_shanten == shanten {
        find_regular_waits(counts34)
    } else {
        [false; TILE_COUNT]
    };
    let waits_count = waits_tiles.iter().filter(|&&flag| flag).count() as u8;

    let mut ukeire_tiles = [false; TILE_COUNT];
    if shanten > 0 {
        if shanten > 2 {
            for item in calc_required_tiles(counts34, visible_counts34, len_div3) {
                ukeire_tiles[item.tile34 as usize] = true;
            }
        } else {
            for draw in calc_draw_deltas(counts34, visible_counts34, len_div3) {
                if draw.shanten_diff < 0 {
                    ukeire_tiles[draw.tile34 as usize] = true;
                    continue;
                }
                let mut counts_3n2 = *counts34;
                counts_3n2[draw.tile34 as usize] += 1;
                let discard_deltas = calc_discard_deltas(&counts_3n2, len_div3);
                if discard_deltas
                    .iter()
                    .any(|discard| draw.shanten_diff + discard.shanten_diff < 0)
                {
                    ukeire_tiles[draw.tile34 as usize] = true;
                }
            }
        }
    }

    let mut ukeire_type_count = 0u8;
    let mut ukeire_live_count = 0u8;
    for (tile34, &flag) in ukeire_tiles.iter().enumerate() {
        if !flag {
            continue;
        }
        ukeire_type_count += 1;
        ukeire_live_count = ukeire_live_count.saturating_add(4u8.saturating_sub(visible_counts34[tile34]));
    }

    Summary3n1 {
        shanten,
        waits_count,
        waits_tiles,
        tehai_count,
        ukeire_type_count,
        ukeire_live_count,
        ukeire_tiles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(entries: &[(usize, u8)]) -> [u8; TILE_COUNT] {
        let mut tiles = [0u8; TILE_COUNT];
        for &(idx, count) in entries {
            tiles[idx] = count;
        }
        tiles
    }

    #[test]
    fn summarize_3n1_reports_waits_for_tenpai_hand() {
        let counts = hand(&[
            (4, 1), (5, 1),
            (17, 3),
            (21, 3),
            (22, 1), (23, 1), (24, 1),
            (32, 2),
        ]);
        let summary = summarize_3n1(&counts, &counts);
        assert_eq!(summary.shanten, 0);
        assert!(summary.waits_count > 0);
    }

    #[test]
    fn summarize_3n1_reports_ukeire_for_non_tenpai_hand() {
        let counts = hand(&[
            (0, 1), (1, 1), (2, 1),
            (3, 1), (4, 1), (5, 1),
            (6, 1), (7, 1), (9, 1),
            (10, 1), (12, 1), (13, 1),
            (18, 1),
        ]);
        let summary = summarize_3n1(&counts, &counts);
        assert!(summary.shanten > 0);
        assert!(summary.ukeire_type_count > 0);
    }
}
