//! Rust-side Xmodel1 discard export.
//!
//! This module now produces real candidate-centric `.npz` exports instead of
//! placeholder smoke tensors. The implementation intentionally mirrors the
//! current Python preprocessing contract closely enough for training to start,
//! while staying focused on the discard-only Xmodel1 schema.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use half::f16;
use serde::Serialize;
use serde_json::{json, Value};
use zip::ZipWriter;

use crate::export_common::{
    collect_mjson_files, finalize_temp_npz, output_npz_path, print_export_progress,
    read_npy_first_dim_from_zip, temp_npz_path, write_json_manifest, write_npy_f16, write_npy_f32,
    write_npy_i16, write_npy_i32, write_npy_i8, write_npy_u8,
};
use crate::progress_summary::{summarize_3n1, summarize_like_python, Summary3n1};
use crate::replay_export_core as replay_core;
use crate::replay_export_core::{normalize_tile_repr, strip_aka, tile34_from_pai};
use crate::shanten_table::ensure_init;
use crate::state_core::GameStateCore;
use crate::xmodel1_schema::{
    validate_candidate_mask_and_choice, XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES, XMODEL1_MAX_SPECIAL_CANDIDATES, XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM, XMODEL1_SPECIAL_TYPE_ANKAN,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH, XMODEL1_SPECIAL_TYPE_CHI_LOW, XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_DAIMINKAN, XMODEL1_SPECIAL_TYPE_DAMA, XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_KAKAN, XMODEL1_SPECIAL_TYPE_NONE, XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_RYUKYOKU,
};

const XMODEL1_STATE_TILE_CHANNELS: usize = 57;
const XMODEL1_STATE_SCALAR_DIM: usize = 56;
const XMODEL1_SAMPLE_TYPE_DISCARD: i8 = 0;
const MC_RETURN_GAMMA: f32 = 0.99;
const SCORE_NORM: f32 = 30000.0;
const TILE_KIND_COUNT: usize = 34;

pub fn xmodel1_schema_info() -> (&'static str, u32, usize, usize, usize) {
    (
        XMODEL1_SCHEMA_NAME,
        XMODEL1_SCHEMA_VERSION,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
    )
}

pub fn validate_xmodel1_discard_record(
    chosen_candidate_idx: i16,
    candidate_mask: &[u8],
    candidate_tile_id: &[i16],
) -> Result<(), String> {
    validate_candidate_mask_and_choice(chosen_candidate_idx, candidate_mask, candidate_tile_id)
}

#[derive(Debug, Clone, Serialize)]
struct ExportManifest<'a> {
    schema_name: &'a str,
    schema_version: u32,
    max_candidates: usize,
    candidate_feature_dim: usize,
    candidate_flag_dim: usize,
    file_count: usize,
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    shard_file_counts: BTreeMap<String, usize>,
    shard_sample_counts: BTreeMap<String, usize>,
    used_fallback: bool,
    export_mode: &'a str,
    files: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ExportRunOptions {
    pub smoke: bool,
    pub resume: bool,
    pub progress_every: usize,
    pub jobs: usize,
    pub limit_files: usize,
}

type RoundState = replay_core::RoundState;

#[derive(Debug, Clone)]
struct FullRecord {
    state_tile_feat: Vec<u16>,
    state_scalar: Vec<u16>,
    candidate_feat: Vec<u16>,
    candidate_tile_id: Vec<i16>,
    candidate_mask: Vec<u8>,
    candidate_flags: Vec<u8>,
    chosen_candidate_idx: i16,
    candidate_quality: Vec<f32>,
    candidate_rank: Vec<i8>,
    candidate_hard_bad: Vec<u8>,
    special_candidate_feat: Vec<u16>,
    special_candidate_type_id: Vec<i16>,
    special_candidate_mask: Vec<u8>,
    special_candidate_quality: Vec<f32>,
    special_candidate_rank: Vec<i8>,
    special_candidate_hard_bad: Vec<u8>,
    chosen_special_candidate_idx: i16,
    action_idx_target: i16,
    global_value_target: f32,
    score_delta_target: f32,
    win_target: f32,
    dealin_target: f32,
    pts_given_win_target: f32,
    pts_given_dealin_target: f32,
    // Stage 2: 决策时刻对手 tenpai 标签 (1.0 = 对手当前向听 ≤ 0)。
    // 顺序: [(actor+1)%4, (actor+2)%4, (actor+3)%4] —— 与 Python reference
    // `mahjong_env.replay._compute_opp_tenpai_target` 完全一致。
    opp_tenpai_target: [f32; 3],
    // Stage 2 Rust 迁移: 事件历史窗口。
    // 右对齐,最新事件在末尾;仅含当前 kyoku 内决策时刻之前发生的事件。
    // 见 xmodel1_model_design_v1.md §C.4。
    event_history: [[i16; replay_core::EVENT_HISTORY_FEATURE_DIM]; replay_core::EVENT_HISTORY_LEN],
    offense_quality_target: f32,
    sample_type: i8,
    actor: i8,
    event_index: i32,
    kyoku: i8,
    honba: i8,
    is_open_hand: u8,
}

#[derive(Debug, Clone, Copy)]
struct CandidateMetrics {
    quality: f32,
    rank_bucket: i8,
    hard_bad: u8,
}

#[derive(Debug, Clone)]
struct SpecialCandidateArrays {
    feat: Vec<u16>,
    type_id: Vec<i16>,
    mask: Vec<u8>,
    quality: Vec<f32>,
    rank: Vec<i8>,
    hard_bad: Vec<u8>,
    chosen_idx: i16,
}

pub(crate) fn action_idx_from_action(action: &Value) -> i16 {
    let et = action.get("type").and_then(Value::as_str).unwrap_or("none");
    match et {
        "dahai" => action
            .get("pai")
            .and_then(Value::as_str)
            .and_then(tile34_from_pai)
            .unwrap_or(0),
        "reach" => 34,
        "chi" => {
            let pai = action.get("pai").and_then(Value::as_str).unwrap_or("");
            let consumed = action
                .get("consumed")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut ranks: Vec<i32> = consumed
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|tile| {
                    strip_aka(tile)
                        .chars()
                        .next()
                        .and_then(|ch| ch.to_digit(10))
                        .map(|v| v as i32)
                })
                .collect();
            ranks.sort();
            let pai_rank = strip_aka(pai)
                .chars()
                .next()
                .and_then(|ch| ch.to_digit(10))
                .map(|v| v as i32)
                .unwrap_or(0);
            if ranks.len() >= 2 {
                if pai_rank < ranks[0] {
                    35
                } else if pai_rank < ranks[1] {
                    36
                } else {
                    37
                }
            } else {
                35
            }
        }
        "pon" => 38,
        "daiminkan" => 39,
        "ankan" => 40,
        "kakan" => 41,
        "hora" => 42,
        "ryukyoku" => 43,
        _ => 44,
    }
}

pub(crate) fn encode_state_features_for_actor(
    round_state: &replay_core::RoundState,
    actor: usize,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let before_counts34 = round_state.feature_tracker.players[actor].hand_counts34;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let before_progress = analyze_progress_like_python(&before_counts34, &before_visible34);
    let tile = encode_state_tile_features(round_state, actor, &before_progress, &before_visible34)?;
    let scalar =
        encode_state_scalar_features(round_state, actor, &before_progress, &before_visible34)?;
    Ok((tile, scalar))
}

fn pair_taatsu_metrics_from_counts(counts: &[u8; TILE_KIND_COUNT]) -> (usize, usize) {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count();
    let mut taatsu_count = 0usize;
    for base in [0usize, 9, 18] {
        let suit = &counts[base..base + 9];
        for i in 0..8 {
            if suit[i] > 0 && suit[i + 1] > 0 {
                taatsu_count += 1;
            }
        }
        for i in 0..7 {
            if suit[i] > 0 && suit[i + 2] > 0 {
                taatsu_count += 1;
            }
        }
    }
    (pair_count, taatsu_count)
}

fn pair_taatsu_ankoutsu_metrics_from_counts(
    counts: &[u8; TILE_KIND_COUNT],
) -> (usize, usize, usize) {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count();
    let ankoutsu_count = counts.iter().filter(|&&count| count >= 3).count();
    let (_, taatsu_count) = pair_taatsu_metrics_from_counts(counts);
    (pair_count, taatsu_count, ankoutsu_count)
}

fn analyze_progress_like_python(
    hand_counts34: &[u8; TILE_KIND_COUNT],
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> Summary3n1 {
    summarize_like_python(hand_counts34, visible_counts34)
}

fn visible_counts_for_decision(round_state: &RoundState, actor: usize) -> [u8; TILE_KIND_COUNT] {
    let mut visible = round_state.feature_tracker.players[actor].visible_counts34;
    if let Some(tsumo_tile) = round_state
        .last_tsumo
        .get(actor)
        .and_then(|value| value.as_ref())
    {
        if let Some(idx) = tile34_from_pai(tsumo_tile).map(|value| value as usize) {
            visible[idx] = visible[idx].saturating_add(1);
        }
    }
    visible
}

fn chiitoi_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count() as i32;
    let distinct_count = counts.iter().filter(|&&count| count > 0).count() as i32;
    6 - pair_count + (7 - distinct_count).max(0)
}

fn kokushi_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    const YAOCHUU: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let unique_count = YAOCHUU.iter().filter(|&&tile| counts[tile] > 0).count() as i32;
    let has_pair = YAOCHUU.iter().any(|&tile| counts[tile] >= 2);
    13 - unique_count - i32::from(has_pair)
}

fn is_suited_sequence_start(tile34: usize) -> bool {
    tile34 < 27 && (tile34 % 9) <= 6
}

fn is_complete_regular_counts_for_waits(
    counts: &mut [u8; TILE_KIND_COUNT],
    melds_needed: u8,
    need_pair: bool,
) -> bool {
    let Some(first) = counts.iter().position(|&count| count > 0) else {
        return melds_needed == 0 && !need_pair;
    };
    if need_pair && counts[first] >= 2 {
        counts[first] -= 2;
        if is_complete_regular_counts_for_waits(counts, melds_needed, false) {
            counts[first] += 2;
            return true;
        }
        counts[first] += 2;
    }
    if melds_needed > 0 && counts[first] >= 3 {
        counts[first] -= 3;
        if is_complete_regular_counts_for_waits(counts, melds_needed - 1, need_pair) {
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
        if is_complete_regular_counts_for_waits(counts, melds_needed - 1, need_pair) {
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

fn find_regular_waits_tiles(counts34: &[u8; TILE_KIND_COUNT]) -> [bool; TILE_KIND_COUNT] {
    let mut waits = [false; TILE_KIND_COUNT];
    let tile_count: usize = counts34.iter().map(|&count| count as usize).sum();
    if tile_count % 3 != 1 {
        return waits;
    }
    let melds_needed = (((tile_count + 1) - 2) / 3) as u8;
    for tile34 in 0..TILE_KIND_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        waits[tile34] = is_complete_regular_counts_for_waits(&mut work, melds_needed, true);
    }
    waits
}

fn find_special_waits_tiles(counts34: &[u8; TILE_KIND_COUNT]) -> (i32, [bool; TILE_KIND_COUNT]) {
    let chiitoi = chiitoi_shanten(counts34);
    let kokushi = kokushi_shanten(counts34);
    let special_shanten = chiitoi.min(kokushi);
    let mut waits = [false; TILE_KIND_COUNT];
    if special_shanten != 0 {
        return (special_shanten, waits);
    }
    for tile34 in 0..TILE_KIND_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        if (chiitoi == 0 && chiitoi_shanten(&work) == -1)
            || (kokushi == 0 && kokushi_shanten(&work) == -1)
        {
            waits[tile34] = true;
        }
    }
    (special_shanten, waits)
}

fn calc_shanten_waits_like_python(
    counts34: &[u8; TILE_KIND_COUNT],
    has_melds: bool,
) -> (i8, u8, [bool; TILE_KIND_COUNT]) {
    let tile_count: u8 = counts34.iter().sum();
    let regular_shanten = crate::shanten_table::calc_shanten_all(counts34, tile_count / 3);
    let mut shanten = regular_shanten;
    let mut waits = if regular_shanten == 0 && tile_count % 3 == 1 {
        find_regular_waits_tiles(counts34)
    } else {
        [false; TILE_KIND_COUNT]
    };
    if !has_melds {
        let (special_shanten, special_waits) = find_special_waits_tiles(counts34);
        if special_shanten < shanten as i32 {
            shanten = special_shanten as i8;
            waits = if special_shanten == 0 {
                special_waits
            } else {
                [false; TILE_KIND_COUNT]
            };
        } else if special_shanten == shanten as i32 && shanten == 0 {
            for tile34 in 0..TILE_KIND_COUNT {
                waits[tile34] |= special_waits[tile34];
            }
        }
    }
    let waits_count = waits.iter().filter(|flag| **flag).count() as u8;
    (shanten, waits_count, waits)
}

fn score_rank(scores: &[i32; 4], actor_score: i32) -> usize {
    scores.iter().filter(|&&score| score > actor_score).count()
}

fn dora_next(marker: &str) -> Option<&'static str> {
    match strip_aka(marker).as_str() {
        "1m" => Some("2m"),
        "2m" => Some("3m"),
        "3m" => Some("4m"),
        "4m" => Some("5m"),
        "5m" => Some("6m"),
        "6m" => Some("7m"),
        "7m" => Some("8m"),
        "8m" => Some("9m"),
        "9m" => Some("1m"),
        "1p" => Some("2p"),
        "2p" => Some("3p"),
        "3p" => Some("4p"),
        "4p" => Some("5p"),
        "5p" => Some("6p"),
        "6p" => Some("7p"),
        "7p" => Some("8p"),
        "8p" => Some("9p"),
        "9p" => Some("1p"),
        "1s" => Some("2s"),
        "2s" => Some("3s"),
        "3s" => Some("4s"),
        "4s" => Some("5s"),
        "5s" => Some("6s"),
        "6s" => Some("7s"),
        "7s" => Some("8s"),
        "8s" => Some("9s"),
        "9s" => Some("1s"),
        "E" => Some("S"),
        "S" => Some("W"),
        "W" => Some("N"),
        "N" => Some("E"),
        "P" => Some("F"),
        "F" => Some("C"),
        "C" => Some("P"),
        _ => None,
    }
}

fn yakuhai_tiles(bakaze: &str, actor: i8, oya: i8) -> [bool; TILE_KIND_COUNT] {
    let mut flags = [false; TILE_KIND_COUNT];
    flags[31] = true;
    flags[32] = true;
    flags[33] = true;
    match bakaze {
        "E" => flags[27] = true,
        "S" => flags[28] = true,
        "W" => flags[29] = true,
        "N" => flags[30] = true,
        _ => {}
    }
    let jikaze = 27 + ((actor - oya).rem_euclid(4) as usize);
    if jikaze < TILE_KIND_COUNT {
        flags[jikaze] = true;
    }
    flags
}

fn summarize_after_discard(
    counts34: &[u8; TILE_KIND_COUNT],
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> Summary3n1 {
    summarize_3n1(counts34, visible_counts34)
}

fn sorted_candidate_tiles(hand_tiles: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tile in hand_tiles {
        let normalized = normalize_tile_repr(tile);
        if !out.iter().any(|value| value == &normalized) {
            out.push(normalized);
        }
    }
    out.sort_by(|lhs, rhs| {
        let lhs_tile34 = tile34_from_pai(&strip_aka(lhs)).unwrap_or(99);
        let rhs_tile34 = tile34_from_pai(&strip_aka(rhs)).unwrap_or(99);
        lhs_tile34
            .cmp(&rhs_tile34)
            .then_with(|| strip_aka(lhs).cmp(&strip_aka(rhs)))
            .then_with(|| lhs.cmp(rhs))
    });
    out
}

fn candidate_metrics(
    before_shanten: i8,
    _before_waits_live: usize,
    after_shanten: i8,
    after_ukeire_live: u8,
    after_waits_live: usize,
    drop_open_yakuhai_pair: u8,
    drop_dual_pon_value: u8,
    confirmed_han_floor: f32,
    after_dora_count: f32,
    tanyao_path: f32,
    flush_path: f32,
    discard_dead: f32,
) -> CandidateMetrics {
    let break_tenpai = u8::from(before_shanten == 0 && after_shanten > 0);
    let break_meld_structure = u8::from(before_shanten <= 1 && after_shanten > before_shanten);
    let hard_bad =
        u8::from(break_tenpai == 1 || drop_open_yakuhai_pair == 1 || break_meld_structure == 1);
    let quality = 1.5 * f32::from(after_shanten == 0) - 0.8 * f32::from(after_shanten)
        + 0.04 * (after_ukeire_live as f32)
        + 0.06 * (after_waits_live as f32)
        + 0.25 * confirmed_han_floor
        + 0.15 * after_dora_count
        + 0.1 * tanyao_path
        + 0.08 * flush_path
        - 2.0 * (break_tenpai as f32)
        - 1.0 * (break_meld_structure as f32)
        - 1.2 * (drop_open_yakuhai_pair as f32)
        - 0.7 * (drop_dual_pon_value as f32)
        - 0.15 * discard_dead;
    let quality = if quality.abs() < 1e-6 {
        0.0
    } else if (quality - 1.0).abs() < 1e-6 {
        1.0
    } else {
        quality
    };
    let rank_bucket = if hard_bad == 1 {
        0
    } else if quality >= 1.0 {
        3
    } else if quality >= 0.0 {
        2
    } else {
        1
    };
    CandidateMetrics {
        quality,
        rank_bucket,
        hard_bad,
    }
}

fn dora_target_flags(round_state: &RoundState) -> [bool; TILE_KIND_COUNT] {
    let mut flags = [false; TILE_KIND_COUNT];
    for marker in &round_state.dora_markers {
        if let Some(tile34) = dora_next(marker)
            .and_then(tile34_from_pai)
            .map(|value| value as usize)
        {
            if tile34 < TILE_KIND_COUNT {
                flags[tile34] = true;
            }
        }
    }
    flags
}

fn after_state_path_metrics(
    after_counts34: &[u8; TILE_KIND_COUNT],
    yakuhai: &[bool; TILE_KIND_COUNT],
    dora_flags: &[bool; TILE_KIND_COUNT],
    after_aka_total: usize,
    is_open_hand: bool,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let yakuhai_pair_preserved = f32::from(
        yakuhai
            .iter()
            .enumerate()
            .any(|(idx, flag)| *flag && after_counts34[idx] >= 2),
    );
    let dual_yakuhai_pair_value = f32::from(
        yakuhai
            .iter()
            .enumerate()
            .filter(|(idx, flag)| **flag && after_counts34[*idx] >= 2)
            .count()
            >= 2,
    );
    let tanyao_path = f32::from(
        after_counts34.iter().sum::<u8>() > 0
            && after_counts34.iter().enumerate().all(|(idx, count)| {
                *count == 0
                    || !matches!(
                        idx,
                        0 | 8 | 9 | 17 | 18 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33
                    )
            }),
    );
    let suit_presence = [
        after_counts34[0..9].iter().sum::<u8>() > 0,
        after_counts34[9..18].iter().sum::<u8>() > 0,
        after_counts34[18..27].iter().sum::<u8>() > 0,
    ];
    let flush_path = f32::from(suit_presence.iter().filter(|flag| **flag).count() <= 1);
    let after_dora_count = after_counts34
        .iter()
        .enumerate()
        .filter(|(idx, _)| dora_flags[*idx])
        .map(|(_, count)| *count as usize)
        .sum::<usize>() as f32
        + after_aka_total as f32;
    let yakuhai_triplet_count = yakuhai
        .iter()
        .enumerate()
        .filter(|(idx, flag)| **flag && after_counts34[*idx] >= 3)
        .count() as f32;
    let confirmed_han_floor = (yakuhai_triplet_count + tanyao_path + after_dora_count).min(8.0);
    let hand_value_survives = f32::from(!is_open_hand || confirmed_han_floor > 0.0);
    (
        yakuhai_pair_preserved,
        dual_yakuhai_pair_value,
        tanyao_path,
        flush_path,
        after_dora_count,
        yakuhai_triplet_count,
        confirmed_han_floor,
        hand_value_survives,
    )
}

fn candidate_quality_for_discard(
    round_state: &RoundState,
    actor: usize,
    discard_tile: &str,
) -> Option<CandidateMetrics> {
    let tracker = &round_state.feature_tracker.players[actor];
    let before_counts34 = tracker.hand_counts34;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let before_tile_count: u8 = before_counts34.iter().sum();
    let before_shanten_raw =
        crate::shanten_table::calc_shanten_all(&before_counts34, before_tile_count / 3);
    let (_before_decision_shanten, _before_decision_waits_count, before_decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &before_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let before_waits_live: usize = before_decision_waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(before_visible34[tile34])))
        .sum();
    let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
    let tile34 = tile34_from_pai(discard_tile)?;
    let discard_idx = tile34 as usize;
    let mut after_counts34 = before_counts34;
    if after_counts34[discard_idx] == 0 {
        return None;
    }
    after_counts34[discard_idx] -= 1;
    let after_visible34 = before_visible34;
    let after_progress = summarize_after_discard(&after_counts34, &after_visible34);
    let (after_shanten_raw, _after_waits_count_raw, after_waits_tiles_raw) =
        calc_shanten_waits_like_python(
            &after_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let after_waits_live: usize = after_waits_tiles_raw
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(after_visible34[tile34])))
        .sum();
    let before_same_tile_count = before_counts34[discard_idx];
    let after_same_tile_count = after_counts34[discard_idx];
    let drop_open_yakuhai_pair = u8::from(
        !round_state.players[actor].melds.is_empty()
            && yakuhai[discard_idx]
            && before_same_tile_count >= 2
            && after_same_tile_count < 2,
    );
    let drop_dual_pon_value = u8::from(before_same_tile_count >= 2 && after_same_tile_count < 2);
    Some(candidate_metrics(
        before_shanten_raw,
        before_waits_live,
        after_shanten_raw,
        after_progress.ukeire_live_count,
        after_waits_live,
        drop_open_yakuhai_pair,
        drop_dual_pon_value,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ))
}

fn special_call_family_score(action: &Value) -> f32 {
    let consumed_len = action
        .get("consumed")
        .and_then(Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0) as f32;
    match action.get("type").and_then(Value::as_str).unwrap_or("none") {
        "pon" => 1.25 + 0.1 * consumed_len,
        "chi" => 1.0 + 0.05 * consumed_len,
        _ => 1.15 + 0.08 * consumed_len,
    }
}

fn chi_special_type(action: &Value) -> i16 {
    let pai_rank = action
        .get("pai")
        .and_then(Value::as_str)
        .map(normalize_tile_repr)
        .and_then(|tile| tile.chars().next())
        .and_then(|ch| ch.to_digit(10))
        .unwrap_or(0);
    let mut consumed_ranks: Vec<u32> = action
        .get("consumed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .filter_map(|tile| tile.chars().next().and_then(|ch| ch.to_digit(10)))
        .collect();
    consumed_ranks.sort_unstable();
    if consumed_ranks.len() < 2 || pai_rank < consumed_ranks[0] {
        return XMODEL1_SPECIAL_TYPE_CHI_LOW;
    }
    if pai_rank < consumed_ranks[1] {
        return XMODEL1_SPECIAL_TYPE_CHI_MID;
    }
    XMODEL1_SPECIAL_TYPE_CHI_HIGH
}

fn special_type_from_action(action: &Value, include_terminal_actions: bool) -> Option<i16> {
    match action.get("type").and_then(Value::as_str).unwrap_or("none") {
        "chi" => Some(chi_special_type(action)),
        "pon" => Some(XMODEL1_SPECIAL_TYPE_PON),
        "daiminkan" => Some(XMODEL1_SPECIAL_TYPE_DAIMINKAN),
        "ankan" => Some(XMODEL1_SPECIAL_TYPE_ANKAN),
        "kakan" => Some(XMODEL1_SPECIAL_TYPE_KAKAN),
        "hora" if include_terminal_actions => Some(XMODEL1_SPECIAL_TYPE_HORA),
        "ryukyoku" if include_terminal_actions => Some(XMODEL1_SPECIAL_TYPE_RYUKYOKU),
        "none" => Some(XMODEL1_SPECIAL_TYPE_NONE),
        _ => None,
    }
}

fn special_action_bonus_metrics(
    action: &Value,
    yakuhai: &[bool; TILE_KIND_COUNT],
    dora_flags: &[bool; TILE_KIND_COUNT],
) -> (f32, f32, f32, f32) {
    let mut action_tile_ids: Vec<usize> = Vec::new();
    if let Some(consumed) = action.get("consumed").and_then(Value::as_array) {
        for tile in consumed.iter().filter_map(Value::as_str) {
            if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
                action_tile_ids.push(tile34);
            }
        }
    }
    if let Some(tile) = action.get("pai").and_then(Value::as_str) {
        if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
            action_tile_ids.push(tile34);
        }
    }
    let action_dora_bonus = (action_tile_ids
        .iter()
        .filter(|tile34| dora_flags[**tile34])
        .count() as f32
        + action_tile_ids
            .iter()
            .filter(|tile34| {
                action
                    .get("consumed")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items.iter().filter_map(Value::as_str).any(|tile| {
                            replay_core::tile_is_aka(tile)
                                && tile34_from_pai(tile).map(|v| v as usize) == Some(**tile34)
                        })
                    })
                    .unwrap_or(false)
            })
            .count() as f32)
        / 4.0;
    let action_yakuhai_bonus = action_tile_ids
        .iter()
        .filter(|tile34| yakuhai[**tile34])
        .count() as f32
        / 4.0;
    let action_type = action.get("type").and_then(Value::as_str).unwrap_or("none");
    let call_breaks_closed = f32::from(matches!(action_type, "chi" | "pon" | "daiminkan"));
    let kan_rinshan_bonus = f32::from(matches!(action_type, "daiminkan" | "ankan" | "kakan"));
    (
        action_dora_bonus.min(1.0),
        action_yakuhai_bonus.min(1.0),
        call_breaks_closed,
        kan_rinshan_bonus,
    )
}

#[allow(unused_assignments)]
fn encode_special_candidate_arrays_with_legal_actions(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    chosen_action: &Value,
) -> SpecialCandidateArrays {
    let tracker = &round_state.feature_tracker.players[actor];
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let (before_decision_shanten, before_waits_count, before_waits_tiles) =
        calc_shanten_waits_like_python(
            &tracker.hand_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let before_waits_live: usize = before_waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(before_visible34[tile34])))
        .sum();
    let actor_score = round_state.scores.get(actor).copied().unwrap_or(25000) as f32;
    let mean_score = round_state
        .scores
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / 4.0;
    let score_gap = ((actor_score - mean_score) / SCORE_NORM).clamp(-1.0, 1.0);
    let threat_proxy = f32::from(
        round_state
            .players
            .iter()
            .enumerate()
            .any(|(pid, player)| pid != actor && player.reached),
    );
    let total_discards: usize = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum();
    let round_progress = (total_discards as f32 / 60.0).min(1.0);
    let is_open = f32::from(!round_state.players[actor].melds.is_empty());
    let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
    let dora_flags = dora_target_flags(round_state);
    let (
        _current_yakuhai_pair,
        _current_dual_yakuhai,
        _current_tanyao,
        _current_flush,
        current_dora_count,
        current_yakuhai_triplet_count,
        current_han_floor,
        _current_hand_value_survives,
    ) = after_state_path_metrics(
        &tracker.hand_counts34,
        &yakuhai,
        &dora_flags,
        round_state.feature_tracker.players[actor].aka_counts[0],
        !round_state.players[actor].melds.is_empty(),
    );

    let mut best_discard_quality = 0.0f32;
    for action in legal_actions {
        if action.get("type").and_then(Value::as_str) != Some("dahai") {
            continue;
        }
        if let Some(pai) = action.get("pai").and_then(Value::as_str) {
            if let Some(metrics) = candidate_quality_for_discard(round_state, actor, pai) {
                best_discard_quality = best_discard_quality.max(metrics.quality);
            }
        }
    }

    let mut feat =
        vec![0u16; XMODEL1_MAX_SPECIAL_CANDIDATES * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM];
    let mut type_id = vec![-1i16; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut mask = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut quality = vec![0.0f32; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut rank = vec![0i8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut hard_bad = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut chosen_idx = -1i16;

    let mut grouped: Vec<Option<Value>> = vec![None; XMODEL1_MAX_SPECIAL_CANDIDATES];
    if legal_actions
        .iter()
        .any(|a| a.get("type").and_then(Value::as_str) == Some("reach"))
    {
        grouped[XMODEL1_SPECIAL_TYPE_REACH as usize] = Some(json!({"type":"reach"}));
        grouped[XMODEL1_SPECIAL_TYPE_DAMA as usize] = Some(json!({"type":"dama"}));
    }
    for action in legal_actions {
        let Some(special_type) = special_type_from_action(action, true) else {
            continue;
        };
        if matches!(
            special_type,
            XMODEL1_SPECIAL_TYPE_REACH | XMODEL1_SPECIAL_TYPE_DAMA
        ) {
            continue;
        }
        let replace = grouped[special_type as usize]
            .as_ref()
            .map(|prev| special_call_family_score(action) > special_call_family_score(prev))
            .unwrap_or(true);
        if replace {
            grouped[special_type as usize] = Some(action.clone());
        }
    }
    let has_call_special = [
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_KAKAN,
    ]
    .iter()
    .any(|special_type| grouped[*special_type as usize].is_some());
    if has_call_special {
        grouped[XMODEL1_SPECIAL_TYPE_NONE as usize] = Some(json!({"type":"none"}));
    }

    let mut slot = 0usize;
    for special_type in [
        XMODEL1_SPECIAL_TYPE_REACH,
        XMODEL1_SPECIAL_TYPE_DAMA,
        XMODEL1_SPECIAL_TYPE_HORA,
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_KAKAN,
        XMODEL1_SPECIAL_TYPE_RYUKYOKU,
        XMODEL1_SPECIAL_TYPE_NONE,
    ] {
        if slot >= XMODEL1_MAX_SPECIAL_CANDIDATES {
            break;
        }
        let Some(action) = grouped[special_type as usize].as_ref() else {
            continue;
        };
        type_id[slot] = special_type;
        mask[slot] = 1;
        let (
            speed_gain,
            retain_value,
            value_loss,
            han_floor,
            action_dora_bonus,
            action_yakuhai_bonus,
            context_bonus,
            hand_value_survives,
            candidate_quality,
        ) = if special_type == XMODEL1_SPECIAL_TYPE_REACH {
            let speed_gain = 0.5;
            let retain_value = before_waits_live as f32 / 20.0;
            let value_loss = -0.2;
            let han_floor = 0.0;
            let action_dora_bonus = ((current_han_floor + current_dora_count) / 8.0).min(1.0);
            let action_yakuhai_bonus = (current_yakuhai_triplet_count / 3.0).min(1.0);
            let context_bonus = (best_discard_quality / 3.0).clamp(0.0, 1.0);
            let hand_value_survives = 1.0;
            let candidate_quality = 1.2 * f32::from(before_decision_shanten == 0)
                + 0.15 * before_waits_live as f32
                + 0.5 * score_gap
                - 0.6 * threat_proxy
                + 0.3 * (1.0 - round_progress)
                + 0.15 * action_dora_bonus;
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        } else if special_type == XMODEL1_SPECIAL_TYPE_DAMA {
            let speed_gain = 0.0;
            let retain_value = best_discard_quality;
            let value_loss = 0.0;
            let han_floor = 0.0;
            let action_dora_bonus = (current_dora_count / 10.0).min(1.0);
            let action_yakuhai_bonus = (current_han_floor / 8.0).min(1.0);
            let context_bonus = (best_discard_quality / 3.0).clamp(0.0, 1.0);
            let hand_value_survives = 1.0;
            let candidate_quality = 0.8 * f32::from(before_decision_shanten == 0)
                + 0.35 * retain_value
                + 0.25 * score_gap
                + 0.15 * threat_proxy
                + 0.1 * action_yakuhai_bonus;
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        } else if matches!(
            special_type,
            XMODEL1_SPECIAL_TYPE_CHI_LOW
                | XMODEL1_SPECIAL_TYPE_CHI_MID
                | XMODEL1_SPECIAL_TYPE_CHI_HIGH
                | XMODEL1_SPECIAL_TYPE_PON
                | XMODEL1_SPECIAL_TYPE_DAIMINKAN
                | XMODEL1_SPECIAL_TYPE_ANKAN
                | XMODEL1_SPECIAL_TYPE_KAKAN
        ) {
            let is_chi = matches!(
                special_type,
                XMODEL1_SPECIAL_TYPE_CHI_LOW
                    | XMODEL1_SPECIAL_TYPE_CHI_MID
                    | XMODEL1_SPECIAL_TYPE_CHI_HIGH
            );
            let is_kan = matches!(
                special_type,
                XMODEL1_SPECIAL_TYPE_DAIMINKAN
                    | XMODEL1_SPECIAL_TYPE_ANKAN
                    | XMODEL1_SPECIAL_TYPE_KAKAN
            );
            let speed_gain = if is_chi {
                0.35
            } else if special_type == XMODEL1_SPECIAL_TYPE_PON {
                0.5
            } else {
                0.4
            };
            let retain_value = best_discard_quality;
            let value_loss = if is_chi {
                0.35
            } else if special_type == XMODEL1_SPECIAL_TYPE_PON {
                0.2
            } else {
                0.45
            };
            let (dora_bonus, yakuhai_bonus, call_breaks_closed, kan_rinshan_bonus) =
                special_action_bonus_metrics(action, &yakuhai, &dora_flags);
            let han_floor = (dora_bonus + yakuhai_bonus).min(1.0);
            let hand_value_survives =
                f32::from(is_open > 0.0 || current_han_floor > 0.0 || han_floor > 0.0);
            let context_bonus = if is_kan {
                kan_rinshan_bonus
            } else {
                call_breaks_closed
            };
            let action_dora_bonus = dora_bonus;
            let action_yakuhai_bonus = yakuhai_bonus;
            let candidate_quality = 0.7 * speed_gain
                + 0.25 * best_discard_quality
                + 0.4 * han_floor
                + 0.2 * action_dora_bonus
                + 0.25 * action_yakuhai_bonus
                + 0.15 * f32::from(is_kan)
                - 0.6 * value_loss
                - 0.4 * threat_proxy
                - 0.2 * context_bonus * (1.0 - is_open);
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        } else if special_type == XMODEL1_SPECIAL_TYPE_HORA {
            let speed_gain = 1.0;
            let retain_value = ((current_han_floor + current_dora_count) / 8.0).min(1.0);
            let value_loss = 0.0;
            let han_floor = 0.0;
            let action_dora_bonus = (current_dora_count / 10.0).min(1.0);
            let action_yakuhai_bonus = (current_han_floor / 8.0).min(1.0);
            let context_bonus = 1.0;
            let hand_value_survives = 1.0;
            let candidate_quality =
                3.0 + retain_value + 0.25 * action_dora_bonus + 0.15 * action_yakuhai_bonus;
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        } else if special_type == XMODEL1_SPECIAL_TYPE_RYUKYOKU {
            let speed_gain = 0.1;
            let retain_value = (before_decision_shanten as f32 / 8.0).min(1.0);
            let value_loss = 0.0;
            let han_floor = 0.0;
            let action_dora_bonus = 0.0;
            let action_yakuhai_bonus = 0.0;
            let context_bonus = (1.0 - round_progress).clamp(0.0, 1.0);
            let hand_value_survives = 1.0;
            let candidate_quality = 0.15 + 0.25 * context_bonus + 0.2 * threat_proxy;
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        } else {
            let speed_gain = 0.0;
            let retain_value = best_discard_quality;
            let value_loss = 0.0;
            let han_floor = 0.0;
            let action_dora_bonus = threat_proxy;
            let action_yakuhai_bonus = (current_han_floor / 8.0).min(1.0);
            let context_bonus = threat_proxy;
            let hand_value_survives = 1.0;
            let candidate_quality = 0.55 * retain_value
                + 0.4 * (1.0 - is_open)
                + 0.2 * threat_proxy
                + 0.1 * action_yakuhai_bonus;
            (
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            )
        };

        let feature_values = [
            before_decision_shanten as f32 / 8.0,
            f32::from(before_decision_shanten == 0),
            before_waits_count as f32 / 34.0,
            (before_waits_live as f32 / 20.0).min(1.0),
            round_progress,
            score_gap.clamp(-1.0, 1.0),
            threat_proxy,
            is_open,
            speed_gain.clamp(0.0, 1.0),
            (retain_value / 3.0).clamp(-1.0, 1.0),
            value_loss.clamp(0.0, 1.0),
            han_floor.clamp(0.0, 1.0),
            action_dora_bonus.clamp(0.0, 1.0),
            action_yakuhai_bonus.clamp(0.0, 1.0),
            context_bonus.clamp(0.0, 1.0),
            hand_value_survives,
        ];
        let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
        for (idx, value) in feature_values.iter().enumerate() {
            feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        quality[slot] = candidate_quality;
        let candidate_hard_bad = u8::from(
            (special_type == XMODEL1_SPECIAL_TYPE_REACH
                && threat_proxy > 0.5
                && before_waits_live < 4)
                || (matches!(
                    special_type,
                    XMODEL1_SPECIAL_TYPE_CHI_LOW
                        | XMODEL1_SPECIAL_TYPE_CHI_MID
                        | XMODEL1_SPECIAL_TYPE_CHI_HIGH
                        | XMODEL1_SPECIAL_TYPE_DAIMINKAN
                        | XMODEL1_SPECIAL_TYPE_ANKAN
                        | XMODEL1_SPECIAL_TYPE_KAKAN
                ) && value_loss >= 0.4
                    && han_floor <= 0.0),
        );
        hard_bad[slot] = candidate_hard_bad;
        rank[slot] = if candidate_hard_bad == 1 {
            0
        } else if candidate_quality >= 1.0 {
            3
        } else if candidate_quality >= 0.25 {
            2
        } else {
            1
        };

        let chosen_type = chosen_action
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("none");
        if (chosen_type == "reach" && special_type == XMODEL1_SPECIAL_TYPE_REACH)
            || (chosen_type == "hora" && special_type == XMODEL1_SPECIAL_TYPE_HORA)
            || (chosen_type == "ryukyoku" && special_type == XMODEL1_SPECIAL_TYPE_RYUKYOKU)
            || (chosen_type == "none" && special_type == XMODEL1_SPECIAL_TYPE_NONE)
            || (chosen_type == "chi"
                && special_type == special_type_from_action(chosen_action, true).unwrap_or(-1))
            || (chosen_type == "pon" && special_type == XMODEL1_SPECIAL_TYPE_PON)
            || (chosen_type == "daiminkan" && special_type == XMODEL1_SPECIAL_TYPE_DAIMINKAN)
            || (chosen_type == "ankan" && special_type == XMODEL1_SPECIAL_TYPE_ANKAN)
            || (chosen_type == "kakan" && special_type == XMODEL1_SPECIAL_TYPE_KAKAN)
            || (chosen_type == "dahai"
                && special_type == XMODEL1_SPECIAL_TYPE_DAMA
                && grouped[XMODEL1_SPECIAL_TYPE_REACH as usize].is_some())
        {
            chosen_idx = slot as i16;
        }
        slot += 1;
    }

    SpecialCandidateArrays {
        feat,
        type_id,
        mask,
        quality,
        rank,
        hard_bad,
        chosen_idx,
    }
}

fn encode_special_candidate_arrays(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    before_decision_shanten: i8,
    before_waits_count: u8,
    before_waits_live: usize,
    best_discard_quality: f32,
) -> SpecialCandidateArrays {
    let mut feat =
        vec![0u16; XMODEL1_MAX_SPECIAL_CANDIDATES * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM];
    let mut type_id = vec![-1i16; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut mask = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut quality = vec![0.0f32; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut rank = vec![0i8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut hard_bad = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut chosen_idx = -1i16;

    let actor_score = round_state.scores.get(actor).copied().unwrap_or(25000) as f32;
    let mean_score = round_state
        .scores
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / 4.0;
    let score_gap = ((actor_score - mean_score) / SCORE_NORM).clamp(-1.0, 1.0);
    let threat_proxy = f32::from(
        round_state
            .players
            .iter()
            .enumerate()
            .any(|(pid, player)| pid != actor && player.reached),
    );
    let total_discards: usize = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum();
    let round_progress = (total_discards as f32 / 60.0).min(1.0);
    let has_any_meld = !round_state.players[actor].melds.is_empty();
    let has_open_meld = round_state.players[actor]
        .melds
        .iter()
        .any(|meld| !matches!(meld.kind.as_str(), "ankan"));
    let is_open = f32::from(has_any_meld);
    let tracker = &round_state.feature_tracker.players[actor];
    let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
    let dora_flags = dora_target_flags(round_state);
    let (
        _current_yakuhai_pair,
        _current_dual_yakuhai,
        _current_tanyao,
        _current_flush,
        current_dora_count,
        current_yakuhai_triplet_count,
        current_han_floor,
        _current_hand_value_survives,
    ) = after_state_path_metrics(
        &tracker.hand_counts34,
        &yakuhai,
        &dora_flags,
        round_state.feature_tracker.players[actor].aka_counts[0],
        has_any_meld,
    );
    let reach_available = before_decision_shanten <= 0
        && !has_open_meld
        && !round_state.players[actor].reached
        && !round_state.players[actor].pending_reach
        && round_state.scores[actor] >= 1000;

    if reach_available {
        let candidates = [
            (
                XMODEL1_SPECIAL_TYPE_REACH,
                0.5f32,
                before_waits_live as f32 / 20.0,
                -0.2f32,
                0.0f32,
                ((current_han_floor + current_dora_count) / 8.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                (best_discard_quality / 3.0).clamp(0.0, 1.0),
                1.0f32,
                1.2 * f32::from(before_decision_shanten == 0)
                    + 0.15 * before_waits_live as f32
                    + 0.5 * score_gap
                    - 0.6 * threat_proxy
                    + 0.3 * (1.0 - round_progress)
                    + 0.15 * ((current_han_floor + current_dora_count) / 8.0).min(1.0),
            ),
            (
                XMODEL1_SPECIAL_TYPE_DAMA,
                0.0f32,
                best_discard_quality,
                0.0f32,
                0.0f32,
                (current_dora_count / 10.0).min(1.0),
                (current_han_floor / 8.0).min(1.0),
                (best_discard_quality / 3.0).clamp(0.0, 1.0),
                1.0f32,
                0.8 * f32::from(before_decision_shanten == 0)
                    + 0.35 * best_discard_quality
                    + 0.25 * score_gap
                    + 0.15 * threat_proxy
                    + 0.1 * (current_han_floor / 8.0).min(1.0),
            ),
        ];
        for (
            slot,
            (
                special_type,
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            ),
        ) in candidates.iter().enumerate()
        {
            type_id[slot] = *special_type;
            mask[slot] = 1;
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap,
                threat_proxy,
                is_open,
                (*speed_gain).clamp(0.0, 1.0),
                (*retain_value / 3.0).clamp(-1.0, 1.0),
                (*value_loss).clamp(0.0, 1.0),
                (*han_floor).clamp(0.0, 1.0),
                (*action_dora_bonus).clamp(0.0, 1.0),
                (*action_yakuhai_bonus).clamp(0.0, 1.0),
                (*context_bonus).clamp(0.0, 1.0),
                (*hand_value_survives).clamp(0.0, 1.0),
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::from(best_discard_quality > 0.0),
            ];
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] = *candidate_quality;
            let candidate_hard_bad = u8::from(
                *special_type == XMODEL1_SPECIAL_TYPE_REACH
                    && threat_proxy > 0.5
                    && before_waits_live < 4,
            );
            hard_bad[slot] = candidate_hard_bad;
            rank[slot] = if candidate_hard_bad == 1 {
                0
            } else if *candidate_quality >= 1.0 {
                3
            } else if *candidate_quality >= 0.25 {
                2
            } else {
                1
            };
        }
        chosen_idx = 1;
    }

    if legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("hora"))
    {
        let slot = mask
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(XMODEL1_MAX_SPECIAL_CANDIDATES);
        if slot < XMODEL1_MAX_SPECIAL_CANDIDATES {
            type_id[slot] = XMODEL1_SPECIAL_TYPE_HORA;
            mask[slot] = 1;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap.clamp(-1.0, 1.0),
                threat_proxy,
                is_open,
                1.0,
                ((current_han_floor + current_dora_count) / 8.0).min(1.0) / 3.0,
                0.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_han_floor / 8.0).min(1.0),
                1.0,
                1.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                f32::from(best_discard_quality > 0.0),
            ];
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] = 3.0 + ((current_han_floor + current_dora_count) / 8.0).min(1.0);
            rank[slot] = 3;
        }
    }

    if legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("ryukyoku"))
    {
        let slot = mask
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(XMODEL1_MAX_SPECIAL_CANDIDATES);
        if slot < XMODEL1_MAX_SPECIAL_CANDIDATES {
            type_id[slot] = XMODEL1_SPECIAL_TYPE_RYUKYOKU;
            mask[slot] = 1;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap.clamp(-1.0, 1.0),
                threat_proxy,
                is_open,
                0.1,
                (before_decision_shanten as f32 / 8.0).min(1.0) / 3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (1.0 - round_progress).clamp(0.0, 1.0),
                1.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                f32::from(best_discard_quality > 0.0),
            ];
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] =
                0.15 + 0.25 * (1.0 - round_progress).clamp(0.0, 1.0) + 0.2 * threat_proxy;
            rank[slot] = 1;
        }
    }

    SpecialCandidateArrays {
        feat,
        type_id,
        mask,
        quality,
        rank,
        hard_bad,
        chosen_idx,
    }
}

fn encode_special_sample_record(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    chosen_action: &Value,
    sample_type: i8,
    event_index: i32,
    event_history: [[i16; replay_core::EVENT_HISTORY_FEATURE_DIM]; replay_core::EVENT_HISTORY_LEN],
) -> Result<FullRecord, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let before_counts34 = tracker.hand_counts34;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let before_progress = analyze_progress_like_python(&before_counts34, &before_visible34);
    let state_tile_feat =
        encode_state_tile_features(round_state, actor, &before_progress, &before_visible34)?;
    let state_scalar =
        encode_state_scalar_features(round_state, actor, &before_progress, &before_visible34)?;
    let special = encode_special_candidate_arrays_with_legal_actions(
        round_state,
        actor,
        legal_actions,
        chosen_action,
    );
    let action_idx_target = action_idx_from_action(chosen_action);
    let offense_quality_target = if special.chosen_idx >= 0 {
        special.quality[special.chosen_idx as usize]
    } else {
        0.0
    };
    let opp_tenpai_target =
        replay_core::compute_opp_tenpai_target(round_state, actor, &before_visible34);
    Ok(FullRecord {
        state_tile_feat,
        state_scalar,
        candidate_feat: vec![0u16; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM],
        candidate_tile_id: vec![-1i16; XMODEL1_MAX_CANDIDATES],
        candidate_mask: vec![0u8; XMODEL1_MAX_CANDIDATES],
        candidate_flags: vec![0u8; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM],
        chosen_candidate_idx: -1,
        candidate_quality: vec![0.0f32; XMODEL1_MAX_CANDIDATES],
        candidate_rank: vec![0i8; XMODEL1_MAX_CANDIDATES],
        candidate_hard_bad: vec![0u8; XMODEL1_MAX_CANDIDATES],
        special_candidate_feat: special.feat,
        special_candidate_type_id: special.type_id,
        special_candidate_mask: special.mask,
        special_candidate_quality: special.quality,
        special_candidate_rank: special.rank,
        special_candidate_hard_bad: special.hard_bad,
        chosen_special_candidate_idx: special.chosen_idx,
        action_idx_target,
        global_value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        opp_tenpai_target,
        event_history,
        offense_quality_target,
        sample_type,
        actor: actor as i8,
        event_index,
        kyoku: round_state.kyoku,
        honba: round_state.honba,
        is_open_hand: u8::from(!round_state.players[actor].melds.is_empty()),
    })
}

fn encode_candidate_features(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    chosen_tile: &str,
    event_index: i32,
    event_history: [[i16; replay_core::EVENT_HISTORY_FEATURE_DIM]; replay_core::EVENT_HISTORY_LEN],
) -> Result<FullRecord, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let before_counts34 = tracker.hand_counts34;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let before_progress = analyze_progress_like_python(&before_counts34, &before_visible34);
    let before_tile_count: u8 = before_counts34.iter().sum();
    let before_shanten_raw =
        crate::shanten_table::calc_shanten_all(&before_counts34, before_tile_count / 3);
    let (before_decision_shanten, before_decision_waits_count, before_decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &before_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let before_waits_live: usize = before_decision_waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(before_visible34[tile34])))
        .sum();
    let state_tile_feat =
        encode_state_tile_features(round_state, actor, &before_progress, &before_visible34)?;
    let state_scalar =
        encode_state_scalar_features(round_state, actor, &before_progress, &before_visible34)?;

    let candidate_tiles = sorted_candidate_tiles(&tracker.hand_tiles);
    let mut candidate_feat = vec![0u16; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_flags = vec![0u8; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM];
    let mut candidate_quality = vec![0.0f32; XMODEL1_MAX_CANDIDATES];
    let mut candidate_rank = vec![0i8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_hard_bad = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut chosen_candidate_idx: Option<i16> = None;

    let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
    let dora_flags = dora_target_flags(round_state);
    let any_other_reached = round_state
        .players
        .iter()
        .enumerate()
        .any(|(pid, player)| pid != actor && player.reached);

    for (slot, discard_tile) in candidate_tiles
        .iter()
        .take(XMODEL1_MAX_CANDIDATES)
        .enumerate()
    {
        let Some(tile34) = tile34_from_pai(discard_tile) else {
            continue;
        };
        let mut after_counts34 = before_counts34;
        let discard_idx = tile34 as usize;
        if after_counts34[discard_idx] == 0 {
            continue;
        }
        after_counts34[discard_idx] -= 1;
        let after_visible34 = before_visible34;
        let after_progress = summarize_after_discard(&after_counts34, &after_visible34);
        let (after_shanten_raw, after_waits_count_raw, after_waits_tiles_raw) =
            calc_shanten_waits_like_python(
                &after_counts34,
                !round_state.players[actor].melds.is_empty(),
            );
        let after_waits_live: usize = after_waits_tiles_raw
            .iter()
            .enumerate()
            .filter(|(_, flag)| **flag)
            .map(|(tile34, _)| usize::from(4u8.saturating_sub(after_visible34[tile34])))
            .sum();
        let (pair_count, taatsu_count) = pair_taatsu_metrics_from_counts(&after_counts34);
        let before_same_tile_count = before_counts34[discard_idx];
        let after_same_tile_count = after_counts34[discard_idx];
        let drop_open_yakuhai_pair = u8::from(
            !round_state.players[actor].melds.is_empty()
                && yakuhai[discard_idx]
                && before_same_tile_count >= 2
                && after_same_tile_count < 2,
        );
        let drop_dual_pon_value =
            u8::from(before_same_tile_count >= 2 && after_same_tile_count < 2);
        let break_tenpai = u8::from(before_shanten_raw == 0 && after_shanten_raw > 0);
        let break_best_wait =
            u8::from(before_shanten_raw == 0 && after_waits_live < before_waits_live);
        let break_meld_structure =
            u8::from(before_shanten_raw <= 1 && after_shanten_raw > before_shanten_raw);
        let is_honor = u8::from(tile34 >= 27);
        let is_terminal = u8::from(matches!(tile34, 0 | 8 | 9 | 17 | 18 | 26));
        let is_aka = u8::from(replay_core::tile_is_aka(discard_tile));
        let after_aka_total = round_state.feature_tracker.players[actor].aka_counts[0]
            .saturating_sub(is_aka as usize);
        let (
            yakuhai_pair_preserved,
            dual_yakuhai_pair_value,
            tanyao_path,
            flush_path,
            after_dora_count,
            yakuhai_triplet_count,
            confirmed_han_floor,
            hand_value_survives,
        ) = after_state_path_metrics(
            &after_counts34,
            &yakuhai,
            &dora_flags,
            after_aka_total,
            !round_state.players[actor].melds.is_empty(),
        );
        let wait_density = if after_waits_count_raw > 0 {
            (after_waits_live as f32 / (4.0 * after_waits_count_raw as f32).max(1.0)).min(1.0)
        } else {
            0.0
        };
        let structure_density = ((pair_count + taatsu_count) as f32 / 8.0).min(1.0);
        let is_dora = u8::from(dora_flags[discard_idx] || is_aka == 1);
        let is_yakuhai = u8::from(yakuhai[discard_idx]);
        let discard_dead = f32::from(after_visible34[discard_idx] >= 3);
        let metrics = candidate_metrics(
            before_shanten_raw,
            before_waits_live,
            after_shanten_raw,
            after_progress.ukeire_live_count,
            after_waits_live,
            drop_open_yakuhai_pair,
            drop_dual_pon_value,
            confirmed_han_floor,
            after_dora_count,
            tanyao_path,
            flush_path,
            discard_dead,
        );

        let feat = [
            f32::from(after_shanten_raw) / 8.0,
            f32::from(after_shanten_raw == 0),
            after_waits_count_raw as f32 / 34.0,
            (after_waits_live as f32 / 20.0).min(1.0),
            (after_progress.ukeire_type_count as f32 / 34.0).min(1.0),
            (after_progress.ukeire_live_count as f32 / 34.0).min(1.0),
            0.0,
            wait_density,
            pair_count as f32 / 7.0,
            taatsu_count as f32 / 6.0,
            structure_density,
            (confirmed_han_floor / 8.0).min(1.0),
            hand_value_survives,
            yakuhai_pair_preserved,
            dual_yakuhai_pair_value,
            tanyao_path,
            flush_path,
            f32::from(any_other_reached),
            (after_dora_count / 10.0).min(1.0),
            (yakuhai_triplet_count / 3.0).min(1.0),
            discard_dead,
            (((after_shanten_raw - before_shanten_raw) as f32) / 4.0).clamp(-1.0, 1.0),
            (after_waits_live as f32 / 136.0).min(1.0),
            (after_progress.ukeire_live_count as f32 / 136.0).min(1.0),
            0.0,
            0.0,
            (after_dora_count / 4.0).min(1.0),
            (yakuhai_triplet_count / 2.0).min(1.0),
            f32::from(break_tenpai),
            f32::from(break_best_wait),
            f32::from(break_meld_structure),
            hand_value_survives,
            f32::from(is_dora),
            f32::from(is_yakuhai),
            f32::from(before_waits_live > after_waits_live),
        ];
        let flags = [
            break_tenpai,
            break_best_wait,
            break_meld_structure,
            drop_open_yakuhai_pair,
            drop_dual_pon_value,
            is_honor,
            is_terminal,
            is_aka,
            is_dora,
            is_yakuhai,
        ];

        let feat_offset = slot * XMODEL1_CANDIDATE_FEATURE_DIM;
        for (idx, value) in feat.iter().enumerate() {
            candidate_feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        let flags_offset = slot * XMODEL1_CANDIDATE_FLAG_DIM;
        candidate_flags[flags_offset..flags_offset + XMODEL1_CANDIDATE_FLAG_DIM]
            .copy_from_slice(&flags);
        candidate_tile_id[slot] = tile34;
        candidate_mask[slot] = 1;
        candidate_quality[slot] = metrics.quality;
        candidate_rank[slot] = metrics.rank_bucket;
        candidate_hard_bad[slot] = metrics.hard_bad;
        if normalize_tile_repr(chosen_tile) == *discard_tile {
            chosen_candidate_idx = Some(slot as i16);
        }
    }

    let chosen_candidate_idx = chosen_candidate_idx.ok_or_else(|| {
        format!(
            "failed to map chosen discard into candidate set: actor={} event_index={} chosen_tile={}",
            actor, event_index, chosen_tile
        )
    })?;
    validate_xmodel1_discard_record(chosen_candidate_idx, &candidate_mask, &candidate_tile_id)?;
    let chosen_quality = candidate_quality[chosen_candidate_idx as usize];
    let best_discard_quality = candidate_quality
        .iter()
        .enumerate()
        .filter(|(idx, _)| candidate_mask[*idx] > 0)
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let best_discard_quality = if best_discard_quality.is_finite() {
        best_discard_quality
    } else {
        0.0
    };
    let special = encode_special_candidate_arrays(
        round_state,
        actor,
        &replay_core::enumerate_actor_legal_actions(core_state, actor)?,
        before_decision_shanten,
        before_decision_waits_count,
        before_waits_live,
        best_discard_quality,
    );

    Ok(FullRecord {
        state_tile_feat,
        state_scalar,
        candidate_feat,
        candidate_tile_id,
        candidate_mask,
        candidate_flags,
        chosen_candidate_idx,
        candidate_quality,
        candidate_rank,
        candidate_hard_bad,
        special_candidate_feat: special.feat,
        special_candidate_type_id: special.type_id,
        special_candidate_mask: special.mask,
        special_candidate_quality: special.quality,
        special_candidate_rank: special.rank,
        special_candidate_hard_bad: special.hard_bad,
        chosen_special_candidate_idx: special.chosen_idx,
        action_idx_target: tile34_from_pai(chosen_tile).unwrap_or(0),
        global_value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        opp_tenpai_target: replay_core::compute_opp_tenpai_target(
            round_state,
            actor,
            &before_visible34,
        ),
        event_history,
        offense_quality_target: chosen_quality,
        sample_type: XMODEL1_SAMPLE_TYPE_DISCARD,
        actor: actor as i8,
        event_index,
        kyoku: round_state.kyoku,
        honba: round_state.honba,
        is_open_hand: u8::from(!round_state.players[actor].melds.is_empty()),
    })
}

fn encode_state_tile_features(
    round_state: &RoundState,
    actor: usize,
    progress: &Summary3n1,
    visible_counts: &[u8; TILE_KIND_COUNT],
) -> Result<Vec<u16>, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let (_decision_shanten, _decision_waits_count, decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &tracker.hand_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let mut tile_feat = vec![0f32; XMODEL1_STATE_TILE_CHANNELS * TILE_KIND_COUNT];
    let discards = &round_state.players;
    let other_pids: Vec<usize> = (0..4).filter(|&pid| pid != actor).collect();
    let mut latest_riichi_pid: Option<usize> = None;
    let mut latest_riichi_turn: i32 = -1;
    let mut opponent_meld_counts = [0.0f32; 3];

    for tile34 in 0..TILE_KIND_COUNT {
        let count = tracker.hand_counts34[tile34].min(4);
        for plane in 0..count as usize {
            tile_feat[(plane * TILE_KIND_COUNT) + tile34] = 1.0;
        }
    }

    for (meld_slot, meld) in round_state.players[actor].melds.iter().take(4).enumerate() {
        let channel = 4 + meld_slot;
        for tile in meld.consumed.iter().chain(meld.pai.iter()) {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                tile_feat[channel * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (slot, pid) in other_pids.iter().enumerate() {
        let channel_base = 8 + slot * 2;
        let pid_discards = &discards[*pid].discards;
        for (turn, discard) in pid_discards.iter().enumerate() {
            let idx = discard.tile34 as usize;
            if discard.reach_declared {
                tile_feat[channel_base * TILE_KIND_COUNT + idx] = 1.0;
                if (turn as i32) > latest_riichi_turn {
                    latest_riichi_turn = turn as i32;
                    latest_riichi_pid = Some(*pid);
                }
            }
            if !discard.tsumogiri {
                tile_feat[(channel_base + 1) * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (slot, pid) in other_pids.iter().enumerate() {
        let pid_melds = &round_state.players[*pid].melds;
        opponent_meld_counts[slot] = pid_melds.len() as f32 / 4.0;
        for meld in pid_melds {
            for tile in meld.consumed.iter().chain(meld.pai.iter()) {
                if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                    tile_feat[(14 + slot) * TILE_KIND_COUNT + idx] = 1.0;
                }
            }
        }
    }

    for discard in &round_state.players[actor].discards {
        tile_feat[17 * TILE_KIND_COUNT + discard.tile34 as usize] = 1.0;
    }

    for marker in &round_state.dora_markers {
        if let Some(actual) = dora_next(marker)
            .and_then(tile34_from_pai)
            .map(|value| value as usize)
        {
            tile_feat[18 * TILE_KIND_COUNT + actual] += 1.0;
        }
    }

    for (idx, wait) in decision_waits_tiles.iter().enumerate() {
        if *wait {
            tile_feat[19 * TILE_KIND_COUNT + idx] = 1.0;
        }
    }

    let mut slot_for_pid = [0usize; 4];
    for (slot, pid) in other_pids.iter().enumerate() {
        slot_for_pid[*pid] = slot;
    }
    slot_for_pid[actor] = 3;
    for pid in 0..4 {
        let slot = slot_for_pid[pid];
        for (turn, discard) in round_state.players[pid].discards.iter().enumerate() {
            let seg = (turn / 3).min(7);
            let channel = 20 + slot * 8 + seg;
            tile_feat[channel * TILE_KIND_COUNT + discard.tile34 as usize] = 1.0;
        }
    }

    if let Some(latest_riichi_pid) = latest_riichi_pid {
        let mut safe_tiles = [false; TILE_KIND_COUNT];
        for pid in 0..4 {
            if pid == latest_riichi_pid {
                continue;
            }
            for discard in &round_state.players[pid].discards {
                safe_tiles[discard.tile34 as usize] = true;
            }
        }
        for (idx, flag) in safe_tiles.iter().enumerate() {
            if *flag {
                tile_feat[53 * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (idx, flag) in progress.ukeire_tiles.iter().enumerate() {
        if *flag {
            tile_feat[54 * TILE_KIND_COUNT + idx] = 1.0;
        }
    }

    for idx in 0..TILE_KIND_COUNT {
        tile_feat[56 * TILE_KIND_COUNT + idx] = (visible_counts[idx] as f32 / 4.0).clamp(0.0, 1.0);
    }

    Ok(tile_feat
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect())
}

fn encode_state_scalar_features(
    round_state: &RoundState,
    actor: usize,
    progress: &Summary3n1,
    visible_counts: &[u8; TILE_KIND_COUNT],
) -> Result<Vec<u16>, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let (_decision_shanten, _decision_waits_count, decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &tracker.hand_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let actor_score = round_state.scores[actor];
    let other_pids: Vec<usize> = (0..4).filter(|&pid| pid != actor).collect();
    let reached: Vec<bool> = round_state
        .players
        .iter()
        .map(|player| player.reached)
        .collect();
    let mut scalar = vec![0f32; XMODEL1_STATE_SCALAR_DIM];

    let mut all34_list: Vec<usize> = Vec::new();
    for (tile34, &count) in tracker.hand_counts34.iter().enumerate() {
        all34_list.extend(std::iter::repeat(tile34).take(count as usize));
    }
    for (tile34, &count) in tracker.meld_counts34.iter().enumerate() {
        all34_list.extend(std::iter::repeat(tile34).take(count as usize));
    }
    let total_tiles = all34_list.len();
    let all34_set: std::collections::BTreeSet<usize> = all34_list.iter().copied().collect();
    let yaochuu_tiles = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let yaochuu_set: std::collections::BTreeSet<usize> = yaochuu_tiles.into_iter().collect();
    let man_cnt = tracker.suit_counts[0];
    let pin_cnt = tracker.suit_counts[1];
    let sou_cnt = tracker.suit_counts[2];
    let honor_cnt = tracker.suit_counts[3];
    let yaochuu_cnt = all34_list
        .iter()
        .filter(|tile| yaochuu_set.contains(tile))
        .count();
    let has_tanyao = !all34_set.is_empty() && all34_set.is_disjoint(&yaochuu_set);
    let suupai_yaochuu_cnt = all34_list
        .iter()
        .filter(|tile| matches!(**tile, 0 | 8 | 9 | 17 | 18 | 26))
        .count();
    let is_open = !round_state.players[actor].melds.is_empty();
    let has_junchan =
        !is_open && honor_cnt == 0 && total_tiles > 0 && suupai_yaochuu_cnt * 3 >= total_tiles;
    let has_chanta = total_tiles > 0 && yaochuu_cnt * 3 >= total_tiles;
    let suit_counts_non_zero: Vec<usize> = [man_cnt, pin_cnt, sou_cnt]
        .into_iter()
        .filter(|count| *count > 0)
        .collect();
    let has_honitsu = suit_counts_non_zero.len() == 1 && total_tiles > 0;
    let has_chinitsu = suit_counts_non_zero.len() == 1 && honor_cnt == 0 && total_tiles > 0;
    let (_, taatsu_count, _ank_unused) =
        pair_taatsu_ankoutsu_metrics_from_counts(&tracker.hand_counts34);
    let pair_count = tracker.pair_count;
    let ankoutsu_cnt = tracker.ankoutsu_count;
    let chiitoi = if is_open {
        6
    } else {
        chiitoi_shanten(&tracker.hand_counts34)
    };
    let kokushi = if is_open {
        13
    } else {
        kokushi_shanten(&tracker.hand_counts34)
    };
    let mut estimated_wall_draws = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum::<usize>();
    let mut chi_pon_calls = 0usize;
    let mut kan_calls = 0usize;
    let mut opponent_meld_counts = [0.0f32; 3];
    for (slot, pid) in other_pids.iter().enumerate() {
        let melds = &round_state.players[*pid].melds;
        opponent_meld_counts[slot] = melds.len() as f32 / 4.0;
        for meld in melds {
            if matches!(meld.kind.as_str(), "chi" | "pon") {
                chi_pon_calls += 1;
            }
            if matches!(meld.kind.as_str(), "daiminkan" | "ankan" | "kakan") {
                kan_calls += 1;
            }
        }
    }
    estimated_wall_draws = estimated_wall_draws.saturating_sub(chi_pon_calls) + kan_calls;
    let tiles_left = (70i32 - estimated_wall_draws as i32).max(0);
    let top_gap = (*round_state.scores.iter().max().unwrap_or(&actor_score) - actor_score).max(0);
    let bottom_gap =
        (actor_score - *round_state.scores.iter().min().unwrap_or(&actor_score)).max(0);
    let riichi_threat_count = reached
        .iter()
        .enumerate()
        .filter(|(pid, flag)| *pid != actor && **flag)
        .count();
    let is_all_last_like = f32::from(
        matches!(round_state.bakaze.as_str(), "W" | "N")
            || (round_state.bakaze == "S" && round_state.kyoku >= 4),
    );
    let tenpai_live_tile_count: usize = decision_waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(visible_counts[tile34])))
        .sum();
    let wait_hiddenness_ratio = if progress.waits_count > 0 {
        tenpai_live_tile_count as f32 / (4.0 * progress.waits_count as f32).max(1.0)
    } else {
        0.0
    };

    let dora_count: usize = round_state
        .dora_markers
        .iter()
        .filter_map(|marker| dora_next(marker))
        .filter_map(tile34_from_pai)
        .map(|tile34| {
            let tile34 = tile34 as usize;
            all34_list.iter().filter(|&&value| value == tile34).count()
        })
        .sum();
    let jikaze = ((actor as i8 - round_state.oya).rem_euclid(4)) as usize;

    scalar[0] = f32::from(round_state.bakaze == "E");
    scalar[1] = f32::from(round_state.bakaze == "S");
    scalar[2] = f32::from(round_state.bakaze == "W");
    scalar[3] = round_state.kyoku as f32 / 4.0;
    scalar[4] = round_state.honba as f32 / 8.0;
    scalar[5] = round_state.kyotaku as f32 / 8.0;
    scalar[6] = actor_score as f32 / 50000.0;
    scalar[7] = score_rank(&round_state.scores, actor_score) as f32 / 3.0;
    scalar[8] = progress.shanten as f32 / 8.0;
    scalar[9] = progress.waits_count as f32 / 34.0;
    scalar[10] = round_state.players[actor].discards.len() as f32 / 18.0;
    scalar[11] = round_state.players[actor].melds.len() as f32 / 4.0;

    let all_pids_by_slot = [actor, other_pids[0], other_pids[1], other_pids[2]];
    for (slot, pid) in all_pids_by_slot.into_iter().enumerate() {
        scalar[12 + slot] = f32::from(reached[pid]);
    }

    let aka_total = tracker.aka_counts[0];
    let aka_m = tracker.aka_counts[1];
    let aka_p = tracker.aka_counts[2];
    let aka_s = tracker.aka_counts[3];
    scalar[16] = aka_total as f32 / 4.0;
    scalar[17] = f32::from(has_tanyao);
    scalar[18] = f32::from(aka_m > 0);
    scalar[19] = f32::from(aka_p > 0);
    scalar[20] = f32::from(aka_s > 0);
    scalar[21] = opponent_meld_counts[0];
    scalar[22] = opponent_meld_counts[1];
    scalar[23] = opponent_meld_counts[2];
    scalar[24] = (round_state.scores[other_pids[0]] - actor_score) as f32 / 30000.0;
    scalar[25] = (round_state.scores[other_pids[1]] - actor_score) as f32 / 30000.0;
    scalar[26] = (round_state.scores[other_pids[2]] - actor_score) as f32 / 30000.0;
    scalar[27] = jikaze as f32 / 3.0;
    scalar[28] = ((dora_count + aka_total) as f32 / 10.0).min(1.0);

    let bakaze_34 = match round_state.bakaze.as_str() {
        "E" => 27usize,
        "S" => 28,
        "W" => 29,
        "N" => 30,
        _ => 27,
    };
    let jikaze_34 = 27 + jikaze;
    let mut confirmed_han = 0usize;
    let mut honor_koutsu = [false; TILE_KIND_COUNT];
    for meld in &round_state.players[actor].melds {
        if !matches!(meld.kind.as_str(), "pon" | "daiminkan" | "ankan" | "kakan") {
            continue;
        }
        if let Some(tile34) = meld
            .pai
            .as_ref()
            .and_then(|tile| tile34_from_pai(tile))
            .map(|value| value as usize)
        {
            if tile34 >= 27 {
                honor_koutsu[tile34] = true;
            }
        }
    }
    for (tile34, &count) in tracker.hand_counts34.iter().enumerate() {
        if tile34 >= 27 && count >= 3 {
            honor_koutsu[tile34] = true;
        }
    }
    if honor_koutsu[bakaze_34] {
        confirmed_han += 1;
    }
    if jikaze_34 != bakaze_34 && honor_koutsu[jikaze_34] {
        confirmed_han += 1;
    }
    for tile34 in [31usize, 32, 33] {
        if honor_koutsu[tile34] {
            confirmed_han += 1;
        }
    }
    if has_tanyao {
        confirmed_han += 1;
    }
    if has_chinitsu {
        confirmed_han += if is_open { 5 } else { 6 };
    } else if has_honitsu {
        confirmed_han += if is_open { 2 } else { 3 };
    }
    confirmed_han += dora_count + aka_total;
    scalar[29] = (confirmed_han as f32 / 8.0).min(1.0);

    if total_tiles > 0 {
        scalar[30] = man_cnt as f32 / total_tiles as f32;
        scalar[31] = pin_cnt as f32 / total_tiles as f32;
        scalar[32] = sou_cnt as f32 / total_tiles as f32;
        scalar[33] = honor_cnt as f32 / total_tiles as f32;
    }
    scalar[34] = f32::from(has_junchan);
    scalar[35] = f32::from(has_chanta);
    scalar[36] = f32::from(has_honitsu);
    scalar[37] = f32::from(has_chinitsu);
    scalar[38] = (progress.ukeire_live_count as f32 / 34.0).min(1.0);
    scalar[39] = 0.0;
    scalar[40] = (tenpai_live_tile_count as f32 / 136.0).min(1.0);
    scalar[41] = pair_count as f32 / 7.0;
    scalar[42] = taatsu_count as f32 / 6.0;
    scalar[43] = ankoutsu_cnt as f32 / 4.0;
    scalar[44] = f32::from(round_state.players[actor].furiten);
    scalar[45] = riichi_threat_count as f32 / 3.0;
    scalar[46] = chiitoi as f32 / 6.0;
    scalar[47] = kokushi as f32 / 13.0;
    scalar[48] = f32::from(actor as i8 == round_state.oya);
    scalar[49] = is_all_last_like;
    scalar[50] = top_gap as f32 / 30000.0;
    scalar[51] = bottom_gap as f32 / 30000.0;
    scalar[52] = tiles_left as f32 / 70.0;
    scalar[53] = 0.0;
    scalar[54] = 0.0;
    scalar[55] = wait_hiddenness_ratio.clamp(0.0, 1.0);

    Ok(scalar
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect())
}

fn existing_export_sample_count(path: &Path) -> Result<Option<usize>, String> {
    read_npy_first_dim_from_zip(path, "state_tile_feat.npy")
}

#[derive(Debug)]
struct FileExportResult {
    ds_name: String,
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    produced_npz: bool,
    elapsed_s: f64,
}

fn resolved_export_jobs(requested_jobs: usize, total_files: usize) -> usize {
    if total_files <= 1 {
        return total_files.max(1);
    }
    let auto_jobs = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let jobs = if requested_jobs == 0 {
        auto_jobs
    } else {
        requested_jobs
    };
    jobs.clamp(1, total_files)
}

fn process_export_file(
    file: &str,
    output_path: &Path,
    resume: bool,
) -> Result<FileExportResult, String> {
    let file_t0 = Instant::now();
    let input_path = Path::new(file);
    let ds_name = input_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    let out_file = output_npz_path(output_path, file);

    if resume {
        match existing_export_sample_count(&out_file) {
            Ok(Some(existing_sample_count)) => {
                return Ok(FileExportResult {
                    ds_name,
                    exported_file_count: 1,
                    exported_sample_count: existing_sample_count,
                    processed_file_count: 0,
                    skipped_existing_file_count: 1,
                    produced_npz: true,
                    elapsed_s: file_t0.elapsed().as_secs_f64(),
                });
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!(
                    "[xmodel1 preprocess] ignoring corrupt existing export {}: {}",
                    out_file.display(),
                    err
                );
                if out_file.exists() {
                    fs::remove_file(&out_file).map_err(|remove_err| {
                        format!(
                            "failed to remove corrupt export {} after probe error ({err}): {remove_err}",
                            out_file.display()
                        )
                    })?;
                }
            }
        }
    }

    let records = collect_records_from_mjson(file)
        .map_err(|err| format!("failed to export {file}: {err}"))?;
    if !records.is_empty() {
        let out_dir = out_file.parent().unwrap_or(output_path);
        fs::create_dir_all(out_dir).map_err(|err| {
            format!(
                "failed to create dataset output dir {}: {err}",
                out_dir.display()
            )
        })?;
        write_full_npz(&out_file, &records)
            .map_err(|err| format!("failed to write export for {file}: {err}"))?;
        return Ok(FileExportResult {
            ds_name,
            exported_file_count: 1,
            exported_sample_count: records.len(),
            processed_file_count: 1,
            skipped_existing_file_count: 0,
            produced_npz: true,
            elapsed_s: file_t0.elapsed().as_secs_f64(),
        });
    }

    Ok(FileExportResult {
        ds_name,
        exported_file_count: 0,
        exported_sample_count: 0,
        processed_file_count: 0,
        skipped_existing_file_count: 0,
        produced_npz: false,
        elapsed_s: file_t0.elapsed().as_secs_f64(),
    })
}

fn apply_round_target_updates(
    records: &mut [FullRecord],
    updates: &[replay_core::RoundTargetUpdate],
) {
    replay_core::apply_round_target_updates(records, updates, |record, update| {
        record.score_delta_target = update.score_delta_target;
        record.global_value_target = update.global_value_target;
        record.win_target = update.win_target;
        record.dealin_target = update.dealin_target;
        record.pts_given_win_target = update.pts_given_win_target;
        record.pts_given_dealin_target = update.pts_given_dealin_target;
    });
}

fn collect_records_from_mjson(path: &str) -> Result<Vec<FullRecord>, String> {
    ensure_init();
    let events = replay_core::normalize_replay_events(path)?;
    // Stage 2 Rust 迁移: event_history 需要从完整 events 列表里按 event_index 切窗口,
    // 通过闭包捕获 &events 传进 encode 函数。drive_export_records 本身也持有 &events
    // 的不可变借用,这里再拿一个不可变 slice 引用不冲突。
    let events_slice: &[Value] = events.as_slice();
    replay_core::drive_export_records(
        &events,
        SCORE_NORM,
        MC_RETURN_GAMMA,
        |ctx| {
            let Some(decision) = replay_core::build_special_actor_chosen_action_context(ctx)?
            else {
                return Ok(None);
            };
            let sample_type = if decision.decision.event.et == "reach" {
                1
            } else {
                2
            };
            let event_history = replay_core::compute_event_history(
                events_slice,
                decision.decision.event.event_index,
            );
            encode_special_sample_record(
                decision.decision.event.state,
                decision.decision.event.actor,
                &decision.decision.legal_actions,
                &decision.chosen_action,
                sample_type,
                decision.decision.event.event_index,
                event_history,
            )
            .map(Some)
        },
        |ctx| {
            let Some(discard) = replay_core::build_discard_decision_context(ctx) else {
                return Ok(None);
            };
            let event_history =
                replay_core::compute_event_history(events_slice, discard.event.event_index);
            encode_candidate_features(
                discard.event.state,
                discard.event.core_state,
                discard.event.actor,
                &discard.chosen_tile,
                discard.event.event_index,
                event_history,
            )
            .map(Some)
        },
        |ctx| {
            let reaction = replay_core::build_reaction_none_context(ctx);
            let event_history =
                replay_core::compute_event_history(events_slice, reaction.reaction.event_index);
            encode_special_sample_record(
                reaction.reaction.state,
                reaction.reaction.actor,
                reaction.reaction.legal_actions,
                &reaction.chosen_action,
                2,
                reaction.reaction.event_index,
                event_history,
            )
            .map(Some)
        },
        apply_round_target_updates,
    )
}

fn write_full_npz(path: &Path, records: &[FullRecord]) -> Result<(), String> {
    let temp_path = temp_npz_path(path);
    let file = fs::File::create(&temp_path)
        .map_err(|err| format!("failed to create temp npz {}: {err}", temp_path.display()))?;
    let mut zip = ZipWriter::new(file);
    let n = records.len();

    let mut state_tile_feat = Vec::with_capacity(n * XMODEL1_STATE_TILE_CHANNELS * TILE_KIND_COUNT);
    let mut state_scalar = Vec::with_capacity(n * XMODEL1_STATE_SCALAR_DIM);
    let mut candidate_feat =
        Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM);
    let mut candidate_tile_id = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
    let mut candidate_mask = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
    let mut candidate_flags =
        Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM);
    let mut chosen_candidate_idx = Vec::with_capacity(n);
    let mut candidate_quality = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
    let mut candidate_rank = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
    let mut candidate_hard_bad = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
    let mut special_candidate_feat = Vec::with_capacity(
        n * XMODEL1_MAX_SPECIAL_CANDIDATES * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    );
    let mut special_candidate_type_id = Vec::with_capacity(n * XMODEL1_MAX_SPECIAL_CANDIDATES);
    let mut special_candidate_mask = Vec::with_capacity(n * XMODEL1_MAX_SPECIAL_CANDIDATES);
    let mut special_candidate_quality = Vec::with_capacity(n * XMODEL1_MAX_SPECIAL_CANDIDATES);
    let mut special_candidate_rank = Vec::with_capacity(n * XMODEL1_MAX_SPECIAL_CANDIDATES);
    let mut special_candidate_hard_bad = Vec::with_capacity(n * XMODEL1_MAX_SPECIAL_CANDIDATES);
    let mut chosen_special_candidate_idx = Vec::with_capacity(n);
    let mut action_idx_target = Vec::with_capacity(n);
    let mut global_value_target = Vec::with_capacity(n);
    let mut score_delta_target = Vec::with_capacity(n);
    let mut win_target = Vec::with_capacity(n);
    let mut dealin_target = Vec::with_capacity(n);
    let mut pts_given_win_target = Vec::with_capacity(n);
    let mut pts_given_dealin_target = Vec::with_capacity(n);
    // Stage 2: opp_tenpai × 3 (顺序 actor+1, actor+2, actor+3 mod 4)。
    let mut opp_tenpai_target: Vec<f32> = Vec::with_capacity(n * 3);
    // Stage 2 Rust 迁移: event_history flatten 成 int16,shape [n, 48, 5]。
    let mut event_history: Vec<i16> = Vec::with_capacity(
        n * replay_core::EVENT_HISTORY_LEN * replay_core::EVENT_HISTORY_FEATURE_DIM,
    );
    let mut offense_quality_target = Vec::with_capacity(n);
    let mut sample_type = Vec::with_capacity(n);
    let mut actor = Vec::with_capacity(n);
    let mut event_index = Vec::with_capacity(n);
    let mut kyoku = Vec::with_capacity(n);
    let mut honba = Vec::with_capacity(n);
    let mut is_open_hand = Vec::with_capacity(n);

    for record in records {
        state_tile_feat.extend_from_slice(&record.state_tile_feat);
        state_scalar.extend_from_slice(&record.state_scalar);
        candidate_feat.extend_from_slice(&record.candidate_feat);
        candidate_tile_id.extend_from_slice(&record.candidate_tile_id);
        candidate_mask.extend_from_slice(&record.candidate_mask);
        candidate_flags.extend_from_slice(&record.candidate_flags);
        chosen_candidate_idx.push(record.chosen_candidate_idx);
        candidate_quality.extend_from_slice(&record.candidate_quality);
        candidate_rank.extend_from_slice(&record.candidate_rank);
        candidate_hard_bad.extend_from_slice(&record.candidate_hard_bad);
        special_candidate_feat.extend_from_slice(&record.special_candidate_feat);
        special_candidate_type_id.extend_from_slice(&record.special_candidate_type_id);
        special_candidate_mask.extend_from_slice(&record.special_candidate_mask);
        special_candidate_quality.extend_from_slice(&record.special_candidate_quality);
        special_candidate_rank.extend_from_slice(&record.special_candidate_rank);
        special_candidate_hard_bad.extend_from_slice(&record.special_candidate_hard_bad);
        chosen_special_candidate_idx.push(record.chosen_special_candidate_idx);
        action_idx_target.push(record.action_idx_target);
        global_value_target.push(record.global_value_target);
        score_delta_target.push(record.score_delta_target);
        win_target.push(record.win_target);
        dealin_target.push(record.dealin_target);
        pts_given_win_target.push(record.pts_given_win_target);
        pts_given_dealin_target.push(record.pts_given_dealin_target);
        opp_tenpai_target.extend_from_slice(&record.opp_tenpai_target);
        for row in record.event_history.iter() {
            event_history.extend_from_slice(row);
        }
        offense_quality_target.push(record.offense_quality_target);
        sample_type.push(record.sample_type);
        actor.push(record.actor);
        event_index.push(record.event_index);
        kyoku.push(record.kyoku);
        honba.push(record.honba);
        is_open_hand.push(record.is_open_hand);
    }

    write_npy_f16(
        &mut zip,
        "state_tile_feat.npy",
        &[n, XMODEL1_STATE_TILE_CHANNELS, TILE_KIND_COUNT],
        &state_tile_feat,
    )?;
    write_npy_f16(
        &mut zip,
        "state_scalar.npy",
        &[n, XMODEL1_STATE_SCALAR_DIM],
        &state_scalar,
    )?;
    write_npy_f16(
        &mut zip,
        "candidate_feat.npy",
        &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM],
        &candidate_feat,
    )?;
    write_npy_i16(
        &mut zip,
        "candidate_tile_id.npy",
        &[n, XMODEL1_MAX_CANDIDATES],
        &candidate_tile_id,
    )?;
    write_npy_u8(
        &mut zip,
        "candidate_mask.npy",
        &[n, XMODEL1_MAX_CANDIDATES],
        &candidate_mask,
    )?;
    write_npy_u8(
        &mut zip,
        "candidate_flags.npy",
        &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM],
        &candidate_flags,
    )?;
    write_npy_i16(
        &mut zip,
        "chosen_candidate_idx.npy",
        &[n],
        &chosen_candidate_idx,
    )?;
    write_npy_f32(
        &mut zip,
        "candidate_quality_score.npy",
        &[n, XMODEL1_MAX_CANDIDATES],
        &candidate_quality,
    )?;
    write_npy_i8(
        &mut zip,
        "candidate_rank_bucket.npy",
        &[n, XMODEL1_MAX_CANDIDATES],
        &candidate_rank,
    )?;
    write_npy_u8(
        &mut zip,
        "candidate_hard_bad_flag.npy",
        &[n, XMODEL1_MAX_CANDIDATES],
        &candidate_hard_bad,
    )?;
    write_npy_f16(
        &mut zip,
        "special_candidate_feat.npy",
        &[
            n,
            XMODEL1_MAX_SPECIAL_CANDIDATES,
            XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
        ],
        &special_candidate_feat,
    )?;
    write_npy_i16(
        &mut zip,
        "special_candidate_type_id.npy",
        &[n, XMODEL1_MAX_SPECIAL_CANDIDATES],
        &special_candidate_type_id,
    )?;
    write_npy_u8(
        &mut zip,
        "special_candidate_mask.npy",
        &[n, XMODEL1_MAX_SPECIAL_CANDIDATES],
        &special_candidate_mask,
    )?;
    write_npy_f32(
        &mut zip,
        "special_candidate_quality_score.npy",
        &[n, XMODEL1_MAX_SPECIAL_CANDIDATES],
        &special_candidate_quality,
    )?;
    write_npy_i8(
        &mut zip,
        "special_candidate_rank_bucket.npy",
        &[n, XMODEL1_MAX_SPECIAL_CANDIDATES],
        &special_candidate_rank,
    )?;
    write_npy_u8(
        &mut zip,
        "special_candidate_hard_bad_flag.npy",
        &[n, XMODEL1_MAX_SPECIAL_CANDIDATES],
        &special_candidate_hard_bad,
    )?;
    write_npy_i16(
        &mut zip,
        "chosen_special_candidate_idx.npy",
        &[n],
        &chosen_special_candidate_idx,
    )?;
    write_npy_i16(&mut zip, "action_idx_target.npy", &[n], &action_idx_target)?;
    write_npy_f32(
        &mut zip,
        "global_value_target.npy",
        &[n],
        &global_value_target,
    )?;
    write_npy_f32(
        &mut zip,
        "score_delta_target.npy",
        &[n],
        &score_delta_target,
    )?;
    write_npy_f32(&mut zip, "win_target.npy", &[n], &win_target)?;
    write_npy_f32(&mut zip, "dealin_target.npy", &[n], &dealin_target)?;
    write_npy_f32(
        &mut zip,
        "pts_given_win_target.npy",
        &[n],
        &pts_given_win_target,
    )?;
    write_npy_f32(
        &mut zip,
        "pts_given_dealin_target.npy",
        &[n],
        &pts_given_dealin_target,
    )?;
    write_npy_f32(
        &mut zip,
        "opp_tenpai_target.npy",
        &[n, 3],
        &opp_tenpai_target,
    )?;
    write_npy_i16(
        &mut zip,
        "event_history.npy",
        &[
            n,
            replay_core::EVENT_HISTORY_LEN,
            replay_core::EVENT_HISTORY_FEATURE_DIM,
        ],
        &event_history,
    )?;
    write_npy_f32(
        &mut zip,
        "offense_quality_target.npy",
        &[n],
        &offense_quality_target,
    )?;
    write_npy_i8(&mut zip, "sample_type.npy", &[n], &sample_type)?;
    write_npy_i8(&mut zip, "actor.npy", &[n], &actor)?;
    write_npy_i32(&mut zip, "event_index.npy", &[n], &event_index)?;
    write_npy_i8(&mut zip, "kyoku.npy", &[n], &kyoku)?;
    write_npy_i8(&mut zip, "honba.npy", &[n], &honba)?;
    write_npy_u8(&mut zip, "is_open_hand.npy", &[n], &is_open_hand)?;
    finalize_temp_npz(zip, &temp_path, path, true)
}

fn write_manifest(
    output_dir: &str,
    files: &[String],
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    shard_file_counts: &BTreeMap<String, usize>,
    shard_sample_counts: &BTreeMap<String, usize>,
    used_fallback: bool,
    export_mode: &'static str,
) -> Result<String, String> {
    let output_path = Path::new(output_dir);
    let manifest = ExportManifest {
        schema_name: XMODEL1_SCHEMA_NAME,
        schema_version: XMODEL1_SCHEMA_VERSION,
        max_candidates: XMODEL1_MAX_CANDIDATES,
        candidate_feature_dim: XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim: XMODEL1_CANDIDATE_FLAG_DIM,
        file_count: files.len(),
        exported_file_count,
        exported_sample_count,
        processed_file_count,
        skipped_existing_file_count,
        shard_file_counts: shard_file_counts.clone(),
        shard_sample_counts: shard_sample_counts.clone(),
        used_fallback,
        export_mode,
        files: files.to_vec(),
    };
    write_json_manifest(output_path, "xmodel1_export_manifest.json", &manifest)
}

pub fn build_xmodel1_discard_records(
    data_dirs: &[String],
    output_dir: &str,
    smoke: bool,
) -> Result<(usize, String, bool), String> {
    build_xmodel1_discard_records_with_options(
        data_dirs,
        output_dir,
        ExportRunOptions {
            smoke,
            resume: false,
            progress_every: 0,
            jobs: 1,
            limit_files: 0,
        },
    )
}

pub fn build_xmodel1_discard_records_with_options(
    data_dirs: &[String],
    output_dir: &str,
    options: ExportRunOptions,
) -> Result<(usize, String, bool), String> {
    let mut files = collect_mjson_files(data_dirs, options.smoke)?;
    if options.limit_files > 0 {
        files.truncate(options.limit_files);
    }
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path).map_err(|err| {
        format!(
            "failed to create xmodel1 output dir {}: {err}",
            output_path.display()
        )
    })?;
    let start = Instant::now();
    let jobs_count = resolved_export_jobs(options.jobs, files.len());
    let mut produced_npz = false;
    let mut exported_file_count = 0usize;
    let mut exported_sample_count = 0usize;
    let mut processed_file_count = 0usize;
    let mut skipped_existing_file_count = 0usize;
    let mut shard_file_counts = BTreeMap::<String, usize>::new();
    let mut shard_sample_counts = BTreeMap::<String, usize>::new();
    let progress_every = options.progress_every.max(1);
    let mut recent_file_seconds = VecDeque::with_capacity(32);

    let (tx, rx) = mpsc::channel::<Result<FileExportResult, String>>();
    let work_queue = Arc::new(Mutex::new(files.clone()));
    let mut handles = Vec::new();
    for _ in 0..jobs_count {
        let tx = tx.clone();
        let queue = work_queue.clone();
        let output_dir = output_path.to_path_buf();
        let resume = options.resume;
        handles.push(thread::spawn(move || loop {
            let maybe_file = {
                let mut queue = queue.lock().expect("work queue poisoned");
                queue.pop()
            };
            let Some(file) = maybe_file else {
                break;
            };
            let result = process_export_file(&file, &output_dir, resume);
            let _ = tx.send(result);
        }));
    }
    drop(tx);

    let mut done = 0usize;
    let mut first_error: Option<String> = None;
    while let Ok(result) = rx.recv() {
        done += 1;
        match result {
            Ok(result) => {
                produced_npz |= result.produced_npz;
                exported_file_count += result.exported_file_count;
                exported_sample_count += result.exported_sample_count;
                processed_file_count += result.processed_file_count;
                skipped_existing_file_count += result.skipped_existing_file_count;
                if result.exported_file_count > 0 {
                    *shard_file_counts.entry(result.ds_name.clone()).or_insert(0) +=
                        result.exported_file_count;
                    *shard_sample_counts.entry(result.ds_name).or_insert(0) +=
                        result.exported_sample_count;
                }
                recent_file_seconds.push_back(result.elapsed_s);
                while recent_file_seconds.len() > 32 {
                    recent_file_seconds.pop_front();
                }
            }
            Err(err) => {
                first_error = Some(err);
                break;
            }
        }
        if options.progress_every > 0 && ((done % progress_every == 0) || (done == files.len())) {
            print_export_progress(
                "xmodel1",
                &start,
                done,
                files.len(),
                processed_file_count,
                skipped_existing_file_count,
                exported_sample_count,
                &recent_file_seconds,
                jobs_count,
            );
        }
    }
    for handle in handles {
        let _ = handle.join();
    }
    if let Some(err) = first_error {
        return Err(err);
    }
    let export_mode = if options.smoke {
        "rust_full_npz_smoke_export"
    } else {
        "rust_full_npz_export"
    };
    let manifest_path = write_manifest(
        output_dir,
        &files,
        exported_file_count,
        exported_sample_count,
        processed_file_count,
        skipped_existing_file_count,
        &shard_file_counts,
        &shard_sample_counts,
        false,
        export_mode,
    )?;
    Ok((files.len(), manifest_path, produced_npz))
}
