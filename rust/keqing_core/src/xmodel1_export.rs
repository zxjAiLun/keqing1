//! Xmodel1 export scaffolding.
//!
//! This module now owns the Rust-side request surface for Xmodel1 preprocessing:
//! input directory validation, `.mjson` discovery, and manifest emission.
//! The full candidate-centric `.npz` export is still pending, but the API shape
//! is no longer a placeholder with no inputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::io::{Seek, Write};

use serde::Serialize;
use serde_json::Value;
use zip::write::FileOptions;
use zip::CompressionMethod;
use zip::ZipWriter;

use crate::progress_delta::calc_required_tiles;
use crate::shanten_table::{calc_shanten_all, ensure_init};
use crate::xmodel1_schema::{
    validate_candidate_mask_and_choice,
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
};

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
    used_fallback: bool,
    export_mode: &'a str,
    files: Vec<String>,
}

const XMODEL1_STATE_TILE_CHANNELS: usize = 57;
const XMODEL1_STATE_SCALAR_DIM: usize = 56;

#[derive(Debug, Clone)]
struct SmokeRecord {
    actor: i8,
    event_index: i32,
    kyoku: i8,
    honba: i8,
    oya: i8,
    bakaze: String,
    hand_tile34s: Vec<i16>,
    chosen_tile34: i16,
}

fn collect_mjson_files(data_dirs: &[String], smoke: bool) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for data_dir in data_dirs {
        let path = Path::new(data_dir);
        if !path.exists() {
            return Err(format!("data dir does not exist: {data_dir}"));
        }
        if !path.is_dir() {
            return Err(format!("data dir is not a directory: {data_dir}"));
        }
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|err| format!("failed to read dir {data_dir}: {err}"))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mjson"))
            .collect();
        entries.sort();
        if smoke {
            entries.truncate(1);
        }
        out.extend(entries.into_iter().map(|p| p.to_string_lossy().to_string()));
    }
    Ok(out)
}

fn tile34_from_pai(pai: &str) -> Option<i16> {
    let mut chars = pai.chars();
    let first = chars.next()?;
    if let Some(suit) = chars.next() {
        if matches!(suit, 'm' | 'p' | 's') {
            let rank = if first == '0' { 5 } else { first.to_digit(10)? as i16 };
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return None,
            };
            return Some(base + rank - 1);
        }
    }
    match pai {
        "E" => Some(27),
        "S" => Some(28),
        "W" => Some(29),
        "N" => Some(30),
        "P" => Some(31),
        "F" => Some(32),
        "C" => Some(33),
        _ => None,
    }
}

fn remove_one_tile(hand: &mut Vec<i16>, tile34: i16) {
    if let Some(pos) = hand.iter().position(|v| *v == tile34) {
        hand.remove(pos);
    }
}

fn tile34s_from_tehai(value: &Value) -> Vec<i16> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().and_then(tile34_from_pai))
                .collect()
        })
        .unwrap_or_default()
}

fn collect_records_from_mjson(path: &str) -> Result<Vec<SmokeRecord>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut kyoku: i8 = 1;
    let mut honba: i8 = 0;
    let mut oya: i8 = 0;
    let mut bakaze = String::from("E");
    let mut hands: [Vec<i16>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut out = Vec::new();
    for (event_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("failed to parse JSON line in {path}: {err}"))?;
        let etype = value.get("type").and_then(Value::as_str).unwrap_or("");
        if etype == "start_kyoku" {
            kyoku = value.get("kyoku").and_then(Value::as_i64).unwrap_or(1) as i8;
            honba = value.get("honba").and_then(Value::as_i64).unwrap_or(0) as i8;
            oya = value.get("oya").and_then(Value::as_i64).unwrap_or(0) as i8;
            bakaze = value.get("bakaze").and_then(Value::as_str).unwrap_or("E").to_string();
            if let Some(tehais) = value.get("tehais").and_then(Value::as_array) {
                for (idx, tehai) in tehais.iter().enumerate().take(4) {
                    hands[idx] = tile34s_from_tehai(tehai);
                }
            }
            continue;
        }
        if etype == "tsumo" {
            let actor = value.get("actor").and_then(Value::as_i64).unwrap_or(0) as usize;
            let pai = value.get("pai").and_then(Value::as_str).unwrap_or("");
            if actor < 4 {
                if let Some(tile34) = tile34_from_pai(pai) {
                    hands[actor].push(tile34);
                }
            }
            continue;
        }
        if etype != "dahai" {
            if matches!(etype, "chi" | "pon" | "daiminkan" | "ankan" | "kakan") {
                let actor = value.get("actor").and_then(Value::as_i64).unwrap_or(0) as usize;
                if actor < 4 {
                    if let Some(consumed) = value.get("consumed").and_then(Value::as_array) {
                        for item in consumed {
                            if let Some(tile34) = item.as_str().and_then(tile34_from_pai) {
                                remove_one_tile(&mut hands[actor], tile34);
                            }
                        }
                    }
                    if etype == "ankan" {
                        if let Some(pai) = value.get("pai").and_then(Value::as_str).and_then(tile34_from_pai) {
                            remove_one_tile(&mut hands[actor], pai);
                        }
                    }
                    if etype == "kakan" {
                        if let Some(pai) = value.get("pai").and_then(Value::as_str).and_then(tile34_from_pai) {
                            remove_one_tile(&mut hands[actor], pai);
                        }
                    }
                }
            }
            continue;
        }
        let actor = value.get("actor").and_then(Value::as_i64).unwrap_or(0) as i8;
        let pai = value.get("pai").and_then(Value::as_str).unwrap_or("");
        let Some(tile34) = tile34_from_pai(pai) else {
            continue;
        };
        let actor_idx = actor as usize;
        let hand_before = if actor_idx < 4 { hands[actor_idx].clone() } else { Vec::new() };
        out.push(SmokeRecord {
            actor,
            event_index: event_index as i32,
            kyoku,
            honba,
            oya,
            bakaze: bakaze.clone(),
            hand_tile34s: hand_before,
            chosen_tile34: tile34,
        });
        if actor_idx < 4 {
            remove_one_tile(&mut hands[actor_idx], tile34);
        }
    }
    Ok(out)
}

fn pair_taatsu_metrics(hand_tile34s: &[i16]) -> (usize, usize) {
    let mut counts = [0usize; 34];
    for tile in hand_tile34s {
        let idx = *tile as usize;
        if idx < 34 {
            counts[idx] += 1;
        }
    }
    let pair_count = counts.iter().filter(|c| **c >= 2).count();
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

fn counts34_from_hand(hand_tile34s: &[i16]) -> [u8; 34] {
    let mut counts = [0u8; 34];
    for tile in hand_tile34s {
        let idx = *tile as usize;
        if idx < 34 {
            counts[idx] = counts[idx].saturating_add(1);
        }
    }
    counts
}

fn yakuhai_tiles(bakaze: &str, actor: i8, oya: i8) -> [bool; 34] {
    let mut flags = [false; 34];
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
    if jikaze < 34 {
        flags[jikaze] = true;
    }
    flags
}

fn required_wait_metrics(counts34: &[u8; 34], visible_counts34: &[u8; 34]) -> (usize, usize) {
    let total_tiles: u8 = counts34.iter().sum();
    let len_div3 = total_tiles / 3;
    let waits = calc_required_tiles(counts34, visible_counts34, len_div3);
    let wait_type_count = waits.len();
    let wait_live_count = waits.into_iter().map(|item| item.live_count as usize).sum();
    (wait_type_count, wait_live_count)
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|v| v.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    )
    .into_bytes();
    let preamble_len = 10usize;
    while (preamble_len + header.len() + 1) % 16 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = Vec::with_capacity(preamble_len + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out
}

fn write_npy_f16<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[u16]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<f2", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes()).map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

fn write_npy_f32<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<f4", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes()).map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

fn write_npy_i16<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[i16]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<i2", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes()).map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

fn write_npy_i32<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[i32]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<i4", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes()).map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

fn write_npy_i8<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[i8]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("|i1", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    let bytes: Vec<u8> = data.iter().map(|v| *v as u8).collect();
    zip.write_all(&bytes).map_err(|err| format!("failed to write data {name}: {err}"))?;
    Ok(())
}

fn write_npy_u8<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, shape: &[usize], data: &[u8]) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("|u1", shape)).map_err(|err| format!("failed to write header {name}: {err}"))?;
    zip.write_all(data).map_err(|err| format!("failed to write data {name}: {err}"))?;
    Ok(())
}

fn write_smoke_npz(path: &Path, records: &[SmokeRecord]) -> Result<(), String> {
    ensure_init();
    let file = fs::File::create(path)
        .map_err(|err| format!("failed to create npz {}: {err}", path.display()))?;
    let mut zip = ZipWriter::new(file);
    let n = records.len();

    let zeros_f16_state_tile = vec![0u16; n * XMODEL1_STATE_TILE_CHANNELS * 34];
    let zeros_f16_state_scalar = vec![0u16; n * XMODEL1_STATE_SCALAR_DIM];
    let mut candidate_feat = vec![0u16; n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM];
    let mut candidate_tile_id = vec![-1i16; n * XMODEL1_MAX_CANDIDATES];
    let mut candidate_mask = vec![0u8; n * XMODEL1_MAX_CANDIDATES];
    let mut candidate_flags = vec![0u8; n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM];
    let mut chosen_candidate_idx = vec![0i16; n];
    let mut candidate_quality = vec![0.0f32; n * XMODEL1_MAX_CANDIDATES];
    let mut candidate_rank = vec![0i8; n * XMODEL1_MAX_CANDIDATES];
    let mut candidate_hard_bad = vec![0u8; n * XMODEL1_MAX_CANDIDATES];
    let zeros_f32_targets = vec![0.0f32; n];
    let sample_type = vec![0i8; n];
    let actor: Vec<i8> = records.iter().map(|r| r.actor).collect();
    let event_index: Vec<i32> = records.iter().map(|r| r.event_index).collect();
    let kyoku: Vec<i8> = records.iter().map(|r| r.kyoku).collect();
    let honba: Vec<i8> = records.iter().map(|r| r.honba).collect();
    let is_open_hand = vec![0u8; n];

    for (idx, record) in records.iter().enumerate() {
        let base = idx * XMODEL1_MAX_CANDIDATES;
        let yakuhai = yakuhai_tiles(&record.bakaze, record.actor, record.oya);
        let before_counts34 = counts34_from_hand(&record.hand_tile34s);
        let before_visible = before_counts34;
        let before_total: u8 = before_counts34.iter().sum();
        let before_shanten = calc_shanten_all(&before_counts34, before_total / 3) as i32;
        let (_before_wait_type_count, before_wait_live_count) =
            required_wait_metrics(&before_counts34, &before_visible);
        let before_wait_live = before_wait_live_count as i32;
        let mut unique_tiles = record.hand_tile34s.clone();
        unique_tiles.sort_unstable();
        unique_tiles.dedup();
        for (slot, tile34) in unique_tiles.into_iter().take(XMODEL1_MAX_CANDIDATES).enumerate() {
            let out_idx = base + slot;
            candidate_tile_id[out_idx] = tile34;
            candidate_mask[out_idx] = 1;
            if tile34 == record.chosen_tile34 {
                chosen_candidate_idx[idx] = slot as i16;
            }
            let mut after_hand = record.hand_tile34s.clone();
            if let Some(pos) = after_hand.iter().position(|v| *v == tile34) {
                after_hand.remove(pos);
            }
            let after_counts34 = counts34_from_hand(&after_hand);
            let mut after_visible = after_counts34;
            if tile34 >= 0 && (tile34 as usize) < 34 {
                after_visible[tile34 as usize] = after_visible[tile34 as usize].saturating_add(1).min(4);
            }
            let after_total: u8 = after_counts34.iter().sum();
            let after_shanten = calc_shanten_all(&after_counts34, after_total / 3) as i32;
            let (after_wait_type_count, after_wait_live_count) =
                required_wait_metrics(&after_counts34, &after_visible);
            let after_wait_live = after_wait_live_count as i32;
            let (pair_count, taatsu_count) = pair_taatsu_metrics(&after_hand);
            let is_honor = (tile34 >= 27) as u8;
            let is_terminal = matches!(tile34, 0 | 8 | 9 | 17 | 18 | 26) as u8;
            let is_yakuhai = yakuhai[tile34 as usize] as u8;
            let tile_count_before = record.hand_tile34s.iter().filter(|v| **v == tile34).count();
            let drop_open_yakuhai_pair = (is_yakuhai == 1 && tile_count_before >= 2) as u8;
            let drop_dual_pon_value = (tile_count_before >= 2) as u8;
            let break_tenpai = (before_shanten == 0 && after_shanten > 0) as u8;
            let break_best_wait = (before_shanten == 0 && after_wait_live < before_wait_live) as u8;
            let break_meld_structure = (before_shanten <= 1 && after_shanten > before_shanten) as u8;
            let hard_bad = (break_tenpai == 1 || drop_open_yakuhai_pair == 1 || break_meld_structure == 1) as u8;
            let quality = 1.5 * (after_shanten == 0) as i32 as f32
                - 0.8 * (after_shanten as f32)
                + 0.06 * (after_wait_live as f32)
                - 2.0 * (break_tenpai as f32)
                - 1.0 * (break_meld_structure as f32)
                - 1.2 * (drop_open_yakuhai_pair as f32)
                - 0.7 * (drop_dual_pon_value as f32);
            candidate_quality[out_idx] = quality;
            candidate_rank[out_idx] = if hard_bad == 1 { 0 } else if quality >= 1.0 { 3 } else if quality >= 0.0 { 2 } else { 1 };
            candidate_hard_bad[out_idx] = hard_bad;
            let feat_offset = out_idx * XMODEL1_CANDIDATE_FEATURE_DIM;
            let mut feat = [0f32; XMODEL1_CANDIDATE_FEATURE_DIM];
            feat[0] = after_shanten as f32 / 8.0;
            feat[1] = (after_shanten == 0) as i32 as f32;
            feat[2] = (after_wait_type_count as f32 / 34.0).min(1.0);
            feat[3] = (after_wait_live as f32 / 20.0).min(1.0);
            feat[4] = (after_wait_type_count as f32 / 34.0).min(1.0);
            feat[5] = (after_wait_live as f32 / 34.0).min(1.0);
            feat[6] = (after_wait_live as f32 / 20.0).min(1.0);
            feat[7] = (after_wait_live as f32 / 20.0).min(1.0);
            feat[8] = pair_count as f32 / 7.0;
            feat[9] = taatsu_count as f32 / 6.0;
            feat[10] = 1.0 - (((after_shanten - before_shanten).abs() as f32) / 2.0).min(1.0);
            feat[11] = 0.0;
            feat[12] = 1.0 - drop_open_yakuhai_pair as f32;
            feat[13] = 1.0 - drop_open_yakuhai_pair as f32;
            feat[14] = 1.0 - drop_dual_pon_value as f32;
            feat[15] = 0.0;
            feat[16] = 0.0;
            feat[17] = 0.0;
            feat[18] = 0.0;
            feat[19] = 0.0;
            feat[20] = 0.0;
            for (i, value) in feat.iter().enumerate() {
                candidate_feat[feat_offset + i] = half::f16::from_f32(*value).to_bits();
            }
            let flag_offset = out_idx * XMODEL1_CANDIDATE_FLAG_DIM;
            candidate_flags[flag_offset] = break_tenpai;
            candidate_flags[flag_offset + 1] = break_best_wait;
            candidate_flags[flag_offset + 2] = break_meld_structure;
            candidate_flags[flag_offset + 3] = drop_open_yakuhai_pair;
            candidate_flags[flag_offset + 4] = drop_dual_pon_value;
            candidate_flags[flag_offset + 5] = is_honor;
            candidate_flags[flag_offset + 6] = is_terminal;
            candidate_flags[flag_offset + 9] = is_yakuhai;
        }
    }

    write_npy_f16(&mut zip, "state_tile_feat.npy", &[n, XMODEL1_STATE_TILE_CHANNELS, 34], &zeros_f16_state_tile)?;
    write_npy_f16(&mut zip, "state_scalar.npy", &[n, XMODEL1_STATE_SCALAR_DIM], &zeros_f16_state_scalar)?;
    write_npy_f16(&mut zip, "candidate_feat.npy", &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM], &candidate_feat)?;
    write_npy_i16(&mut zip, "candidate_tile_id.npy", &[n, XMODEL1_MAX_CANDIDATES], &candidate_tile_id)?;
    write_npy_u8(&mut zip, "candidate_mask.npy", &[n, XMODEL1_MAX_CANDIDATES], &candidate_mask)?;
    write_npy_u8(&mut zip, "candidate_flags.npy", &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM], &candidate_flags)?;
    write_npy_i16(&mut zip, "chosen_candidate_idx.npy", &[n], &chosen_candidate_idx)?;
    write_npy_f32(&mut zip, "candidate_quality_score.npy", &[n, XMODEL1_MAX_CANDIDATES], &candidate_quality)?;
    write_npy_i8(&mut zip, "candidate_rank_bucket.npy", &[n, XMODEL1_MAX_CANDIDATES], &candidate_rank)?;
    write_npy_u8(&mut zip, "candidate_hard_bad_flag.npy", &[n, XMODEL1_MAX_CANDIDATES], &candidate_hard_bad)?;
    write_npy_f32(&mut zip, "global_value_target.npy", &[n], &zeros_f32_targets)?;
    write_npy_f32(&mut zip, "score_delta_target.npy", &[n], &zeros_f32_targets)?;
    write_npy_f32(&mut zip, "win_target.npy", &[n], &zeros_f32_targets)?;
    write_npy_f32(&mut zip, "dealin_target.npy", &[n], &zeros_f32_targets)?;
    write_npy_f32(&mut zip, "offense_quality_target.npy", &[n], &zeros_f32_targets)?;
    write_npy_i8(&mut zip, "sample_type.npy", &[n], &sample_type)?;
    write_npy_i8(&mut zip, "actor.npy", &[n], &actor)?;
    write_npy_i32(&mut zip, "event_index.npy", &[n], &event_index)?;
    write_npy_i8(&mut zip, "kyoku.npy", &[n], &kyoku)?;
    write_npy_i8(&mut zip, "honba.npy", &[n], &honba)?;
    write_npy_u8(&mut zip, "is_open_hand.npy", &[n], &is_open_hand)?;
    zip.finish().map_err(|err| format!("failed to finish npz {}: {err}", path.display()))?;
    Ok(())
}

fn write_manifest(
    output_dir: &str,
    files: &[String],
    used_fallback: bool,
    export_mode: &'static str,
) -> Result<String, String> {
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .map_err(|err| format!("failed to create output dir {output_dir}: {err}"))?;
    let manifest = ExportManifest {
        schema_name: XMODEL1_SCHEMA_NAME,
        schema_version: XMODEL1_SCHEMA_VERSION,
        max_candidates: XMODEL1_MAX_CANDIDATES,
        candidate_feature_dim: XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim: XMODEL1_CANDIDATE_FLAG_DIM,
        file_count: files.len(),
        used_fallback,
        export_mode,
        files: files.to_vec(),
    };
    let manifest_path = output_path.join("xmodel1_export_manifest.json");
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to serialize export manifest: {err}"))?;
    fs::write(&manifest_path, text)
        .map_err(|err| format!("failed to write manifest {}: {err}", manifest_path.display()))?;
    Ok(manifest_path.to_string_lossy().to_string())
}

pub fn build_xmodel1_discard_records(
    data_dirs: &[String],
    output_dir: &str,
    smoke: bool,
) -> Result<(usize, String, bool), String> {
    let files = collect_mjson_files(data_dirs, smoke)?;
    let output_path = Path::new(output_dir);
    let mut produced_npz = false;
    for file in &files {
        let records = collect_records_from_mjson(file)?;
        if records.is_empty() {
            continue;
        }
        let input_path = Path::new(file);
        let ds_name = input_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("dataset");
        let out_dir = output_path.join(ds_name);
        fs::create_dir_all(&out_dir)
            .map_err(|err| format!("failed to create dataset output dir {}: {err}", out_dir.display()))?;
        let out_file = out_dir.join(
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| format!("{s}.npz"))
                .unwrap_or_else(|| "sample.npz".to_string()),
        );
        write_smoke_npz(&out_file, &records)?;
        produced_npz = true;
    }
    let export_mode = if produced_npz {
        if smoke { "smoke_npz_export" } else { "npz_export" }
    } else {
        "manifest_only"
    };
    let manifest_path = write_manifest(output_dir, &files, !produced_npz, export_mode)?;
    Ok((files.len(), manifest_path, produced_npz))
}
