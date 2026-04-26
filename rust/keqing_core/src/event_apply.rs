use serde_json::{json, Value};

use crate::state_core::{GameStateCore, PlayerStateCore};
use crate::types::{DiscardEntry, LastDiscard};

fn normalize_or_keep_aka(tile: &str) -> String {
    match tile {
        "5mr" | "5pr" | "5sr" => tile.to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

fn get_usize(event: &Value, key: &str) -> Result<usize, String> {
    event
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing or invalid usize field: {}", key))
}

fn get_i32(event: &Value, key: &str) -> Result<i32, String> {
    event
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| format!("missing or invalid i32 field: {}", key))
}

fn get_str<'a>(event: &'a Value, key: &str) -> Result<&'a str, String> {
    event
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid string field: {}", key))
}

fn remove_tile(
    hand: &mut std::collections::BTreeMap<String, u8>,
    tile: &str,
) -> Result<(), String> {
    match hand.get_mut(tile) {
        Some(count) if *count > 1 => {
            *count -= 1;
            Ok(())
        }
        Some(_) => {
            hand.remove(tile);
            Ok(())
        }
        None => Err(format!("missing tile in hand: {}", tile)),
    }
}

fn consume_wall_for_kan(state: &mut GameStateCore) {
    if let Some(remaining_wall) = state.remaining_wall.as_mut() {
        *remaining_wall = (*remaining_wall).max(1) - 1;
    }
}

fn tile_family_key(tile: &str) -> String {
    match tile {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

fn upgrade_pon_meld_to_kakan(player: &mut PlayerStateCore, pai: &str, added_tile: &str) {
    let pai_family = tile_family_key(pai);
    for meld in player.melds.iter_mut() {
        if meld.get("type").and_then(Value::as_str) != Some("pon") {
            continue;
        }
        let Some(meld_pai) = meld.get("pai").and_then(Value::as_str) else {
            continue;
        };
        if tile_family_key(meld_pai) != pai_family {
            continue;
        }

        let mut consumed = meld
            .get("consumed")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let called_tile = normalize_or_keep_aka(
            meld.get("pai_raw")
                .and_then(Value::as_str)
                .or_else(|| meld.get("pai").and_then(Value::as_str))
                .unwrap_or(pai),
        );
        consumed.push(Value::String(called_tile));
        consumed.push(Value::String(added_tile.to_string()));
        let target = meld.get("target").cloned().unwrap_or(Value::Null);
        *meld = serde_json::json!({
            "type": "kakan",
            "pai": pai,
            "pai_raw": pai,
            "consumed": consumed,
            "target": target,
        });
        return;
    }

    player.melds.push(serde_json::json!({
        "type": "kakan",
        "pai": pai,
        "pai_raw": pai,
        "consumed": [pai, pai, pai, pai],
        "target": Value::Null,
    }));
}

pub fn apply_event(state: &mut GameStateCore, event: &Value) -> Result<(), String> {
    let et = get_str(event, "type")?;
    match et {
        "start_game" => {
            state.in_game = true;
            Ok(())
        }
        "start_kyoku" => {
            state.bakaze = get_str(event, "bakaze")?.to_string();
            state.kyoku = get_i32(event, "kyoku")?;
            state.honba = get_i32(event, "honba")?;
            state.kyotaku = get_i32(event, "kyotaku")?;
            state.oya = get_usize(event, "oya")?;
            state.scores = event
                .get("scores")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing scores".to_string())?
                .iter()
                .map(|v| v.as_i64().unwrap_or(25000) as i32)
                .collect();
            state.dora_markers = vec![normalize_or_keep_aka(get_str(event, "dora_marker")?)];
            state.ura_dora_markers.clear();
            state.last_discard = None;
            state.last_kakan = None;
            state.actor_to_move = Some(state.oya);
            state.last_tsumo = vec![None, None, None, None];
            state.last_tsumo_raw = vec![None, None, None, None];
            state.remaining_wall = Some(70);
            state.pending_rinshan_actor = None;
            state.ryukyoku_tenpai_players.clear();
            state.players = vec![
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
            ];
            let tehais = event
                .get("tehais")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing tehais".to_string())?;
            for (pid, tehai) in tehais.iter().enumerate().take(4) {
                let tiles = tehai
                    .as_array()
                    .ok_or_else(|| "invalid tehai".to_string())?;
                for tile_value in tiles {
                    if let Some(tile) = tile_value.as_str() {
                        if tile != "?" {
                            let key = normalize_or_keep_aka(tile);
                            *state.players[pid].hand.entry(key).or_insert(0) += 1;
                        }
                    }
                }
            }
            Ok(())
        }
        "tsumo" => {
            let actor = get_usize(event, "actor")?;
            let pai = get_str(event, "pai")?;
            let is_rinshan = event
                .get("rinshan")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || state.pending_rinshan_actor == Some(actor);
            if pai != "?" {
                let key = normalize_or_keep_aka(pai);
                *state.players[actor].hand.entry(key.clone()).or_insert(0) += 1;
                state.last_tsumo[actor] = Some(key);
                state.last_tsumo_raw[actor] = Some(pai.to_string());
                if !is_rinshan {
                    if let Some(remaining_wall) = state.remaining_wall.as_mut() {
                        *remaining_wall = (*remaining_wall).max(1) - 1;
                    }
                }
            } else {
                state.last_tsumo[actor] = None;
                state.last_tsumo_raw[actor] = None;
            }
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.pending_rinshan_actor = None;
            state.players[actor].doujun_furiten = false;
            state.players[actor].rinshan_tsumo = is_rinshan;
            Ok(())
        }
        "dahai" => {
            let actor = get_usize(event, "actor")?;
            let pai_raw = get_str(event, "pai")?;
            if pai_raw == "?" {
                return Err("unsupported dahai pai='?' in Rust state core slice".to_string());
            }
            let tsumogiri = event
                .get("tsumogiri")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let tile_key = normalize_or_keep_aka(pai_raw);
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                remove_tile(&mut state.players[actor].hand, &tile_key)?;
            }
            let reach_declared = state.players[actor].pending_reach;
            state.players[actor].discards.push(DiscardEntry {
                pai: tile_key.clone(),
                tsumogiri,
                reach_declared,
            });
            state.last_discard = Some(LastDiscard {
                actor,
                pai: tile_key,
                pai_raw: pai_raw.to_string(),
            });
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.actor_to_move = Some((actor + 1) % 4);
            state.players[actor].ippatsu_eligible = false;
            state.players[actor].rinshan_tsumo = false;
            if state.players[actor].pending_reach {
                state.players[actor].reached = true;
                state.players[actor].pending_reach = false;
            }
            Ok(())
        }
        "pon" | "chi" | "daiminkan" => {
            let actor = get_usize(event, "actor")?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let consumed_values = event
                .get("consumed")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing consumed".to_string())?;
            let mut consumed = Vec::new();
            for value in consumed_values {
                let tile = value
                    .as_str()
                    .ok_or_else(|| "invalid consumed tile".to_string())?;
                let key = normalize_or_keep_aka(tile);
                if !event
                    .get("skip_hand_update")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    remove_tile(&mut state.players[actor].hand, &key)?;
                }
                consumed.push(Value::String(key));
            }
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let pai_raw =
                normalize_or_keep_aka(event.get("pai_raw").and_then(Value::as_str).unwrap_or(&pai));
            let target = event.get("target").cloned().unwrap_or(Value::Null);
            let meld = serde_json::json!({
                "type": et,
                "pai": pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": target,
            });
            state.players[actor].melds.push(meld);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.actor_to_move = Some(actor);
            if et == "daiminkan" {
                consume_wall_for_kan(state);
                state.pending_rinshan_actor = Some(actor);
            }
            Ok(())
        }
        "ankan" => {
            let actor = get_usize(event, "actor")?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing consumed for ankan".to_string())?
                .iter()
                .filter_map(Value::as_str)
                .map(normalize_or_keep_aka)
                .collect::<Vec<_>>();
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                for tile in &consumed {
                    remove_tile(&mut state.players[actor].hand, tile)?;
                }
            }
            let ankan_pai = event
                .get("pai")
                .and_then(Value::as_str)
                .map(normalize_or_keep_aka)
                .or_else(|| consumed.first().cloned())
                .ok_or_else(|| "missing pai/consumed for ankan".to_string())?;
            let pai_raw = normalize_or_keep_aka(
                event
                    .get("pai_raw")
                    .and_then(Value::as_str)
                    .unwrap_or(&ankan_pai),
            );
            state.players[actor].melds.push(serde_json::json!({
                "type": "ankan",
                "pai": ankan_pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": actor,
            }));
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.ryukyoku_tenpai_players.clear();
            consume_wall_for_kan(state);
            state.pending_rinshan_actor = Some(actor);
            Ok(())
        }
        "kakan" => {
            let actor = get_usize(event, "actor")?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let pai_raw =
                normalize_or_keep_aka(event.get("pai_raw").and_then(Value::as_str).unwrap_or(&pai));
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .map(|tiles| {
                    tiles
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_or_keep_aka)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let target = event.get("target").cloned().unwrap_or(Value::Null);
            state.last_kakan = Some(serde_json::json!({
                "actor": actor,
                "pai": pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": target,
            }));
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            Ok(())
        }
        "kakan_accepted" => {
            let actor = get_usize(event, "actor")?;
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .map(|tiles| {
                    tiles
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_or_keep_aka)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut added_tile = pai.clone();
            if state.players[actor]
                .hand
                .get(&added_tile)
                .copied()
                .unwrap_or(0)
                == 0
            {
                if let Some(tile) = consumed
                    .iter()
                    .find(|tile| state.players[actor].hand.get(*tile).copied().unwrap_or(0) > 0)
                {
                    added_tile = tile.clone();
                }
            }
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                remove_tile(&mut state.players[actor].hand, &added_tile)?;
            }
            upgrade_pon_meld_to_kakan(&mut state.players[actor], &pai, &added_tile);
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            consume_wall_for_kan(state);
            state.pending_rinshan_actor = Some(actor);
            Ok(())
        }
        "reach" => {
            let actor = get_usize(event, "actor")?;
            state.players[actor].pending_reach = true;
            Ok(())
        }
        "reach_accepted" => {
            let actor = get_usize(event, "actor")?;
            state.players[actor].ippatsu_eligible = true;
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                state.scores = scores
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(25000) as i32)
                    .collect();
            }
            if let Some(kyotaku) = event.get("kyotaku").and_then(Value::as_i64) {
                state.kyotaku = kyotaku as i32;
            }
            Ok(())
        }
        "hora" | "ryukyoku" => {
            state.pending_rinshan_actor = None;
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                state.scores = scores
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(25000) as i32)
                    .collect();
            }
            if let Some(honba) = event
                .get("state_honba")
                .or_else(|| event.get("honba"))
                .and_then(Value::as_i64)
            {
                state.honba = honba as i32;
            }
            if let Some(kyotaku) = event
                .get("state_kyotaku")
                .or_else(|| event.get("kyotaku"))
                .and_then(Value::as_i64)
            {
                state.kyotaku = kyotaku as i32;
            }
            if let Some(oya) = event.get("oya").and_then(Value::as_u64) {
                state.oya = oya as usize;
            }
            if let Some(ura_markers) = event.get("ura_dora_markers").and_then(Value::as_array) {
                state.ura_dora_markers = ura_markers
                    .iter()
                    .filter_map(Value::as_str)
                    .map(normalize_or_keep_aka)
                    .collect();
            }
            if let Some(tenpai_players) = event.get("tenpai_players").and_then(Value::as_array) {
                state.ryukyoku_tenpai_players = tenpai_players
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as usize)
                    .collect();
            } else if et == "hora" {
                state.ryukyoku_tenpai_players.clear();
            }
            state.actor_to_move = None;
            state.last_discard = None;
            state.last_kakan = None;
            Ok(())
        }
        "end_kyoku" | "end_game" => {
            state.actor_to_move = None;
            state.last_discard = None;
            state.last_kakan = None;
            state.pending_rinshan_actor = None;
            state.ryukyoku_tenpai_players.clear();
            Ok(())
        }
        "dora" => {
            let marker = normalize_or_keep_aka(get_str(event, "dora_marker")?);
            state.dora_markers.push(marker);
            Ok(())
        }
        _ => Err(format!(
            "unsupported event type in Rust state core slice: {}",
            et
        )),
    }
}

pub fn replay_state_snapshot_value(events: &[Value], actor: usize) -> Result<Value, String> {
    let mut state = GameStateCore::default();
    for event in events {
        apply_event(&mut state, event)?;
    }
    serde_json::to_value(crate::snapshot::snapshot_for_actor(&state, actor))
        .map_err(|err| err.to_string())
}

pub fn replay_state_snapshot(events: &[Value], actor: usize) -> Result<String, String> {
    let snapshot = replay_state_snapshot_value(events, actor)?;
    serde_json::to_string(&snapshot).map_err(|err| err.to_string())
}

pub fn validate_replay_state_snapshot(
    events: &[Value],
    actor: usize,
    expected_snapshot: Option<&Value>,
) -> Result<String, String> {
    let rust_snapshot = replay_state_snapshot_value(events, actor)?;
    let matches_expected = expected_snapshot.map(|expected| expected == &rust_snapshot);
    let payload = json!({
        "ok": matches_expected.unwrap_or(true),
        "actor": actor,
        "event_count": events.len(),
        "rust_snapshot": rust_snapshot,
        "expected_snapshot": expected_snapshot,
        "mismatch": matches_expected == Some(false),
    });
    serde_json::to_string(&payload).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::replay_state_snapshot;

    #[test]
    fn basic_start_tsumo_dahai_snapshot_works() {
        let events = vec![
            json!({"type":"start_game","names":["p0","p1","p2","p3"]}),
            json!({
                "type":"start_kyoku",
                "bakaze":"E",
                "kyoku":1,
                "honba":0,
                "kyotaku":0,
                "oya":0,
                "scores":[25000,25000,25000,25000],
                "dora_marker":"1m",
                "tehais":[
                    ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
                    ["1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m"],
                    ["2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m"],
                    ["3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m"]
                ]
            }),
            json!({"type":"tsumo","actor":0,"pai":"5p"}),
            json!({"type":"dahai","actor":0,"pai":"5p","tsumogiri":true}),
        ];
        let snapshot = replay_state_snapshot(&events, 0).unwrap();
        assert!(snapshot.contains("\"actor\":0"));
        assert!(snapshot.contains("\"last_discard\""));
        assert!(snapshot.contains("\"pai\":\"5p\""));
    }
}
