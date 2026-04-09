//! Xmodel1 schema constants.
//!
//! This module defines the Rust-side constants for the candidate-centric Xmodel1
//! cache protocol. The first implementation phase keeps the module deliberately
//! small so the Python and Rust layers can converge on the same field contract
//! before the export logic is filled in.

pub const XMODEL1_SCHEMA_NAME: &str = "xmodel1_discard_v1";
pub const XMODEL1_SCHEMA_VERSION: u32 = 1;
pub const XMODEL1_MAX_CANDIDATES: usize = 14;
pub const XMODEL1_CANDIDATE_FEATURE_DIM: usize = 21;
pub const XMODEL1_CANDIDATE_FLAG_DIM: usize = 10;

pub fn validate_candidate_mask_and_choice(
    chosen_candidate_idx: i16,
    candidate_mask: &[u8],
    candidate_tile_id: &[i16],
) -> Result<(), String> {
    if candidate_mask.len() != XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "candidate_mask length {} != {}",
            candidate_mask.len(),
            XMODEL1_MAX_CANDIDATES
        ));
    }
    if candidate_tile_id.len() != XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "candidate_tile_id length {} != {}",
            candidate_tile_id.len(),
            XMODEL1_MAX_CANDIDATES
        ));
    }
    let idx = usize::try_from(chosen_candidate_idx)
        .map_err(|_| format!("chosen_candidate_idx {} is negative", chosen_candidate_idx))?;
    if idx >= XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "chosen_candidate_idx {} out of range for {} candidates",
            chosen_candidate_idx,
            XMODEL1_MAX_CANDIDATES
        ));
    }
    if candidate_mask[idx] == 0 {
        return Err(format!(
            "chosen_candidate_idx {} points to masked candidate",
            chosen_candidate_idx
        ));
    }
    for i in 0..XMODEL1_MAX_CANDIDATES {
        let masked = candidate_mask[i] == 0;
        if masked && candidate_tile_id[i] != -1 {
            return Err(format!(
                "padding candidate {} must use tile_id=-1, got {}",
                i, candidate_tile_id[i]
            ));
        }
        if !masked && !(0..=33).contains(&candidate_tile_id[i]) {
            return Err(format!(
                "active candidate {} has invalid tile_id {}",
                i, candidate_tile_id[i]
            ));
        }
    }
    Ok(())
}
