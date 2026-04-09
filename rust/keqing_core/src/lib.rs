//! Keqing Core - Mahjong AI computation library
//!
//! This crate provides optimized Rust implementations of hot-path
//! computation functions for the Keqing Mahjong AI system.

pub mod counts;
pub mod progress_delta;
pub mod progress_batch;
pub mod progress_summary;
pub mod scoring_pool;
pub mod shanten_table;
pub mod standard;
pub mod shanten;
pub mod xmodel1_export;
pub mod xmodel1_schema;
pub mod py_module;

pub use counts::{Counts34, TILE_COUNT};
pub use progress_delta::{DiscardDelta, DrawDelta, RequiredTile, calc_discard_deltas, calc_draw_deltas, calc_required_tiles};
pub use progress_summary::{Summary3n1, summarize_3n1};
pub use shanten_table::{calc_shanten_all, calc_shanten_normal};
pub use standard::counts34_to_ids;
pub use xmodel1_schema::{
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
};
