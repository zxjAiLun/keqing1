//! Keqing Core - Mahjong AI computation library
//!
//! This crate provides optimized Rust implementations of hot-path
//! computation functions for the Keqing Mahjong AI system.

pub mod counts;
pub mod progress_batch;
pub mod standard;
pub mod shanten;
pub mod py_module;

pub use counts::{Counts34, TILE_COUNT};
pub use standard::counts34_to_ids;
