//! Batch helpers for progress-analysis candidate expansion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::counts::TILE_COUNT;

type CandidateProgressPy = (u8, Vec<u8>, i32, i32, i32, i32, i32, i32, i32, i32);

fn extract_counts34(seq: &Bound<'_, PyAny>, label: &str) -> PyResult<[i32; TILE_COUNT]> {
    let values: Vec<i32> = seq.extract()?;
    if values.len() != TILE_COUNT {
        return Err(PyValueError::new_err(format!(
            "{label} must contain exactly {TILE_COUNT} items, got {}",
            values.len()
        )));
    }
    let mut counts = [0i32; TILE_COUNT];
    for (idx, value) in values.into_iter().enumerate() {
        counts[idx] = value;
    }
    Ok(counts)
}

fn tile_in_obvious_meld(counts34: &[i32; TILE_COUNT], tile34: usize) -> bool {
    let cnt = counts34[tile34];
    if cnt <= 0 {
        return false;
    }
    if cnt >= 3 {
        return true;
    }
    if tile34 >= 27 {
        return false;
    }
    let pos = tile34 % 9;
    let base = tile34 - pos;
    if pos >= 2 && counts34[base + pos - 1] > 0 && counts34[base + pos - 2] > 0 {
        return true;
    }
    if (1..=7).contains(&pos) && counts34[base + pos - 1] > 0 && counts34[base + pos + 1] > 0 {
        return true;
    }
    if pos <= 6 && counts34[base + pos + 1] > 0 && counts34[base + pos + 2] > 0 {
        return true;
    }
    false
}

fn candidate_discards_no_meld_break(counts34: &[i32; TILE_COUNT]) -> Vec<usize> {
    let mut preferred = Vec::new();
    let mut fallback = Vec::new();
    let mut seen = [false; TILE_COUNT];

    for (tile34, &cnt) in counts34.iter().enumerate() {
        if cnt <= 0 || seen[tile34] {
            continue;
        }
        seen[tile34] = true;
        fallback.push(tile34);
        if !tile_in_obvious_meld(counts34, tile34) {
            preferred.push(tile34);
        }
    }

    if preferred.is_empty() {
        fallback
    } else {
        preferred
    }
}

pub fn summarize_3n2_candidates_py_impl(
    py: Python<'_>,
    counts34_seq: &Bound<'_, PyAny>,
    visible_counts34_seq: &Bound<'_, PyAny>,
    summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<CandidateProgressPy>> {
    let counts34 = extract_counts34(counts34_seq, "counts34")?;
    let visible_counts34 = extract_counts34(visible_counts34_seq, "visible_counts34")?;
    let visible_counts_py = PyTuple::new_bound(py, visible_counts34.iter().copied());
    let mut out = Vec::new();

    for discard34 in candidate_discards_no_meld_break(&counts34) {
        let mut after_counts34 = counts34;
        after_counts34[discard34] -= 1;
        let after_counts_py = PyTuple::new_bound(py, after_counts34.iter().copied());
        let summary = summarize_fn.call1((after_counts_py, visible_counts_py.clone()))?;

        out.push((
            discard34 as u8,
            after_counts34.iter().map(|&value| value as u8).collect(),
            summary.getattr("shanten")?.extract::<i32>()?,
            summary.getattr("waits_count")?.extract::<i32>()?,
            summary.getattr("ukeire_type_count")?.extract::<i32>()?,
            summary.getattr("ukeire_live_count")?.extract::<i32>()?,
            summary
                .getattr("good_shape_ukeire_type_count")?
                .extract::<i32>()?,
            summary
                .getattr("good_shape_ukeire_live_count")?
                .extract::<i32>()?,
            summary.getattr("improvement_type_count")?.extract::<i32>()?,
            summary.getattr("improvement_live_count")?.extract::<i32>()?,
        ));
    }

    Ok(out)
}
