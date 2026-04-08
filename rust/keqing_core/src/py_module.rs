//! PyO3 Python bindings for keqing_core._native
//!
//! This module provides Python bindings for the keqing_core Rust library.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::counts::TILE_COUNT;
use crate::progress_batch::summarize_3n2_candidates_py_impl;
use crate::shanten_table::{calc_shanten_all, calc_shanten_normal, ensure_init};
use crate::standard::counts34_to_ids;

#[pyfunction]
fn counts34_to_ids_py(counts34: &Bound<'_, PyList>) -> PyResult<Vec<u16>> {
    let mut counts = [0i32; TILE_COUNT];
    
    for (i, item) in counts34.iter().enumerate() {
        if i >= TILE_COUNT {
            break;
        }
        counts[i] = item.extract::<i32>()?;
    }
    
    Ok(counts34_to_ids(&counts))
}

fn extract_counts34_array(seq: &Bound<'_, PyList>) -> PyResult<[u8; TILE_COUNT]> {
    let mut counts = [0u8; TILE_COUNT];
    for (i, item) in seq.iter().enumerate() {
        if i >= TILE_COUNT {
            break;
        }
        counts[i] = item.extract::<u8>()?;
    }
    Ok(counts)
}

#[pyfunction]
fn calc_shanten_normal_py(counts34: &Bound<'_, PyList>, len_div3: u8) -> PyResult<i32> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    Ok(calc_shanten_normal(&counts, len_div3) as i32)
}

#[pyfunction]
fn calc_shanten_all_py(counts34: &Bound<'_, PyList>, len_div3: u8) -> PyResult<i32> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    Ok(calc_shanten_all(&counts, len_div3) as i32)
}

#[pyfunction]
fn standard_shanten_many_py(_py: Python<'_>, counts_list: &Bound<'_, PyList>) -> PyResult<Vec<i32>> {
    let mut out = Vec::with_capacity(counts_list.len());
    ensure_init();

    for item in counts_list.iter() {
        let counts34 = item.downcast::<PyList>()?;
        let counts = extract_counts34_array(&counts34)?;
        let total_tiles: u8 = counts.iter().sum();
        out.push(calc_shanten_all(&counts, total_tiles / 3) as i32);
    }

    Ok(out)
}

#[pyfunction]
fn summarize_3n2_candidates_py(
    py: Python<'_>,
    counts34: &Bound<'_, PyAny>,
    visible_counts34: &Bound<'_, PyAny>,
    summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u8, Vec<u8>, i32, i32, i32, i32, i32, i32, i32, i32)>> {
    summarize_3n2_candidates_py_impl(py, counts34, visible_counts34, summarize_fn)
}

#[pymodule]
pub fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    ensure_init();
    m.add_function(wrap_pyfunction!(counts34_to_ids_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_shanten_normal_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_shanten_all_py, m)?)?;
    m.add_function(wrap_pyfunction!(standard_shanten_many_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_3n2_candidates_py, m)?)?;
    m.add("TILE_COUNT", TILE_COUNT)?;
    Ok(())
}
