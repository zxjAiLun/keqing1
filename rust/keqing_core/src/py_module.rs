//! PyO3 Python bindings for keqing_core._native
//!
//! This module provides Python bindings for the keqing_core Rust library.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::counts::TILE_COUNT;
use crate::progress_batch::summarize_3n2_candidates_py_impl;
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

#[pyfunction]
fn standard_shanten_many_py(py: Python<'_>, counts_list: &Bound<'_, PyList>) -> PyResult<Vec<i32>> {
    let riichienv = py.import_bound("riichienv")?;
    let calculate_shanten = riichienv.getattr("calculate_shanten")?;
    let mut out = Vec::with_capacity(counts_list.len());

    for item in counts_list.iter() {
        let counts34 = item.downcast::<PyList>()?;
        let mut counts = [0i32; TILE_COUNT];
        for (i, value) in counts34.iter().enumerate() {
            if i >= TILE_COUNT {
                break;
            }
            let extracted: i32 = value.extract()?;
            counts[i] = extracted;
        }
        let ids = counts34_to_ids(&counts);
        let shanten = calculate_shanten.call1((ids,))?.extract::<i32>()?;
        out.push(shanten);
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
    m.add_function(wrap_pyfunction!(counts34_to_ids_py, m)?)?;
    m.add_function(wrap_pyfunction!(standard_shanten_many_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_3n2_candidates_py, m)?)?;
    m.add("TILE_COUNT", TILE_COUNT)?;
    Ok(())
}
