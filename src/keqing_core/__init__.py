"""Optional Rust acceleration for Keqing core helpers.

This package prefers a package-local ``keqing_core._native`` extension.
In source-development mode the Python package under ``src/`` shadows the
installed wheel package, so we also probe installed site-packages for the
compiled submodule and load it explicitly when present.
"""

from __future__ import annotations

from importlib import machinery as _importlib_machinery
from importlib import util as _importlib_util
from pathlib import Path as _Path
import site as _site
import sys as _sys
import warnings as _warnings

from riichienv import calculate_shanten as _riichienv_shanten

_USE_RUST = False
_RUST_AVAILABLE = False
_RUST_FUNC = None
_RUST_SHANTEN_NORMAL = None
_RUST_SHANTEN_ALL = None
_RUST_SHANTEN_MANY = None
_RUST_REQUIRED_TILES = None
_RUST_DRAW_DELTAS = None
_RUST_DISCARD_DELTAS = None
_RUST_SUMMARIZE_3N2_CANDIDATES = None
_RUST_SUMMARIZE_BEST_3N2_CANDIDATE = None
_RUST_IMPORT_ERROR = None


def _is_native_extension(path: _Path) -> bool:
    if not path.is_file():
        return False
    return any(str(path).endswith(suffix) for suffix in _importlib_machinery.EXTENSION_SUFFIXES)


def _candidate_native_paths() -> list[_Path]:
    package_dir = _Path(__file__).resolve().parent
    candidates = sorted(
        path for path in package_dir.glob("_native*") if _is_native_extension(path)
    )
    search_roots = []
    try:
        search_roots.extend(_site.getsitepackages())
    except AttributeError:
        pass
    try:
        search_roots.append(_site.getusersitepackages())
    except AttributeError:
        pass
    for root in search_roots:
        package_root = _Path(root) / "keqing_core"
        if not package_root.exists():
            continue
        candidates.extend(
            sorted(path for path in package_root.glob("_native*") if _is_native_extension(path))
        )
    deduped: list[_Path] = []
    seen: set[_Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_native_module():
    module_name = f"{__name__}._native"
    first_error = None
    for candidate in _candidate_native_paths():
        spec = _importlib_util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = _importlib_util.module_from_spec(spec)
        try:
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except ImportError as exc:
            _sys.modules.pop(module_name, None)
            if first_error is None:
                first_error = exc
            continue
        setattr(_sys.modules[__name__], "_native", module)
        return module, None

    return None, first_error


_rust_ext, _RUST_IMPORT_ERROR = _load_native_module()
if _rust_ext is not None and hasattr(_rust_ext, "counts34_to_ids_py"):
    _RUST_AVAILABLE = True
    _RUST_FUNC = _rust_ext.counts34_to_ids_py
    _RUST_SHANTEN_NORMAL = getattr(_rust_ext, "calc_shanten_normal_py", None)
    _RUST_SHANTEN_ALL = getattr(_rust_ext, "calc_shanten_all_py", None)
    _RUST_SHANTEN_MANY = getattr(_rust_ext, "standard_shanten_many_py", None)
    _RUST_REQUIRED_TILES = getattr(_rust_ext, "calc_required_tiles_py", None)
    _RUST_DRAW_DELTAS = getattr(_rust_ext, "calc_draw_deltas_py", None)
    _RUST_DISCARD_DELTAS = getattr(_rust_ext, "calc_discard_deltas_py", None)
    _RUST_SUMMARIZE_3N2_CANDIDATES = getattr(_rust_ext, "summarize_3n2_candidates_py", None)
    _RUST_SUMMARIZE_BEST_3N2_CANDIDATE = getattr(_rust_ext, "summarize_best_3n2_candidate_py", None)
    _USE_RUST = True


def counts34_to_ids(counts34):
    if _USE_RUST and _RUST_AVAILABLE and _RUST_FUNC is not None:
        return _rust_counts34_to_ids(counts34)
    return _python_counts34_to_ids(counts34)


def _rust_counts34_to_ids(counts34):
    return tuple(_RUST_FUNC(list(counts34)))


def _python_counts34_to_ids(counts34):
    total = sum(counts34)
    if total == 0:
        return ()
    ids = []
    for t34, cnt in enumerate(counts34):
        if cnt > 0:
            for copy_idx in range(cnt):
                ids.append(t34 * 4 + copy_idx)
    return tuple(ids)


def calc_standard_shanten(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_ALL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_ALL(list(counts34), total_tiles // 3))
    ids = counts34_to_ids(counts34)
    return int(_riichienv_shanten(ids))


def calc_shanten_normal(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_NORMAL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_NORMAL(list(counts34), total_tiles // 3))
    return calc_standard_shanten(counts34)


def calc_shanten_all(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_ALL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_ALL(list(counts34), total_tiles // 3))
    return calc_standard_shanten(counts34)


def standard_shanten_many(counts34_list):
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_MANY is not None:
        payload = [list(counts34) for counts34 in counts34_list]
        return tuple(int(v) for v in _RUST_SHANTEN_MANY(payload))
    return tuple(calc_standard_shanten(counts34) for counts34 in counts34_list)


def calc_required_tiles(counts34, visible_counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_REQUIRED_TILES is not None):
        raise RuntimeError("Rust required_tiles capability is not available")
    return tuple(
        (int(tile34), int(live_count))
        for tile34, live_count in _RUST_REQUIRED_TILES(list(counts34), list(visible_counts34), int(len_div3))
    )


def calc_draw_deltas(counts34, visible_counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_DRAW_DELTAS is not None):
        raise RuntimeError("Rust draw_deltas capability is not available")
    return tuple(
        (int(tile34), int(live_count), int(shanten_diff))
        for tile34, live_count, shanten_diff in _RUST_DRAW_DELTAS(list(counts34), list(visible_counts34), int(len_div3))
    )


def calc_discard_deltas(counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_DISCARD_DELTAS is not None):
        raise RuntimeError("Rust discard_deltas capability is not available")
    return tuple(
        (int(tile34), int(shanten_diff))
        for tile34, shanten_diff in _RUST_DISCARD_DELTAS(list(counts34), int(len_div3))
    )


def summarize_3n2_candidates(counts34, visible_counts34, summarize_fn):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_3N2_CANDIDATES is not None):
        raise RuntimeError("Rust 3n+2 candidate summaries are not available")
    return tuple(
        (
            int(discard_tile34),
            tuple(int(value) for value in after_counts34),
            int(shanten),
            int(waits_count),
            int(ukeire_type_count),
            int(ukeire_live_count),
            int(good_shape_ukeire_type_count),
            int(good_shape_ukeire_live_count),
            int(improvement_type_count),
            int(improvement_live_count),
        )
        for (
            discard_tile34,
            after_counts34,
            shanten,
            waits_count,
            ukeire_type_count,
            ukeire_live_count,
            good_shape_ukeire_type_count,
            good_shape_ukeire_live_count,
            improvement_type_count,
            improvement_live_count,
        ) in _RUST_SUMMARIZE_3N2_CANDIDATES(list(counts34), list(visible_counts34), summarize_fn)
    )


def summarize_best_3n2_candidate(counts34, visible_counts34, summarize_fn):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_BEST_3N2_CANDIDATE is not None):
        raise RuntimeError("Rust best 3n+2 candidate summary is not available")
    item = _RUST_SUMMARIZE_BEST_3N2_CANDIDATE(list(counts34), list(visible_counts34), summarize_fn)
    if item is None:
        return None
    return (
        int(item[0]),
        tuple(int(value) for value in item[1]),
        int(item[2]),
        int(item[3]),
        int(item[4]),
        int(item[5]),
        int(item[6]),
        int(item[7]),
        int(item[8]),
        int(item[9]),
    )


def is_available():
    return _RUST_AVAILABLE


def is_enabled():
    return _USE_RUST and _RUST_AVAILABLE


def has_3n2_candidate_summaries():
    return _USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_3N2_CANDIDATES is not None


def enable_rust(enable=True):
    global _USE_RUST
    if enable and not _RUST_AVAILABLE:
        detail = f": {_RUST_IMPORT_ERROR}" if _RUST_IMPORT_ERROR else ""
        _warnings.warn(
            f"Rust library not available{detail}",
            RuntimeWarning,
        )
    _USE_RUST = bool(enable) and _RUST_AVAILABLE


__all__ = [
    "counts34_to_ids",
    "calc_shanten_normal",
    "calc_standard_shanten",
    "calc_shanten_all",
    "standard_shanten_many",
    "calc_required_tiles",
    "calc_draw_deltas",
    "calc_discard_deltas",
    "summarize_3n2_candidates",
    "summarize_best_3n2_candidate",
    "is_available",
    "is_enabled",
    "has_3n2_candidate_summaries",
    "enable_rust",
]
