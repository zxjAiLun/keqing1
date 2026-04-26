"""Xmodel1 package."""

from __future__ import annotations

from importlib import import_module

from xmodel1.schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)

__all__ = [
    "XMODEL1_SCHEMA_NAME",
    "XMODEL1_SCHEMA_VERSION",
    "XMODEL1_MAX_CANDIDATES",
    "XMODEL1_CANDIDATE_FEATURE_DIM",
    "XMODEL1_CANDIDATE_FLAG_DIM",
    "Xmodel1Model",
    "Xmodel1Adapter",
]


def __getattr__(name: str):
    if name == "Xmodel1Model":
        value = getattr(import_module("xmodel1.model"), name)
    elif name == "Xmodel1Adapter":
        value = getattr(import_module("xmodel1.adapter"), name)
    else:
        raise AttributeError(f"module 'xmodel1' has no attribute {name!r}")
    globals()[name] = value
    return value
