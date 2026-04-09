"""Xmodel1 package.

New candidate-centric small-model line focused on offensive tile-efficiency
correctness. This package intentionally starts as a scaffold so the data/schema
and model/training boundaries can be implemented incrementally.
"""

from xmodel1.schema import (
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
)
from xmodel1.model import Xmodel1Model
from xmodel1.adapter import Xmodel1Adapter

__all__ = [
    "XMODEL1_SCHEMA_NAME",
    "XMODEL1_SCHEMA_VERSION",
    "XMODEL1_MAX_CANDIDATES",
    "XMODEL1_CANDIDATE_FEATURE_DIM",
    "XMODEL1_CANDIDATE_FLAG_DIM",
    "Xmodel1Model",
    "Xmodel1Adapter",
]
