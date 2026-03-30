from v5model.features import encode, C_TILE, N_SCALAR
from v5model.action_space import action_to_idx, build_legal_mask, ACTION_SPACE


def __getattr__(name):
    if name == "MahjongModel":
        from v5model.model import MahjongModel
        return MahjongModel
    raise AttributeError(f"module 'v5model' has no attribute {name!r}")


__all__ = ["MahjongModel", "encode", "C_TILE", "N_SCALAR", "action_to_idx", "build_legal_mask", "ACTION_SPACE"]
