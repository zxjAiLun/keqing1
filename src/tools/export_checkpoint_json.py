#!/usr/bin/env python3
"""
导出检查点为 JSON 格式，供 mjai Docker 使用

支持 v1 (1-hidden-layer MLP) 和 v2 (ResNetEncoder + MultiTask) 模型
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _to_jsonable(x: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars into JSON-serializable objects.
    """
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            return x.tolist()
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
    except ImportError:
        pass

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _npz_to_state_dict_json_v1(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    v1: 1-hidden-layer MLP
    """
    return {
        "W1": _to_jsonable(model_state_dict["W1"]),
        "b1": _to_jsonable(model_state_dict["b1"]),
        "Wp": _to_jsonable(model_state_dict["Wp"]),
        "bp": _to_jsonable(model_state_dict["bp"]),
        "Wv": _to_jsonable(model_state_dict["Wv"]),
        "bv": float(model_state_dict["bv"]),
    }


def _npz_to_state_dict_json_v2(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    v2: ResNetEncoder (W1-W5) + PolicyHead + ValueHead + AuxHead
    """
    return {
        # ResNet Encoder
        "W1": _to_jsonable(model_state_dict["W1"]),
        "b1": _to_jsonable(model_state_dict["b1"]),
        "W2": _to_jsonable(model_state_dict["W2"]),
        "b2": _to_jsonable(model_state_dict["b2"]),
        "W3": _to_jsonable(model_state_dict["W3"]),
        "b3": _to_jsonable(model_state_dict["b3"]),
        "W4": _to_jsonable(model_state_dict["W4"]),
        "b4": _to_jsonable(model_state_dict["b4"]),
        "W5": _to_jsonable(model_state_dict["W5"]),
        "b5": _to_jsonable(model_state_dict["b5"]),
        # Policy Head
        "Wp": _to_jsonable(model_state_dict["Wp"]),
        "bp": _to_jsonable(model_state_dict["bp"]),
        # Value Head
        "Wv": _to_jsonable(model_state_dict["Wv"]),
        "bv": _to_jsonable(model_state_dict["bv"]),
        # Aux Head (optional)
        "Waux": _to_jsonable(model_state_dict.get("Waux")),
        "baux": _to_jsonable(model_state_dict.get("baux")),
    }


def detect_model_version(model_state_dict: Dict[str, Any]) -> str:
    """检测模型版本"""
    if "W4" in model_state_dict and "W5" in model_state_dict:
        # v2: ResNetEncoder
        return "v2"
    return "v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.npz (created by train_v2.py or train_sl.py).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for best.json to be bundled into mjai Docker.",
    )
    parser.add_argument(
        "--force-v1",
        action="store_true",
        help="Force v1 export even if W4/W5 are present.",
    )
    args = parser.parse_args()

    import numpy as np

    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_npz = np.load(checkpoint_path, allow_pickle=True)
    ckpt = {
        k: ckpt_npz[k].item() if ckpt_npz[k].dtype == object else ckpt_npz[k]
        for k in ckpt_npz.files
    }

    model_state_dict = ckpt["model_state_dict"]

    if args.force_v1:
        model_state_dict_json = _npz_to_state_dict_json_v1(model_state_dict)
        version = "v1 (forced)"
    else:
        detected = detect_model_version(model_state_dict)
        if detected == "v2":
            model_state_dict_json = _npz_to_state_dict_json_v2(model_state_dict)
            version = "v2"
        else:
            model_state_dict_json = _npz_to_state_dict_json_v1(model_state_dict)
            version = "v1"

    exported = {
        "action_vocab": _to_jsonable(ckpt.get("action_vocab", [])),
        "config": _to_jsonable(ckpt.get("config", {})),
        "model_state_dict": model_state_dict_json,
        "_export_version": version,
    }

    out_path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {version} model to {out_path}")


if __name__ == "__main__":
    main()
