from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", default="artifacts/sl/best.npz")
    parser.add_argument("--out-dir", default="artifacts/rl")
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_npz = np.load(args.init_checkpoint, allow_pickle=True)
    ckpt = {k: ckpt_npz[k].item() if ckpt_npz[k].dtype == object else ckpt_npz[k] for k in ckpt_npz.files}
    ckpt["rl_meta"] = {
        "algorithm": "lightweight_policy_finetune",
        "kl_coef": args.kl_coef,
        "steps": args.steps,
        "note": "MVP placeholder: plug self-play trajectories into policy update loop here.",
    }
    np.savez(out_dir / "rl_finetuned.npz", **ckpt)
    print(json.dumps({"saved": str(out_dir / "rl_finetuned.npz")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

