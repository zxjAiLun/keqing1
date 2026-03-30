from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl
from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from model.policy_value import PolicyValueModel
from train.dataset import OBS_DIM, SupervisedMjaiDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="log.jsonl")
    parser.add_argument("--raw-json", default=None, help="Raw tenhou/majsoul json. If set, convert first.")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--converted-out", default="artifacts/converted/eval_log.jsonl")
    parser.add_argument("--converted-dir", default="artifacts/converted/eval")
    parser.add_argument("--libriichi-bin", default=None)
    parser.add_argument("--view-mode", choices=["all", "single", "me"], default="all")
    parser.add_argument("--view-name", default="keqing1")
    parser.add_argument("--checkpoint", default="artifacts/sl/best.npz")
    args = parser.parse_args()

    ckpt_npz = np.load(args.checkpoint, allow_pickle=True)
    ckpt = {k: ckpt_npz[k].item() if ckpt_npz[k].dtype == object else ckpt_npz[k] for k in ckpt_npz.files}
    actions = ckpt["action_vocab"]
    stoi = {a: i for i, a in enumerate(actions)}
    model = PolicyValueModel(in_dim=OBS_DIM, hidden_dim=ckpt["config"]["hidden_dim"], action_dim=len(actions))
    model.load_state_dict(ckpt["model_state_dict"])

    log_paths = [args.log_path]
    if args.raw_json:
        convert_raw_to_mjai(args.raw_json, args.converted_out, args.libriichi_bin)
        errors = validate_mjai_jsonl(args.converted_out)
        if errors:
            raise RuntimeError(f"converted mjai validation failed: {errors}")
        log_paths = [args.converted_out]
    elif args.raw_dir:
        converted_dir = Path(args.converted_dir)
        converted_dir.mkdir(parents=True, exist_ok=True)
        log_paths = []
        raw_dir_path = Path(args.raw_dir)
        if raw_dir_path.is_file() and raw_dir_path.suffix == ".json":
            src_files = [raw_dir_path]
        else:
            src_files = sorted(raw_dir_path.glob("*.json"))
        for src in src_files:
            out = converted_dir / f"{src.stem}.jsonl"
            convert_raw_to_mjai(str(src), str(out), args.libriichi_bin)
            errors = validate_mjai_jsonl(out)
            if errors:
                raise RuntimeError(f"converted mjai validation failed for {src.name}: {errors}")
            log_paths.append(str(out))
    if args.view_mode == "single":
        name_filter = {args.view_name}
    elif args.view_mode == "me":
        name_filter = {"私"}
    else:
        name_filter = None
    samples = []
    for p in log_paths:
        samples.extend(build_supervised_samples(read_mjai_jsonl(p), actor_name_filter=name_filter))
    ds = SupervisedMjaiDataset(samples, stoi)
    n = len(ds)
    correct = 0
    illegal = 0
    value_mse = 0.0
    for i in range(n):
        obs, mask, label, _ = ds[i]
        logits, value = model.forward(obs[None, :])
        logits = model.masked_logits(logits, mask[None, :])
        pred = int(np.argmax(logits, axis=-1)[0])
        if mask[pred] <= 0:
            illegal += 1
        if pred == int(label):
            correct += 1
        value_mse += float((value[0] - ds[i][3]) ** 2)
    out = {
        "num_samples": n,
        "top1_acc": correct / max(n, 1),
        "illegal_action_rate": illegal / max(n, 1),
        "value_mse": value_mse / max(n, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

