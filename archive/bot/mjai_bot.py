from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from bot.rule_bot import fallback_action
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState, apply_event
from model.vocab import action_to_token
from bot.features import OBS_DIM, vectorize_state_py


class MjaiPolicyBot:
    def __init__(self, player_id: int, checkpoint_path: str = "checkpoints/best.npz") -> None:
        self.player_id = player_id
        self.state = GameState()
        self.action_vocab, self.stoi, self.model = self._load_checkpoint(
            checkpoint_path
        )
        # Optional oracle-state features for shanten/waits_count.
        # In mjai Docker environment `riichi` might not be installed, so we
        # must gracefully fall back to zeros.
        self._riichi_states = None
        try:
            import riichi  # type: ignore

            self._riichi_states = [riichi.state.PlayerState(pid) for pid in range(4)]
        except Exception:
            self._riichi_states = None
        self._debug_policy = os.getenv("BOT_DEBUG_POLICY", "0") == "1"

    @staticmethod
    def _load_checkpoint(
        checkpoint_path: str,
    ) -> Tuple[List[str], Dict[str, int], "_NumpylessPolicyValueModel"]:
        """
        Prefer loading `best.json` to avoid numpy dependency inside mjai Docker.
        """
        ckpt_json_path = checkpoint_path
        if ckpt_json_path.endswith(".npz"):
            ckpt_json_path = ckpt_json_path[:-4] + ".json"

        if os.path.exists(ckpt_json_path):
            ckpt = json.load(open(ckpt_json_path, "r", encoding="utf-8"))
            action_vocab = ckpt["action_vocab"]
            stoi = {a: i for i, a in enumerate(action_vocab)}
            model = _NumpylessPolicyValueModel.from_state_dict_json(ckpt["model_state_dict"])
            return action_vocab, stoi, model

        # Local fallback: allow numpy-based npz loading (useful for dev).
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"checkpoint requires numpy but failed to import numpy: {e}. "
                f"Please export {ckpt_json_path}."
            )

        ckpt_npz = np.load(checkpoint_path, allow_pickle=True)
        ckpt = {
            k: ckpt_npz[k].item() if ckpt_npz[k].dtype == object else ckpt_npz[k]
            for k in ckpt_npz.files
        }
        action_vocab = ckpt["action_vocab"]
        stoi = {a: i for i, a in enumerate(action_vocab)}

        model_state_dict = ckpt["model_state_dict"]
        has_encoder_W5 = "W5" in model_state_dict and model_state_dict["W5"].ndim == 2 and model_state_dict["W5"].shape[0] == 256

        if has_encoder_W5:
            model_state_dict_json = _export_full_state_dict(model_state_dict)
            model = _NumpylessMultiTaskModel.from_state_dict_json(model_state_dict_json)
        else:
            model_state_dict_json = {
                "W1": model_state_dict["W1"].tolist(),
                "b1": model_state_dict["b1"].tolist(),
                "Wp": model_state_dict["Wp"].tolist(),
                "bp": model_state_dict["bp"].tolist(),
                "Wv": model_state_dict["Wv"].tolist(),
                "bv": float(model_state_dict["bv"]),
            }
            model = _NumpylessPolicyValueModel.from_state_dict_json(model_state_dict_json)
        return action_vocab, stoi, model

    def _choose_action(self, legal_actions: List[Dict], snap: Dict) -> Dict:
        action, _ = self._choose_action_with_meta(legal_actions, snap, topk=5)
        return action

    def _choose_action_with_meta(self, legal_actions: List[Dict], snap: Dict, topk: int = 5) -> Tuple[Dict, Dict]:
        if not legal_actions:
            return {"type": "none"}, {"route": "empty_legal", "topk": [], "vocab_miss": 0}

        obs = vectorize_state_py(snap, self.player_id)
        logits = self.model.forward_logits(obs)

        def _token_and_logit(a: Dict) -> Tuple[str, Optional[int], Optional[float]]:
            tok = action_to_token(a)
            idx = self.stoi.get(tok)
            return tok, idx, (logits[idx] if idx is not None else None)

        scored: List[Dict] = []
        vocab_miss = 0
        for a in legal_actions:
            tok, idx, logit = _token_and_logit(a)
            if idx is None:
                vocab_miss += 1
            scored.append(
                {
                    "action": a,
                    "token": tok,
                    "vocab_hit": idx is not None,
                    "idx": idx,
                    "logit": logit,
                }
            )

        valid_scored = [x for x in scored if x["logit"] is not None]
        probs_by_idx: Dict[int, float] = {}
        if valid_scored:
            idxs = [int(x["idx"]) for x in valid_scored]
            vals = [float(x["logit"]) for x in valid_scored]
            max_v = max(vals)
            exps = [math.exp(v - max_v) for v in vals]
            s = sum(exps) or 1.0
            for i, e in zip(idxs, exps):
                probs_by_idx[i] = e / s

        non_none_actions = [a for a in legal_actions if a.get("type") != "none"]
        candidates = non_none_actions if non_none_actions else legal_actions

        best_a: Optional[Dict] = None
        best_logit = float("-inf")
        for a in candidates:
            tok = action_to_token(a)
            idx = self.stoi.get(tok)
            if idx is None:
                continue
            v = logits[idx]
            if v > best_logit:
                best_logit = v
                best_a = a

        if best_a is not None:
            topk_items = sorted(
                valid_scored,
                key=lambda x: float(x["logit"]),
                reverse=True,
            )[: max(topk, 1)]
            topk_payload = [
                {
                    "token": str(x["token"]),
                    "action": x["action"],
                    "logit": float(x["logit"]),
                    "prob": float(probs_by_idx.get(int(x["idx"]), 0.0)),
                }
                for x in topk_items
            ]
            meta = {
                "route": "model_choice",
                "topk": topk_payload,
                "vocab_miss": vocab_miss,
            }
            return best_a, meta
        fb = fallback_action(legal_actions, self.player_id)
        topk_items = sorted(
            valid_scored,
            key=lambda x: float(x["logit"]),
            reverse=True,
        )[: max(topk, 1)]
        topk_payload = [
            {
                "token": str(x["token"]),
                "action": x["action"],
                "logit": float(x["logit"]),
                "prob": float(probs_by_idx.get(int(x["idx"]), 0.0)),
            }
            for x in topk_items
        ]
        meta = {
            "route": "fallback",
            "topk": topk_payload,
            "vocab_miss": vocab_miss,
        }
        return fb, meta

    def react(self, event_json: str) -> str:
        events = json.loads(event_json)
        # mjai server sends a JSON array of events in one request.
        if isinstance(events, dict):
            events = [events]
        for ev in events:
            apply_event(self.state, ev)
            if self._riichi_states is not None:
                # Update oracle state for all player perspectives.
                payload = json.dumps(ev, ensure_ascii=False)
                # Simulator sends hidden-info events containing "?" for other players.
                # libriichi PlayerState may panic on such partial-observation payloads.
                # In that case we gracefully disable oracle features for this hand.
                if "?" not in payload:
                    try:
                        for ps in self._riichi_states:
                            ps.update(payload)
                    except BaseException:
                        self._riichi_states = None
        if self.state.actor_to_move == self.player_id:
            snap = self.state.snapshot(self.player_id)
            if self._riichi_states is not None:
                ps = self._riichi_states[self.player_id]
                snap["shanten"] = int(ps.shanten)
                snap["waits_count"] = int(sum(ps.waits))
            legal = [a.to_mjai() for a in enumerate_legal_actions(snap, self.player_id)]
            action, meta = self._choose_action_with_meta(legal, snap, topk=5)
            # mjai protocol: "none" must not include "actor".
            if action.get("type") != "none" and "actor" not in action:
                action["actor"] = self.player_id
            if self._debug_policy:
                debug_payload = {
                    "type": "policy_debug",
                    "actor": self.player_id,
                    "route": meta.get("route"),
                    "chosen": action,
                    "topk": meta.get("topk", []),
                    "vocab_miss": meta.get("vocab_miss", 0),
                }
                print(json.dumps(debug_payload, ensure_ascii=False), flush=True)
            return json.dumps(action, ensure_ascii=False)
        return json.dumps({"type": "none"}, ensure_ascii=False)


class _NumpylessPolicyValueModel:
    """
    A tiny 1-hidden-layer MLP implemented in pure python.

    Only policy logits are needed for action selection.
    """

    def __init__(
        self,
        W1: List[List[float]],
        b1: List[float],
        Wp: List[List[float]],
        bp: List[float],
        Wv: List[float],
        bv: float,
    ) -> None:
        self.W1 = W1
        self.b1 = b1
        self.Wp = Wp
        self.bp = bp
        self.Wv = Wv
        self.bv = bv
        self.in_dim = len(W1)
        self.hidden_dim = len(W1[0]) if W1 else 0
        self.action_dim = len(Wp[0]) if Wp else 0

    @classmethod
    def from_state_dict_json(cls, state_dict: Dict) -> "_NumpylessPolicyValueModel":
        return cls(
            W1=state_dict["W1"],
            b1=state_dict["b1"],
            Wp=state_dict["Wp"],
            bp=state_dict["bp"],
            Wv=state_dict["Wv"],
            bv=_to_scalar(state_dict["bv"]),
        )

    def forward_logits(self, obs: List[float]) -> List[float]:
        # h_lin[k] = sum_j obs[j] * W1[j][k] + b1[k]
        h = [0.0] * self.hidden_dim
        for k in range(self.hidden_dim):
            s = self.b1[k]
            for j in range(self.in_dim):
                s += obs[j] * self.W1[j][k]
            # ReLU
            if s < 0.0:
                s = 0.0
            h[k] = s

        # logits[a] = sum_k h[k] * Wp[k][a] + bp[a]
        logits = [0.0] * self.action_dim
        for a in range(self.action_dim):
            s = self.bp[a]
            for k in range(self.hidden_dim):
                s += h[k] * self.Wp[k][a]
            logits[a] = s
        return logits


def _to_scalar(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return float(v[0])
    if hasattr(v, "tolist"):
        arr = v.tolist()
        if isinstance(arr, list) and len(arr) == 1:
            return float(arr[0])
        return arr
    return float(v)


def _export_full_state_dict(model_state_dict: Dict) -> Dict:
    result = {}
    for key, arr in model_state_dict.items():
        if hasattr(arr, "tolist"):
            result[key] = arr.tolist()
        elif isinstance(arr, (int, float)):
            result[key] = arr
        elif arr is None:
            result[key] = None
        else:
            result[key] = arr
    return result


class _NumpylessMultiTaskModel:
    """
    Inference-only ResNetEncoder + PolicyHead in pure Python (no numpy required).
    Supports both v1 (1-hidden-layer) and v2 (2-resblock) encoders via presence of W4/W5 keys.
    Only policy logits are needed for action selection.
    """

    def __init__(
        self,
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        W3: List[List[float]],
        b3: List[float],
        W4: Optional[List[List[float]]],
        b4: Optional[List[float]],
        W5: Optional[List[List[float]]],
        b5: Optional[List[float]],
        Wp: List[List[float]],
        bp: List[float],
        Wv: List[List[float]],
        bv: float,
        Waux: Optional[List[List[float]]],
        baux: Optional[List[float]],
    ) -> None:
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.W4 = W4
        self.b4 = b4
        self.W5 = W5
        self.b5 = b5
        self.Wp = Wp
        self.bp = bp
        self.Wv = Wv
        self.bv = bv
        self.Waux = Waux
        self.baux = baux
        self.in_dim = len(W1)
        self.hidden_dim = len(W1[0]) if W1 else 0
        self.action_dim = len(Wp[0]) if Wp else 0
        self._is_v2 = W4 is not None and W5 is not None

    @classmethod
    def from_state_dict_json(cls, state_dict: Dict) -> "_NumpylessMultiTaskModel":
        return cls(
            W1=state_dict["W1"],
            b1=state_dict["b1"],
            W2=state_dict.get("W2", state_dict["W1"]),
            b2=state_dict.get("b2", state_dict["b1"]),
            W3=state_dict.get("W3", state_dict["W1"]),
            b3=state_dict.get("b3", state_dict["b1"]),
            W4=state_dict.get("W4"),
            b4=state_dict.get("b4"),
            W5=state_dict.get("W5"),
            b5=state_dict.get("b5"),
            Wp=state_dict["Wp"],
            bp=state_dict["bp"],
            Wv=state_dict["Wv"],
            bv=_to_scalar(state_dict["bv"]),
            Waux=state_dict.get("Waux"),
            baux=state_dict.get("baux"),
        )

    def _relu(self, x: float) -> float:
        return x if x > 0.0 else 0.0

    def forward_logits(self, obs: List[float]) -> List[float]:
        hd = self.hidden_dim

        h = [0.0] * hd
        for k in range(hd):
            s = self.b1[k]
            for j in range(self.in_dim):
                s += obs[j] * self.W1[j][k]
            h[k] = self._relu(s)

        if self._is_v2:
            h2 = [0.0] * hd
            for k in range(hd):
                s = self.b2[k]
                for j in range(hd):
                    s += h[j] * self.W2[j][k]
                h2[k] = self._relu(s)
            h3 = [0.0] * hd
            for k in range(hd):
                s = self.b3[k]
                for j in range(hd):
                    s += h2[j] * self.W3[j][k]
                h[k] = self._relu(s + h[k])

            h4 = [0.0] * hd
            for k in range(hd):
                s = self.b4[k]
                for j in range(hd):
                    s += h[j] * self.W4[j][k]
                h4[k] = self._relu(s)
            h5 = [0.0] * hd
            for k in range(hd):
                s = self.b5[k]
                for j in range(hd):
                    s += h4[j] * self.W5[j][k]
                h[k] = self._relu(s + h[k])
        else:
            for k in range(hd):
                s = self.b2[k]
                for j in range(hd):
                    s += h[j] * self.W2[j][k]
                h[k] = self._relu(s + h[k])

        logits = [0.0] * self.action_dim
        for a in range(self.action_dim):
            s = self.bp[a]
            for k in range(hd):
                s += h[k] * self.Wp[k][a]
            logits[a] = s
        return logits

