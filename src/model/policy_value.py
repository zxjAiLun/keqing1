from __future__ import annotations

import numpy as np


class ResNetEncoder:
    def __init__(self, obs_dim: int = 270, hidden_dim: int = 256) -> None:
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.W1 = np.random.randn(obs_dim, hidden_dim).astype(np.float32) * 0.02
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)

        self.W4 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.b4 = np.zeros(hidden_dim, dtype=np.float32)
        self.W5 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.b5 = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        h = obs @ self.W1 + self.b1
        h = np.maximum(h, 0.0)

        h_shortcut = h
        h = h @ self.W2 + self.b2
        h = np.maximum(h, 0.0)
        h = h @ self.W3 + self.b3
        h = h + h_shortcut
        h = np.maximum(h, 0.0)

        h_shortcut = h
        h = h @ self.W4 + self.b4
        h = np.maximum(h, 0.0)
        h = h @ self.W5 + self.b5
        h = h + h_shortcut
        h = np.maximum(h, 0.0)

        return h

    def state_dict(self) -> dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3,
            "W4": self.W4, "b4": self.b4, "W5": self.W5, "b5": self.b5,
        }

    def load_state_dict(self, state: dict) -> None:
        self.W1 = state["W1"]
        self.b1 = state["b1"]
        self.W2 = state["W2"]
        self.b2 = state["b2"]
        self.W3 = state["W3"]
        self.b3 = state["b3"]
        self.W4 = state["W4"]
        self.b4 = state["b4"]
        self.W5 = state["W5"]
        self.b5 = state["b5"]


class MultiTaskModel:
    def __init__(self, obs_dim: int = 270, hidden_dim: int = 256, action_dim: int = 5, num_ranks: int = 5) -> None:
        self.encoder = ResNetEncoder(obs_dim, hidden_dim)

        self.Wp = np.random.randn(hidden_dim, action_dim).astype(np.float32) * 0.02
        self.bp = np.zeros(action_dim, dtype=np.float32)

        self.Wv = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.02
        self.bv = np.zeros(1, dtype=np.float32)

        self.Waux = np.random.randn(hidden_dim, num_ranks).astype(np.float32) * 0.02
        self.baux = np.zeros(num_ranks, dtype=np.float32)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.encoder.forward(obs)
        logits = h @ self.Wp + self.bp
        value = h @ self.Wv + self.bv
        aux_logits = h @ self.Waux + self.baux
        return logits, value, aux_logits

    def forward_with_cache(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = self.encoder.forward(obs)
        logits = h @ self.Wp + self.bp
        value = h @ self.Wv + self.bv
        aux_logits = h @ self.Waux + self.baux
        return logits, value, aux_logits, h

    @staticmethod
    def masked_logits(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        out = logits.copy()
        out[legal_mask <= 0] = -1e9
        return out

    def state_dict(self) -> dict:
        state = self.encoder.state_dict()
        state.update({
            "Wp": self.Wp, "bp": self.bp,
            "Wv": self.Wv, "bv": self.bv,
            "Waux": self.Waux, "baux": self.baux,
        })
        return state

    def load_state_dict(self, state: dict) -> None:
        encoder_keys = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"]
        encoder_state = {k: state[k] for k in encoder_keys}
        self.encoder.load_state_dict(encoder_state)

        self.Wp = state["Wp"]
        self.bp = state["bp"]
        self.Wv = state["Wv"]
        self.bv = state["bv"]
        self.Waux = state["Waux"]
        self.baux = state["baux"]


PolicyValueModel = MultiTaskModel
