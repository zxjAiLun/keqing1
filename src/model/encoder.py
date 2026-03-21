from __future__ import annotations

import numpy as np


class ResNetEncoder:
    def __init__(self, obs_dim: int = 270, hidden_dim: int = 256) -> None:
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        scale = 0.02
        self.W1 = np.random.randn(obs_dim, hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(x @ self.W1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
        h3 = np.maximum(h1 + (h2 @ self.W3 + self.b3), 0)
        return h3

    def state_dict(self) -> dict:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }

    def load_state_dict(self, state: dict) -> None:
        self.W1 = state["W1"]
        self.b1 = state["b1"]
        self.W2 = state["W2"]
        self.b2 = state["b2"]
        self.W3 = state["W3"]
        self.b3 = state["b3"]
