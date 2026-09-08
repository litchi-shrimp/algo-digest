"""Pure-NumPy Dueling DQN inference (torch-free, ~50MB footprint instead of ~500MB)."""

from __future__ import annotations

import numpy as np


class DuelingDQNNumpy:
    def __init__(self):
        self.params = None

    def load(self, path):
        data = np.load(path)
        self.params = {k: data[k].astype(np.float32) for k in data.files}

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """obs: (47,) float32 -> q: (3,) float32."""
        p = self.params
        x = obs.astype(np.float32)
        x = np.maximum(np.dot(p["shared.0.weight"], x) + p["shared.0.bias"], 0.0)
        x = np.maximum(np.dot(p["shared.2.weight"], x) + p["shared.2.bias"], 0.0)
        value = np.dot(p["value.weight"], x) + p["value.bias"]
        advantage = np.dot(p["advantage.weight"], x) + p["advantage.bias"]
        return value + (advantage - advantage.mean())
