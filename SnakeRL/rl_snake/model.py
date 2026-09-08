"""Fully-connected dueling Q network (no CNN)."""

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(
        self,
        input_dim: int = 47,
        n_actions: int = 3,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        layers = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        self.shared = nn.Sequential(*layers)

        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.shared(obs)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
