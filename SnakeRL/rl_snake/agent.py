"""Double DQN agent with replay buffer and optional greedy exploration."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DuelingDQN


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.FloatTensor(np.asarray(obs, dtype=np.float32)).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(np.asarray(next_obs, dtype=np.float32)).to(device),
            torch.FloatTensor(done).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        input_dim: int = 47,
        n_actions: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        self.q_net = DuelingDQN(input_dim=input_dim, n_actions=n_actions).to(device)
        self.target_net = DuelingDQN(input_dim=input_dim, n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def act(
        self,
        obs: np.ndarray,
        epsilon: float,
        greedy_action: int | None = None,
        greedy_ratio: float = 0.5,
    ) -> int:
        """Epsilon-greedy action selection.

        In the random branch, with probability `greedy_ratio`, a legal greedy
        move toward the food is selected instead of a uniformly random move.
        """
        if random.random() < epsilon:
            if greedy_action is not None and random.random() < greedy_ratio:
                return int(greedy_action)
            return random.randrange(self.n_actions)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(obs_t)
        return int(q.argmax(dim=1).item())

    def update(self, batch):
        obs, action, reward, next_obs, done = batch

        q = self.q_net(obs)
        q_values = q.gather(1, action.unsqueeze(1)).squeeze(1)
        avg_max_q = q.max(dim=1).values.mean().item()

        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target = reward + (1.0 - done) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item()), avg_max_q

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
