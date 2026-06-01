"""
DQN Components for Artillery Firing Training

Contains the DQN network and ReplayBuffer classes.
"""

import torch
import torch.nn as nn
from collections import deque
import random
from typing import Tuple


class DQN(nn.Module):
    """Deep Q-Network for artillery firing decisions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)