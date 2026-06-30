"""
Tests for DQN network and ReplayBuffer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from rl_training.agents.dqn_components import DQN, ReplayBuffer


def test_dqn_forward_shape():
    net = DQN(state_dim=14, action_dim=2, hidden_dim=32)
    out = net(torch.zeros((4, 14)))
    assert out.shape == (4, 2)


def test_dqn_forward_single():
    net = DQN(state_dim=14, action_dim=2)
    out = net(torch.zeros((1, 14)))
    assert out.shape == (1, 2)


def test_replay_buffer_push_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    for _ in range(10):
        buf.push([0.0] * 3, 1, 1.0, [0.0] * 3, False)
    assert len(buf) == 10


def test_replay_buffer_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push([float(i)] * 3, i % 2, float(i), [0.0] * 3, i == 9)
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert len(states) == 4
    assert len(actions) == 4
    assert len(rewards) == 4
    assert len(next_states) == 4
    assert len(dones) == 4


def test_replay_buffer_capacity_enforced():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(i, i, i, i, False)
    assert len(buf) == 3
