"""
Tests for rl_training/train/train_dqn_curriculum.py

Covers: select_action — greedy branch, random branch, device handling,
        output range, and boundary epsilon values.
"""

import random
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.train.train_dqn_curriculum import select_action
from rl_training.agents.dqn_components import DQN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATE_DIM = 14  # observation space size
_ACTION_DIM = 2  # HOLD=0, FIRE=1
_DEVICE = torch.device("cpu")


def _make_q_network(preferred_action: int = 1) -> DQN:
    """Return a DQN whose output strongly favours *preferred_action*."""
    net = DQN(_STATE_DIM, _ACTION_DIM).to(_DEVICE)
    # Zero all output weights, then set a large bias for the preferred action
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        # The final linear layer — set a large value for preferred_action
        final_layer = list(net.modules())[-1]
        if isinstance(final_layer, torch.nn.Linear):
            final_layer.bias[preferred_action] = 100.0
    return net


def _dummy_state() -> list:
    return [0.0] * _STATE_DIM


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectAction:

    def test_greedy_returns_highest_q_action(self):
        """With epsilon=0 the greedy action matching the network bias is chosen."""
        net = _make_q_network(preferred_action=1)
        action = select_action(_dummy_state(), net, epsilon=0.0,
                               action_dim=_ACTION_DIM, device=_DEVICE)
        assert action == 1

    def test_greedy_honours_network_for_action_zero(self):
        """Greedy picks action 0 when the network is biased that way."""
        net = _make_q_network(preferred_action=0)
        action = select_action(_dummy_state(), net, epsilon=0.0,
                               action_dim=_ACTION_DIM, device=_DEVICE)
        assert action == 0

    def test_random_action_in_valid_range(self):
        """With epsilon=1 the returned action is always in [0, action_dim)."""
        net = _make_q_network()
        for _ in range(50):
            action = select_action(_dummy_state(), net, epsilon=1.0,
                                   action_dim=_ACTION_DIM, device=_DEVICE)
            assert 0 <= action < _ACTION_DIM

    def test_random_branch_called_with_epsilon_one(self):
        """With epsilon=1, random.randrange is always called."""
        net = _make_q_network(preferred_action=1)
        call_count = 0
        original_randrange = random.randrange

        def counting_randrange(n):
            nonlocal call_count
            call_count += 1
            return original_randrange(n)

        import unittest.mock as mock
        with mock.patch("rl_training.train.train_dqn_curriculum.random.randrange",
                        side_effect=counting_randrange):
            for _ in range(10):
                select_action(_dummy_state(), net, epsilon=1.0,
                              action_dim=_ACTION_DIM, device=_DEVICE)

        assert call_count == 10

    def test_greedy_branch_not_random_with_epsilon_zero(self):
        """With epsilon=0, random.randrange is never called."""
        net = _make_q_network(preferred_action=1)

        import unittest.mock as mock
        with mock.patch("rl_training.train.train_dqn_curriculum.random.randrange") as mock_rr:
            for _ in range(10):
                select_action(_dummy_state(), net, epsilon=0.0,
                              action_dim=_ACTION_DIM, device=_DEVICE)
        mock_rr.assert_not_called()

    def test_return_type_is_int(self):
        """select_action always returns a plain Python int."""
        net = _make_q_network()
        action = select_action(_dummy_state(), net, epsilon=0.0,
                               action_dim=_ACTION_DIM, device=_DEVICE)
        assert isinstance(action, int)

    def test_numpy_state_accepted(self):
        """select_action works when state is a numpy array."""
        import numpy as np
        net = _make_q_network(preferred_action=1)
        state = np.zeros(_STATE_DIM, dtype=np.float32)
        action = select_action(state, net, epsilon=0.0,
                               action_dim=_ACTION_DIM, device=_DEVICE)
        assert 0 <= action < _ACTION_DIM
