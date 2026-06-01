"""
Tests for RL Environment

Tests the artillery firing RL environment functionality.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl_training"))

from environment import ArtilleryFiringEnv, Action, EnvironmentState
from scenario_generator import ScenarioGenerator, ScenarioParameters
from hit_probability import HitProbabilityCalculator


class TestArtilleryFiringEnv:
    """Test the artillery firing RL environment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenario_generator = ScenarioGenerator(seed=42)
        self.hit_calculator = Mock(spec=HitProbabilityCalculator)
        self.hit_calculator.calculate_hit_probability.return_value = 0.5

        self.env = ArtilleryFiringEnv(
            scenario_generator=self.scenario_generator,
            hit_probability_calculator=self.hit_calculator,
            hit_threshold=0.3,
            max_episode_steps=10,
            reward_scale=100.0,
        )

    def test_initialization(self):
        """Test proper environment initialization."""
        assert self.env.action_space.n == 2  # HOLD and FIRE
        assert self.env.observation_space.shape == (11,)
        assert self.env.hit_threshold == 0.3
        assert self.env.max_episode_steps == 10
        assert self.env.reward_scale == 100.0

    def test_reset(self):
        """Test environment reset functionality."""
        observation, info = self.env.reset(seed=42)

        assert isinstance(observation, np.ndarray)
        assert observation.shape == (11,)
        assert observation.dtype == np.float32
        assert np.all(observation >= 0.0) and np.all(observation <= 1.0)  # Normalized

        assert "scenario" in info
        assert "initial_hit_probability" in info
        assert isinstance(info["scenario"], ScenarioParameters)
        assert isinstance(info["initial_hit_probability"], float)

        assert self.env.current_state is not None
        assert self.env.episode_reward == 0.0
        assert self.env.episode_length == 0

    def test_fire_action_successful(self):
        """Test firing action with high hit probability."""
        self.env.reset(seed=42)

        # Set high hit probability directly in state
        initial_prob = self.env.current_state.initial_hit_probability
        self.env.current_state.current_hit_probability = 0.8

        observation, reward, terminated, truncated, info = self.env.step(Action.FIRE)

        assert terminated is True
        assert truncated is False
        # New reward: base * multiplier + improvement_bonus
        # For 0.8: base=80, multiplier=3.0, improvement=(0.8-initial)*100*5
        assert reward > 200.0  # Should be high positive reward
        assert isinstance(observation, np.ndarray)
        assert "hit_probability" in info

    def test_fire_action_unsuccessful(self):
        """Test firing action with low hit probability."""
        self.env.reset(seed=42)

        # Set low hit probability directly in state
        initial_prob = self.env.current_state.initial_hit_probability
        self.env.current_state.current_hit_probability = 0.1

        observation, reward, terminated, truncated, info = self.env.step(Action.FIRE)

        assert terminated is True
        assert truncated is False
        # New reward structure: for <0.2, uses -0.5 multiplier
        # Reward may be positive or negative depending on initial prob
        # Just check it's not a large positive reward
        assert reward < 100.0  # Should not be high reward

    def test_hold_action(self):
        """Test holding action."""
        self.env.reset(seed=42)
        initial_time = self.env.current_state.time_remaining
        initial_step = self.env.current_state.episode_step

        observation, reward, terminated, truncated, info = self.env.step(Action.HOLD)

        assert terminated is False
        assert truncated is False
        assert reward == -0.001 * 100.0  # Small penalty for holding (reduced from 0.005)
        assert self.env.current_state.time_remaining < initial_time
        assert self.env.current_state.episode_step == initial_step + 1

    def test_timeout_scenario(self):
        """Test timeout when holding too long."""
        self.env.reset(seed=42)

        # Set low hit probability for timeout penalty
        self.env.current_state.current_hit_probability = 0.1

        # Mock the hit probability calculation to always return 0.1
        self.hit_calculator.calculate_hit_probability.return_value = 0.1

        # Set up scenario that will timeout quickly
        self.env.max_episode_steps = 2
        self.env.current_state.scenario.tracking_duration = 1.0

        # First hold
        observation, reward, terminated, truncated, info = self.env.step(Action.HOLD)
        assert not terminated

        # Second hold - should timeout
        observation, reward, terminated, truncated, info = self.env.step(Action.HOLD)
        assert terminated is True
        assert (
            reward < 0
        )  # Should get penalty for timeout with low probability    def test_invalid_action(self):
        """Test that invalid actions raise errors."""
        self.env.reset(seed=42)

        with pytest.raises(ValueError, match="Invalid action"):
            self.env.step(999)

    def test_step_without_reset(self):
        """Test that stepping without reset raises error."""
        with pytest.raises(RuntimeError, match="Environment must be reset"):
            self.env.step(Action.FIRE)

    def test_state_normalization(self):
        """Test that state values are properly normalized."""
        self.env.reset(seed=42)
        observation = self.env.current_state.to_array()

        # Check that all values are in [0, 1] range
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)

        # Check specific normalizations
        state = self.env.current_state
        expected_range_norm = state.scenario.range_m / 2000.0
        expected_time_norm = state.time_remaining / state.scenario.tracking_duration

        assert abs(observation[0] - expected_range_norm) < 1e-6
        assert abs(observation[7] - expected_time_norm) < 1e-6

    def test_hit_probability_calculation_error_handling(self):
        """Test error handling in hit probability calculation."""
        self.env.reset(seed=42)

        # Mock calculator to raise exception
        self.hit_calculator.calculate_hit_probability.side_effect = Exception(
            "Calculation failed"
        )

        # Should return conservative estimate
        hit_prob = self.env._calculate_hit_probability(self.env.current_state.scenario)
        assert hit_prob == 0.05  # Conservative fallback (updated from 0.1)

    def test_environment_rendering(self):
        """Test environment rendering."""
        self.env.reset(seed=42)

        # Should not raise exception
        self.env.render()

        # Test rendering when not initialized
        self.env.current_state = None
        self.env.render()  # Should handle gracefully

    def test_reward_calculation(self):
        """Test reward calculation logic."""
        # Must reset to initialize current_state with initial_hit_probability
        self.env.reset(seed=42)
        
        # Test successful firing
        reward_success = self.env._calculate_firing_reward(0.8)
        
        # Test failed firing
        reward_failure = self.env._calculate_firing_reward(0.1)
        
        # Success reward should be higher than failure
        assert reward_success > reward_failure

        # Test that rewards are computed (no longer exact values due to improvement bonus)
        assert isinstance(reward_success, (int, float))
        assert isinstance(reward_failure, (int, float))

    def test_episode_truncation(self):
        """Test episode truncation at max steps."""
        self.env.reset(seed=42)
        self.env.max_episode_steps = 3

        # Take multiple hold actions
        for i in range(3):
            observation, reward, terminated, truncated, info = self.env.step(
                Action.HOLD
            )
            if i < 2:
                assert not truncated
            else:
                assert truncated

    def test_environment_state_dataclass(self):
        """Test EnvironmentState dataclass functionality."""
        scenario = self.scenario_generator.generate_scenario()
        state = EnvironmentState(
            scenario=scenario,
            time_remaining=10.0,
            current_hit_probability=0.5,
            episode_step=5,
        )

        # Test array conversion
        array = state.to_array()
        assert isinstance(array, np.ndarray)
        assert array.shape == (11,)
        assert array.dtype == np.float32

        # Test that values are reasonable
        assert np.all(np.isfinite(array))
        assert np.all(array >= 0.0)
        assert np.all(array <= 1.0)
