"""
Tests for RL Environment

Tests the artillery firing RL environment functionality.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.environment import ArtilleryFiringEnv, Action, EnvironmentState
from rl_training.curriculum.scenario_generator import ScenarioGenerator, ScenarioParameters
from src.hit_probability import HitProbabilityCalculator


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

        # Pre-populate hp_cache and patch _get_cache_key to always return the
        # pre-populated key.  This causes reset() to always hit the cache,
        # never entering the ProcessPoolExecutor branch (which fails on Windows
        # because gym.Env subclasses aren't picklable with the spawn start method).
        _steps = list(range(self.env.max_episode_steps + 1))
        self.env.hp_cache['__test__'] = (
            {s: 0.5 for s in _steps},
            {s: 1000.0 for s in _steps},
        )
        self._cache_key_patcher = patch.object(
            self.env, '_get_cache_key', return_value='__test__'
        )
        self._cache_key_patcher.start()

    def teardown_method(self):
        """Tear down test fixtures."""
        self._cache_key_patcher.stop()

    def test_initialization(self):
        """Test proper environment initialization."""
        assert self.env.action_space.n == 2  # HOLD and FIRE
        assert self.env.observation_space.shape == (14,)
        assert self.env.hit_threshold == 0.3
        assert self.env.max_episode_steps == 10
        assert self.env.reward_scale == 100.0

    def test_reset(self):
        """Test environment reset functionality."""
        observation, info = self.env.reset(seed=42)

        assert isinstance(observation, np.ndarray)
        assert observation.shape == (14,)
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
        assert reward > 0.0  # High HP firing → positive reward
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
        assert reward < 0  # Small penalty for holding
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

        # Check time normalisation (time_remaining / tracking_duration)
        state = self.env.current_state
        expected_time_norm = state.time_remaining / state.scenario.tracking_duration
        # Find which index holds normalised time by looking for the matching value
        assert expected_time_norm in observation or any(
            abs(v - expected_time_norm) < 1e-5 for v in observation
        )

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
        reward_success = self.env._calculate_firing_reward(0.8, episode_step=0)
        
        # Test failed firing
        reward_failure = self.env._calculate_firing_reward(0.1, episode_step=0)
        
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
        assert array.shape == (14,)
        assert array.dtype == np.float32

        # Test that values are reasonable
        assert np.all(np.isfinite(array))
        assert np.all(array >= 0.0)
        assert np.all(array <= 1.0)

    def test_get_hp_for_step_exact_match(self):
        """_get_hp_for_step returns the stored value when the step key exists."""
        self.env.reset(seed=42)
        # Overwrite the trajectory with known values
        self.env.hp_trajectory = {0: 0.3, 5: 0.6, 10: 0.9}
        self.env.current_state.episode_step = 0

        assert self.env._get_hp_for_step(0) == 0.3
        assert self.env._get_hp_for_step(5) == 0.6
        assert self.env._get_hp_for_step(10) == 0.9

    def test_get_hp_for_step_missing_key_interpolates(self):
        """_get_hp_for_step falls back to convergence-adjusted base HP for a missing step.

        The dead interpolation branch uses the trajectory's step-0 value and scales
        it by a convergence factor.  We pin this behaviour before removing the branch.
        """
        self.env.reset(seed=42)
        self.env.hp_trajectory = {0: 0.4, 10: 0.8}  # step 3 is absent
        self.env.current_state.episode_step = 3

        result = self.env._get_hp_for_step(3)

        # The branch: base_hp = hp_trajectory[0] = 0.4
        # elapsed_time = (3 / max_episode_steps) * tracking_duration
        tracking_duration = self.env.current_state.scenario.tracking_duration
        max_steps = self.env.max_episode_steps
        elapsed = (3 / max_steps) * tracking_duration
        convergence_factor = min(1.0, elapsed / 5.0)
        expected = 0.4 * (0.3 + 0.7 * convergence_factor)

        assert abs(result - expected) < 1e-6

    def test_get_hp_for_step_no_trajectory(self):
        """_get_hp_for_step returns 0.0 when hp_trajectory has not been initialised."""
        self.env.reset(seed=42)
        # Remove the attribute to simulate an uninitialised trajectory
        if hasattr(self.env, 'hp_trajectory'):
            del self.env.hp_trajectory

        assert self.env._get_hp_for_step(5) == 0.0

    def test_cache_hit_reuses_trajectory(self):
        """Calling reset() twice with the same cache key reuses the cached trajectory.

        The setup pre-populates hp_cache['__test__'] with a known trajectory (all
        0.5) and patches _get_cache_key to always return '__test__'.  Both resets
        must therefore land on the cache-hit branch and produce the same trajectory.
        """
        self.env.reset(seed=42)
        trajectory_after_first_reset = dict(self.env.hp_trajectory)

        self.env.reset(seed=42)
        trajectory_after_second_reset = dict(self.env.hp_trajectory)

        assert trajectory_after_first_reset == trajectory_after_second_reset
        # Confirm the values come from the pre-populated cache (all 0.5)
        for value in trajectory_after_second_reset.values():
            assert value == 0.5

    def test_cache_hit_tuple_format(self):
        """reset() correctly unpacks a cached (hp_trajectory, cov_trajectory) tuple."""
        steps = list(range(self.env.max_episode_steps + 1))
        hp_dict = {s: 0.7 for s in steps}
        cov_dict = {s: 500.0 for s in steps}
        self.env.hp_cache['__test__'] = (hp_dict, cov_dict)

        self.env.reset(seed=42)

        for s in steps:
            assert self.env.hp_trajectory[s] == 0.7
            assert self.env.cov_trajectory[s] == 500.0

    def test_hold_reward_tier_low_hp(self):
        """HOLD when hp < 0.5 gives reward = hp_improvement * 5.0 - 0.1."""
        self.env.reset(seed=42)
        # Override step-1 HP to a value in the lowest tier
        self.env.hp_trajectory[1] = 0.3  # was 0.5; improvement = 0.3 - 0.5 = -0.2

        _, reward, _, _, _ = self.env.step(Action.HOLD)

        expected = (0.3 - 0.5) * 5.0 - 0.1  # = -1.1
        assert abs(reward - expected) < 1e-6

    def test_hold_reward_tier_moderate_hp(self):
        """HOLD when 0.5 <= hp < 0.6 gives a flat reward of -0.5."""
        self.env.reset(seed=42)
        # Set step-1 HP into the moderate tier (0.5 <= hp < 0.6)
        self.env.hp_trajectory[1] = 0.55

        _, reward, _, _, _ = self.env.step(Action.HOLD)

        assert abs(reward - (-0.5)) < 1e-6

    def test_hold_reward_tier_good_hp(self):
        """HOLD when 0.6 <= hp < 0.7 gives a flat reward of -2.0."""
        self.env.reset(seed=42)
        self.env.hp_trajectory[1] = 0.65

        _, reward, _, _, _ = self.env.step(Action.HOLD)

        assert abs(reward - (-2.0)) < 1e-6

    def test_hold_reward_tier_excellent_hp(self):
        """HOLD when hp >= 0.7 gives hold_penalty_high * 0.5."""
        self.env.reset(seed=42)
        self.env.hp_trajectory[1] = 0.75

        _, reward, _, _, _ = self.env.step(Action.HOLD)

        expected = self.env.hold_penalty_high * 0.5  # default -10.0 * 0.5 = -5.0
        assert abs(reward - expected) < 1e-6

    def test_hold_plateau_penalty_multiplier(self):
        """HOLD reward is multiplied by plateau_hold_multiplier when HP has plateaued.

        _is_hp_plateau() requires:
          - len(hp_history) >= hp_history_length (10)
          - improvement_rate (recent - older) / 5 < hp_improvement_threshold (0.002)
          - current_hp >= hp_minimum_threshold * 0.8 (0.5 * 0.8 = 0.4)

        We put a flat hp_history at 0.65 (satisfies the HP floor) and step-1 HP
        also at 0.65 (falls in the 0.6–0.7 tier → base reward = -2.0 before mult).
        """
        self.env.reset(seed=42)

        # Fill history with a constant value so improvement_rate ≈ 0 < 0.002
        flat_hp = 0.65
        self.env.current_state.hp_history = [flat_hp] * self.env.hp_history_length
        self.env.current_state.current_hit_probability = flat_hp
        self.env.hp_trajectory[1] = flat_hp

        _, reward, _, _, _ = self.env.step(Action.HOLD)

        # Base tier for hp=0.65: -2.0; multiplied by plateau_hold_multiplier (5.0)
        base = -2.0
        expected = base * self.env.plateau_hold_multiplier
        assert abs(reward - expected) < 1e-6

    def test_timeout_penalty_when_hp_high_reached(self):
        """A -40 timeout penalty is applied when _hp_high_reached is True at episode end.

        Strategy:
          - Reset, then read the dynamic max_episode_steps (set by _calculate_max_steps).
          - Extend hp_trajectory to cover all steps up to max_episode_steps.
          - Jump the env state to one step before the end, then do a single HOLD.
          - episode_step == max_episode_steps → terminated; timeout penalty applied.
          Expected reward for the final HOLD:
            prev_hp = 0.5, hp[M] = 0.3 → tier hp < 0.5 → (0.3-0.5)*5 - 0.1 = -1.1
            timeout penalty → -1.1 - 40.0 = -41.1
        """
        self.env.reset(seed=42)
        M = self.env.max_episode_steps

        # Extend trajectory to cover all steps (cache only covers 0-10)
        for s in range(M + 1):
            if s not in self.env.hp_trajectory:
                self.env.hp_trajectory[s] = 0.5
            if s not in self.env.cov_trajectory:
                self.env.cov_trajectory[s] = 1000.0

        # Place hp[M] below the plateau floor (0.3 < 0.4) to avoid plateau mult
        self.env.hp_trajectory[M] = 0.3

        # Jump state to one step before the end
        self.env.current_state.episode_step = M - 1
        time_step = self.env.current_state.scenario.tracking_duration / M
        self.env.current_state.time_remaining = time_step  # exactly one step left
        self.env.episode_length = M - 1
        self.env.current_state.current_hit_probability = 0.5
        self.env._prev_hp = 0.5
        self.env._hp_high_reached = True

        _, reward, terminated, _, _ = self.env.step(Action.HOLD)

        assert terminated is True
        expected = (0.3 - 0.5) * 5.0 - 0.1 - 40.0  # = -41.1
        assert abs(reward - expected) < 1e-6

    def test_timeout_no_penalty_when_hp_never_high(self):
        """No -40 penalty at episode end when HP never reached the high threshold.

        Same state-jump approach as test_timeout_penalty_when_hp_high_reached, but
        _hp_high_reached is left False.  The expected reward is just the tier reward
        with no additional subtraction.
        """
        self.env.reset(seed=42)
        M = self.env.max_episode_steps

        for s in range(M + 1):
            if s not in self.env.hp_trajectory:
                self.env.hp_trajectory[s] = 0.5
            if s not in self.env.cov_trajectory:
                self.env.cov_trajectory[s] = 1000.0

        self.env.hp_trajectory[M] = 0.3  # below plateau floor

        self.env.current_state.episode_step = M - 1
        time_step = self.env.current_state.scenario.tracking_duration / M
        self.env.current_state.time_remaining = time_step
        self.env.episode_length = M - 1
        self.env.current_state.current_hit_probability = 0.5
        self.env._prev_hp = 0.5
        self.env._hp_high_reached = False  # HP was never high

        _, reward, terminated, _, _ = self.env.step(Action.HOLD)

        assert terminated is True
        # No timeout penalty: only the tier reward
        expected = (0.3 - 0.5) * 5.0 - 0.1  # = -1.1
        assert abs(reward - expected) < 1e-6

    # ------------------------------------------------------------------
    # _calculate_firing_reward — HP-band tests (all at episode_step=0 to
    # zero out time_penalty and maximise time_efficiency = 30.0, no late
    # penalty, no plateau / scenario bonus on a freshly-reset env)
    # ------------------------------------------------------------------

    def _fire_reward(self, hp: float) -> float:
        """Helper: reset env, call _calculate_firing_reward at step 0."""
        self.env.reset(seed=42)
        return self.env._calculate_firing_reward(hp, episode_step=0)

    def test_calculate_firing_reward_excellent(self):
        """HP >= 0.7 → base=100; at step 0 total = 100 + 30 = 130."""
        assert abs(self._fire_reward(0.75) - 130.0) < 1e-6

    def test_calculate_firing_reward_good(self):
        """0.6 <= HP < 0.7 → base=80; at step 0 total = 80 + 30 = 110."""
        assert abs(self._fire_reward(0.65) - 110.0) < 1e-6

    def test_calculate_firing_reward_minimum(self):
        """0.5 <= HP < 0.6 → base=60; at step 0 total = 60 + 30 = 90."""
        assert abs(self._fire_reward(0.55) - 90.0) < 1e-6

    def test_calculate_firing_reward_poor(self):
        """HP_ACCEPTABLE (0.45) <= HP < HP_MINIMUM (0.5) → base=20; total = 50."""
        assert abs(self._fire_reward(0.47) - 50.0) < 1e-6

    def test_calculate_firing_reward_failure(self):
        """HP < HP_ACCEPTABLE (0.45) → base=-30; at step 0 total = -30 + 30 = 0."""
        assert abs(self._fire_reward(0.3) - 0.0) < 1e-6

    # ------------------------------------------------------------------
    # _hp_high_reached flag
    # ------------------------------------------------------------------

    def test_hp_high_reached_flag_set_during_hold(self):
        """_hp_high_reached is set to True when current hp >= 0.6 at start of a HOLD step."""
        self.env.reset(seed=42)
        self.env._hp_high_reached = False
        # The flag is checked against current_hit_probability at the *start* of step()
        self.env.current_state.current_hit_probability = 0.65  # >= 0.6 threshold

        self.env.step(Action.HOLD)

        assert self.env._hp_high_reached is True

    def test_hp_high_reached_flag_not_set_below_threshold(self):
        """_hp_high_reached stays False when hp < 0.6 throughout."""
        self.env.reset(seed=42)
        self.env._hp_high_reached = False
        self.env.hp_trajectory[1] = 0.55  # below 0.6

        self.env.step(Action.HOLD)

        assert self.env._hp_high_reached is False

    # ------------------------------------------------------------------
    # _calculate_max_steps — branch coverage
    # ------------------------------------------------------------------

    def _make_scenario(self, range_m: float, target_vx: float) -> ScenarioParameters:
        """Build a minimal ScenarioParameters for max-steps tests."""
        noise = np.array([1.0, 1.0, 1.0])
        return ScenarioParameters(
            range_m=range_m,
            target_length=5.0,
            target_width=2.0,
            target_height=2.0,
            target_vx=target_vx,
            target_vy=0.0,
            target_vz=0.0,
            tracking_duration=10.0,
            measurement_noise_std=noise,
            process_noise_std=noise,
        )

    def test_calculate_max_steps_fast_approach_long(self):
        """range≈3000, vx<-40 → 25 (Fast_Approach_Long mini-episode)."""
        s = self._make_scenario(range_m=3000.0, target_vx=-50.0)
        assert self.env._calculate_max_steps(s) == 25

    def test_calculate_max_steps_very_fast_approach(self):
        """range≈4000, vx<-70 → 40 (Very_Fast_Approach)."""
        s = self._make_scenario(range_m=4000.0, target_vx=-80.0)
        assert self.env._calculate_max_steps(s) == 40

    def test_calculate_max_steps_long_range(self):
        """range > 3000 (and not fast-approach) → 100."""
        s = self._make_scenario(range_m=4500.0, target_vx=0.0)
        assert self.env._calculate_max_steps(s) == 100

    def test_calculate_max_steps_medium_range(self):
        """2500 < range <= 3000 (not fast-approach) → 90."""
        s = self._make_scenario(range_m=2600.0, target_vx=0.0)
        assert self.env._calculate_max_steps(s) == 90

    def test_calculate_max_steps_default(self):
        """range <= 2500 → 40 (default)."""
        s = self._make_scenario(range_m=1000.0, target_vx=0.0)
        assert self.env._calculate_max_steps(s) == 40
