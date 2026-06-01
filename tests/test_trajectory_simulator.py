"""
Unit tests for TrajectorySimulator class.
"""

import pytest
import numpy as np
from src.trajectory_simulator import TrajectorySimulator, MotionModel

# Test constants
DEFAULT_DT = 0.1
KINEMATIC_ATOL = 1e-10  # Absolute tolerance for kinematic tests
NUMERICAL_ATOL = 1e-3   # Absolute tolerance for numerical integration
DEFAULT_PROCESS_NOISE = np.array([0.1, 0.1, 0.1])
DEFAULT_MEASUREMENT_NOISE = np.array([5.0, 5.0, 5.0])


class TestTrajectorySimulator:
    """Test cases for TrajectorySimulator class."""

    def test_initialization(self):
        """Test proper initialization of the simulator."""
        initial_state = np.array([100.0, 0.0, 50.0, 20.0, 5.0, 0.0])
        dt = 0.05

        sim = TrajectorySimulator(initial_state, dt=dt)

        assert np.allclose(sim.get_current_state(), initial_state)
        assert sim.dt == dt
        assert sim.motion_model == MotionModel.CONSTANT_VELOCITY
        assert np.allclose(sim.process_noise_std, DEFAULT_PROCESS_NOISE)

    def test_initialization_custom_noise(self):
        """Test initialization with custom process noise."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        custom_noise = np.array([0.5, 0.3, 0.2])

        sim = TrajectorySimulator(initial_state, process_noise_std=custom_noise)

        assert np.allclose(sim.process_noise_std, custom_noise)

    def test_initialization_invalid_state(self):
        """Test that invalid initial state raises error."""
        with pytest.raises(ValueError, match=r"Initial state must be 6-element array \[x, y, z, vx, vy, vz\]"):
            TrajectorySimulator(np.array([1.0, 2.0, 3.0]))  # Only 3 elements

    def test_constant_velocity_no_noise(self):
        """Test constant velocity motion without noise."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 2.0])  # 10 m/s east, 5 m/s north, 2 m/s up
        dt = 0.1

        # Create simulator with zero noise
        sim = TrajectorySimulator(initial_state, process_noise_std=np.zeros(3), dt=dt)

        # Step once
        new_state = sim.step()

        # Position should change by velocity * dt
        expected_position = initial_state[0:3] + initial_state[3:6] * dt
        assert np.allclose(new_state[0:3], expected_position, atol=KINEMATIC_ATOL)

        # Velocity should remain the same (no noise)
        assert np.allclose(new_state[3:6], initial_state[3:6], atol=KINEMATIC_ATOL)

    def test_constant_velocity_with_noise(self):
        """Test constant velocity motion with process noise."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        dt = 0.1
        noise_std = np.array([1.0, 1.0, 1.0])

        sim = TrajectorySimulator(initial_state, process_noise_std=noise_std, dt=dt)

        # Store initial velocity
        initial_velocity = initial_state[3:6].copy()

        # Step multiple times to accumulate noise effects
        for _ in range(100):
            sim.step()

        final_state = sim.get_current_state()

        # Position should have changed significantly
        assert not np.allclose(final_state[0:3], initial_state[0:3])

        # Velocity should have changed due to accumulated noise
        assert not np.allclose(final_state[3:6], initial_velocity)

    def test_constant_acceleration_model(self):
        """Test constant acceleration motion model."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.zeros(3),  # No noise for deterministic test
            dt=dt
        )

        # Step once
        new_state = sim.step()

        # With zero acceleration initially, should behave like constant velocity
        expected_position = initial_state[0:3] + initial_state[3:6] * dt
        assert np.allclose(new_state[0:3], expected_position, atol=KINEMATIC_ATOL)
        assert np.allclose(new_state[3:6], initial_state[3:6], atol=KINEMATIC_ATOL)

    def test_simulate_trajectory_constant_velocity(self):
        """Test trajectory simulation for constant velocity."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])  # Moving northeast at 10 m/s east, 5 m/s north
        duration = 1.0  # 1 second
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.zeros(3),  # No noise
            dt=dt
        )

        states, times = sim.simulate_trajectory(duration)

        # Should have 11 points (including initial)
        assert states.shape == (11, 6)
        assert times.shape == (11,)

        # Check time points
        expected_times = np.arange(0, 1.1, 0.1)
        assert np.allclose(times, expected_times)

        # Check final position
        final_position = states[-1, 0:3]
        expected_final = initial_state[0:3] + initial_state[3:6] * duration
        assert np.allclose(final_position, expected_final, atol=KINEMATIC_ATOL)

        # All velocities should be the same (no noise)
        for i in range(1, len(states)):
            assert np.allclose(states[i, 3:6], initial_state[3:6], atol=KINEMATIC_ATOL)

    def test_simulate_trajectory_without_initial(self):
        """Test trajectory simulation excluding initial state."""
        initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        duration = 0.5
        dt = 0.1

        sim = TrajectorySimulator(initial_state, dt=dt)

        states, times = sim.simulate_trajectory(duration, include_initial=False)

        # Should have 5 points (not including initial)
        assert states.shape == (5, 6)
        assert times.shape == (5,)

        # Times should start from dt, not 0
        assert times[0] == dt
        assert times[-1] == duration

    def test_noise_characteristics(self):
        """Test that process noise has correct statistical properties."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        dt = 0.1
        noise_std = np.array([1.0, 0.5, 0.2])
        n_samples = 10000

        sim = TrajectorySimulator(initial_state, process_noise_std=noise_std, dt=dt)

        # Collect velocity changes over many steps
        velocity_changes = []

        for _ in range(n_samples):
            initial_vel = sim.get_current_state()[3:6].copy()
            sim.step()
            final_vel = sim.get_current_state()[3:6]
            velocity_changes.append(final_vel - initial_vel)

        velocity_changes = np.array(velocity_changes)

        # Check that noise has zero mean
        mean_noise = np.mean(velocity_changes, axis=0)
        assert np.allclose(mean_noise, np.zeros(3), atol=0.1)

        # Check that noise has correct standard deviation
        std_noise = np.std(velocity_changes, axis=0)
        expected_std = noise_std * dt
        assert np.allclose(std_noise, expected_std, rtol=0.1)

    def test_set_state_and_reset(self):
        """Test state setting and reset functionality."""
        initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        new_state = np.array([100.0, 50.0, 25.0, 15.0, 8.0, 3.0])

        sim = TrajectorySimulator(initial_state)

        # Change state
        sim.set_state(new_state)
        assert np.allclose(sim.get_current_state(), new_state)

        # Reset to original
        sim.reset()
        assert np.allclose(sim.get_current_state(), initial_state)

        # Reset to different state
        sim.reset(new_state)
        assert np.allclose(sim.get_current_state(), new_state)

    def test_set_state_invalid(self):
        """Test that setting invalid state raises error."""
        sim = TrajectorySimulator(np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match=r"State must be 6-element array \[x, y, z, vx, vy, vz\]"):
            sim.set_state(np.array([1.0, 2.0, 3.0]))  # Only 3 elements

    def test_unsupported_motion_model(self):
        """Test that unsupported motion model raises error."""
        # This would require modifying the enum, but let's test the step method
        sim = TrajectorySimulator(np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]))

        # Manually set unsupported motion model (this is a bit of a hack for testing)
        sim.motion_model = "unsupported"

        with pytest.raises(ValueError, match="Unsupported motion model"):
            sim.step()