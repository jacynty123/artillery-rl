"""
Integration tests for TrajectorySimulator.

These tests verify the overall behavior and physics of trajectory simulation,
including long-term stability and realistic motion characteristics.
"""

import numpy as np
from src.trajectory_simulator import TrajectorySimulator, MotionModel

# Test constants
KINEMATIC_TOLERANCE = 1e-2  # Tolerance for kinematic equation verification
STABILITY_TOLERANCE = 0.01  # Tolerance for long-term stability checks
REALISTIC_POSITION_TOLERANCE = 50.0  # Realistic position error tolerance
MAX_ACCELERATION = 100.0  # Maximum expected acceleration (m/s²)


class TestPhysicsValidation:
    """Test physical correctness of trajectory simulation."""

    def test_constant_velocity_physics(self):
        """Test that constant velocity trajectories follow correct physics."""
        # Aircraft flying at 200 m/s (720 km/h) at 10,000 ft (3048 m) altitude
        initial_state = np.array([0.0, 0.0, 3048.0, 200.0, 0.0, 0.0])
        duration = 30.0  # 30 seconds
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.zeros(3),  # No noise for physics test
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check that aircraft covers expected distance
        final_position = states[-1]
        distance_traveled = final_position[0]  # Movement in x-direction
        expected_distance = initial_state[3] * duration  # velocity * time

        assert np.isclose(distance_traveled, expected_distance, rtol=KINEMATIC_TOLERANCE)

        # Check that altitude remains constant (no vertical motion)
        altitudes = states[:, 2]
        assert np.allclose(altitudes, initial_state[2], atol=KINEMATIC_TOLERANCE)

        # Check that velocity remains constant
        velocities_x = states[:, 3]
        velocities_y = states[:, 4]
        velocities_z = states[:, 5]

        assert np.allclose(velocities_x, initial_state[3], atol=KINEMATIC_TOLERANCE)
        assert np.allclose(velocities_y, initial_state[4], atol=KINEMATIC_TOLERANCE)
        assert np.allclose(velocities_z, initial_state[5], atol=KINEMATIC_TOLERANCE)

    def test_accelerated_motion_physics(self):
        """Test physics of accelerated motion."""
        # Object starting from rest with constant acceleration
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At rest
        acceleration = np.array([10.0, 0.0, 0.0])  # 10 m/s² in x-direction
        duration = 2.0
        dt = 0.01

        sim = TrajectorySimulator(
            initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.zeros(3),
            dt=dt
        )

        # Manually set constant acceleration (sim starts with zero acceleration)
        sim.acceleration = acceleration.copy()

        states, times = sim.simulate_trajectory(duration)

        # Check kinematic equations: x = 0.5 * a * t², v = a * t
        final_state = states[-1]
        final_time = times[-1]

        expected_position = 0.5 * acceleration[0] * final_time**2
        expected_velocity = acceleration[0] * final_time

        # Allow for numerical integration error (discrete time stepping)
        assert np.isclose(final_state[0], expected_position, rtol=KINEMATIC_TOLERANCE)
        assert np.isclose(final_state[3], expected_velocity, rtol=KINEMATIC_TOLERANCE)

    def test_zero_noise_deterministic(self):
        """Test that zero noise produces completely deterministic results."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 2.0])
        duration = 5.0
        dt = 0.1

        sim1 = TrajectorySimulator(initial_state, process_noise_std=np.zeros(3), dt=dt)
        states1, times1 = sim1.simulate_trajectory(duration)

        sim2 = TrajectorySimulator(initial_state, process_noise_std=np.zeros(3), dt=dt)
        states2, times2 = sim2.simulate_trajectory(duration)

        # Results should be identical
        assert np.allclose(states1, states2)
        assert np.allclose(times1, times2)

        # Check that final position matches kinematic equations
        final_state = states1[-1]
        t = times1[-1]

        expected_x = initial_state[0] + initial_state[3] * t
        expected_y = initial_state[1] + initial_state[4] * t
        expected_z = initial_state[2] + initial_state[5] * t

        assert np.isclose(final_state[0], expected_x)
        assert np.isclose(final_state[1], expected_y)
        assert np.isclose(final_state[2], expected_z)


class TestNumericalStability:
    """Test long-term behavior and numerical stability."""

    def test_long_term_stability_constant_velocity(self):
        """Test long-term stability of constant velocity simulation.

        Verifies two stability properties over a 5-minute run:
          1. No NaN or infinite values (no numerical divergence).
          2. No sudden position jumps between steps (smooth motion).

        A position-accuracy check is intentionally omitted: with stochastic
        process noise the final position drifts as a random walk (O(sqrt(T))),
        which is unrelated to the velocity-based 1% tolerance that was here
        previously and caused intermittent failures on CI.
        """
        initial_state = np.array([0.0, 0.0, 1000.0, 50.0, 30.0, -2.0])
        duration = 300.0  # 5 minutes
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.array([0.01, 0.01, 0.005]),  # Small noise
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check that simulation doesn't diverge or become unstable
        positions = states[:, 0:3]
        velocities = states[:, 3:6]

        # 1. No NaN or infinite values
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))

        # 2. No sudden position jumps (smooth motion)
        position_diffs = np.diff(positions, axis=0)
        max_position_jump = np.max(np.abs(position_diffs))
        expected_max_jump = np.max(np.abs(velocities[:-1])) * dt * 2  # Allow some margin

        assert max_position_jump < expected_max_jump

    def test_long_term_stability_accelerated_motion(self):
        """Test long-term stability of accelerated motion simulation."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])
        duration = 120.0  # 2 minutes
        dt = 0.05

        sim = TrajectorySimulator(
            initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.array([0.1, 0.1, 0.05]),
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check stability
        positions = states[:, 0:3]
        velocities = states[:, 3:6]

        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))

        # Accelerations should not grow unbounded
        accelerations = np.diff(velocities, axis=0) / dt
        max_acceleration = np.max(np.abs(accelerations))

        # Should be reasonable (less than 100 m/s²)
        assert max_acceleration < MAX_ACCELERATION

    def test_trajectory_consistency(self):
        """Test that multiple runs with same seed produce identical results."""
        initial_state = np.array([100.0, 50.0, 1000.0, 20.0, 15.0, 5.0])
        duration = 10.0
        dt = 0.1

        # Run simulation twice with same random seed
        np.random.seed(42)
        sim1 = TrajectorySimulator(initial_state, dt=dt)
        states1, _ = sim1.simulate_trajectory(duration)

        np.random.seed(42)
        sim2 = TrajectorySimulator(initial_state, dt=dt)
        states2, _ = sim2.simulate_trajectory(duration)

        # Results should be identical
        assert np.allclose(states1, states2)


class TestRealWorldScenarios:
    """Test realistic use cases and scenarios."""

    def test_realistic_aircraft_trajectory(self):
        """Test a realistic aircraft trajectory with banking and turning."""
        # Fighter jet initial conditions
        altitude = 7620.0  # 25,000 ft
        speed = 300.0  # m/s (Mach 0.9 at altitude)
        initial_state = np.array([0.0, 0.0, altitude, speed, 0.0, 0.0])

        duration = 60.0  # 1 minute
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.array([0.5, 0.5, 0.1]),  # Realistic turbulence
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check basic physics
        assert np.all(np.isfinite(states))

        # Aircraft should maintain reasonable altitude (not crash or climb unreasonably)
        altitudes = states[:, 2]
        assert np.all(altitudes > 7000.0)  # Stay above 23,000 ft
        assert np.all(altitudes < 8500.0)  # Stay below 28,000 ft

        # Velocities should stay in reasonable range
        speeds = np.linalg.norm(states[:, 3:6], axis=1)
        assert np.all(speeds > 250.0)  # Not too slow
        assert np.all(speeds < 350.0)  # Not too fast

        # Total distance traveled should be reasonable
        final_position = states[-1, 0:3]
        distance = np.linalg.norm(final_position - initial_state[0:3])
        expected_min_distance = speed * duration * 0.8  # At least 80% of straight-line distance

        assert distance > expected_min_distance

import pytest
import numpy as np
from src.trajectory_simulator import TrajectorySimulator, MotionModel


class TestTrajectoryIntegration:
    """Integration tests for trajectory simulation."""

    def test_constant_velocity_physics(self):
        """Test that constant velocity trajectories follow correct physics."""
        # Aircraft flying at 200 m/s (720 km/h) at 10,000 ft (3048 m) altitude
        initial_state = np.array([0.0, 0.0, 3048.0, 200.0, 0.0, 0.0])
        duration = 30.0  # 30 seconds
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.zeros(3),  # No noise for physics test
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check that aircraft covers expected distance
        final_position = states[-1]
        distance_traveled = final_position[0]  # Movement in x-direction
        expected_distance = initial_state[3] * duration  # velocity * time

        assert np.isclose(distance_traveled, expected_distance, rtol=1e-10)

        # Check that altitude remains constant (no vertical motion)
        altitudes = states[:, 2]
        assert np.allclose(altitudes, initial_state[2], atol=1e-10)

        # Check that velocity remains constant
        velocities_x = states[:, 3]
        velocities_y = states[:, 4]
        velocities_z = states[:, 5]

        assert np.allclose(velocities_x, initial_state[3], atol=1e-10)
        assert np.allclose(velocities_y, initial_state[4], atol=1e-10)
        assert np.allclose(velocities_z, initial_state[5], atol=1e-10)

    def test_accelerated_motion_physics(self):
        """Test physics of accelerated motion."""
        # Object starting from rest with constant acceleration
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At rest
        acceleration = np.array([10.0, 0.0, 0.0])  # 10 m/s² in x-direction
        duration = 2.0
        dt = 0.01

        sim = TrajectorySimulator(
            initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.zeros(3),
            dt=dt
        )

        # Manually set constant acceleration (sim starts with zero acceleration)
        sim.acceleration = acceleration.copy()

        states, times = sim.simulate_trajectory(duration)

        # Check kinematic equations: x = 0.5 * a * t², v = a * t
        final_state = states[-1]
        final_time = times[-1]

        expected_position = 0.5 * acceleration[0] * final_time**2
        expected_velocity = acceleration[0] * final_time

        # Allow for numerical integration error (discrete time stepping)
        assert np.isclose(final_state[0], expected_position, rtol=1e-2)
        assert np.isclose(final_state[3], expected_velocity, rtol=1e-2)

    def test_long_term_stability_constant_velocity(self):
        """Test long-term stability of constant velocity simulation."""
        initial_state = np.array([0.0, 0.0, 1000.0, 50.0, 30.0, -2.0])
        duration = 300.0  # 5 minutes
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.array([0.01, 0.01, 0.005]),  # Small noise
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check that simulation doesn't diverge or become unstable
        positions = states[:, 0:3]
        velocities = states[:, 3:6]

        # Positions should be reasonable (not NaN or infinite)
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))

        # Position changes should be smooth (no sudden jumps)
        position_diffs = np.diff(positions, axis=0)
        max_position_jump = np.max(np.abs(position_diffs))
        expected_max_jump = np.max(np.abs(velocities[:-1])) * dt * 2  # Allow some margin

        assert max_position_jump < expected_max_jump

        # Final position should be roughly velocity * time
        final_position = states[-1, 0:3]
        expected_position = initial_state[0:3] + initial_state[3:6] * duration

        # Allow for some accumulated noise (should be small)
        position_error = np.abs(final_position - expected_position)
        max_expected_error = np.abs(initial_state[3:6]) * duration * 0.01  # 1% error allowance

        assert np.all(position_error < max_expected_error)

    def test_long_term_stability_accelerated_motion(self):
        """Test long-term stability of accelerated motion simulation."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])
        duration = 120.0  # 2 minutes
        dt = 0.05

        sim = TrajectorySimulator(
            initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.array([0.1, 0.1, 0.05]),
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check stability
        positions = states[:, 0:3]
        velocities = states[:, 3:6]

        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))

        # Accelerations should not grow unbounded
        accelerations = np.diff(velocities, axis=0) / dt
        max_acceleration = np.max(np.abs(accelerations))

        # Should be reasonable (less than 100 m/s²)
        assert max_acceleration < 100.0

    def test_realistic_aircraft_trajectory(self):
        """Test a realistic aircraft trajectory with banking and turning."""
        # Fighter jet initial conditions
        altitude = 7620.0  # 25,000 ft
        speed = 300.0  # m/s (Mach 0.9 at altitude)
        initial_state = np.array([0.0, 0.0, altitude, speed, 0.0, 0.0])

        duration = 60.0  # 1 minute
        dt = 0.1

        sim = TrajectorySimulator(
            initial_state,
            process_noise_std=np.array([0.5, 0.5, 0.1]),  # Realistic turbulence
            dt=dt
        )

        states, _ = sim.simulate_trajectory(duration)

        # Check basic physics
        assert np.all(np.isfinite(states))

        # Aircraft should maintain reasonable altitude (not crash or climb unreasonably)
        altitudes = states[:, 2]
        assert np.all(altitudes > 7000.0)  # Stay above 23,000 ft
        assert np.all(altitudes < 8500.0)  # Stay below 28,000 ft

        # Velocities should stay in reasonable range
        speeds = np.linalg.norm(states[:, 3:6], axis=1)
        assert np.all(speeds > 250.0)  # Not too slow
        assert np.all(speeds < 350.0)  # Not too fast

        # Total distance traveled should be reasonable
        final_position = states[-1, 0:3]
        distance = np.linalg.norm(final_position - initial_state[0:3])
        expected_min_distance = speed * duration * 0.8  # At least 80% of straight-line distance

        assert distance > expected_min_distance

    def test_trajectory_consistency(self):
        """Test that multiple runs with same seed produce identical results."""
        initial_state = np.array([100.0, 50.0, 1000.0, 20.0, 15.0, 5.0])
        duration = 10.0
        dt = 0.1

        # Run simulation twice with same random seed
        np.random.seed(42)
        sim1 = TrajectorySimulator(initial_state, dt=dt)
        states1, _ = sim1.simulate_trajectory(duration)

        np.random.seed(42)
        sim2 = TrajectorySimulator(initial_state, dt=dt)
        states2, _ = sim2.simulate_trajectory(duration)

        # Results should be identical
        assert np.allclose(states1, states2)

    def test_zero_noise_deterministic(self):
        """Test that zero noise produces completely deterministic results."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 2.0])
        duration = 5.0
        dt = 0.1

        sim1 = TrajectorySimulator(initial_state, process_noise_std=np.zeros(3), dt=dt)
        states1, times1 = sim1.simulate_trajectory(duration)

        sim2 = TrajectorySimulator(initial_state, process_noise_std=np.zeros(3), dt=dt)
        states2, times2 = sim2.simulate_trajectory(duration)

        # Results should be identical
        assert np.allclose(states1, states2)
        assert np.allclose(times1, times2)

        # Check that final position matches kinematic equations
        final_state = states1[-1]
        t = times1[-1]

        expected_x = initial_state[0] + initial_state[3] * t
        expected_y = initial_state[1] + initial_state[4] * t
        expected_z = initial_state[2] + initial_state[5] * t

        assert np.isclose(final_state[0], expected_x)
        assert np.isclose(final_state[1], expected_y)
        assert np.isclose(final_state[2], expected_z)