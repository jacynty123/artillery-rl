"""
Integration tests for Kalman filter with trajectory simulator.

These tests verify the Kalman filter works correctly with simulated trajectories,
including tracking performance, covariance evolution, and realistic scenarios.
"""

import pytest
import numpy as np
from src.kalman_filter import TargetKalmanFilter, MeasurementType
from src.trajectory_simulator import TrajectorySimulator, MotionModel

# Test constants
DEFAULT_DT = 0.1
KINEMATIC_ATOL = 1e-10  # Absolute tolerance for kinematic tests
NUMERICAL_ATOL = 1e-3   # Absolute tolerance for numerical integration
DEFAULT_PROCESS_NOISE = np.array([0.1, 0.1, 0.1])
DEFAULT_MEASUREMENT_NOISE_POS = np.array([5.0, 5.0, 5.0])  # Position measurement noise (meters)
DEFAULT_MEASUREMENT_NOISE_VEL = np.array([1.0, 1.0, 1.0])  # Velocity measurement noise (m/s)


class TestKalmanTrajectoryIntegration:
    """Integration tests for Kalman filter with trajectory simulation."""

    def test_basic_tracking_constant_velocity(self):
        """Test basic tracking of constant velocity trajectory."""
        # Create true trajectory
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])  # 10 m/s east, 5 m/s north
        duration = 2.0  # 2 seconds

        simulator = TrajectorySimulator(
            initial_state=initial_state,
            process_noise_std=np.zeros(3),  # Deterministic trajectory for testing
            dt=DEFAULT_DT
        )

        # Generate true trajectory
        true_states, times = simulator.simulate_trajectory(duration)

        # Initialize Kalman filter with some uncertainty
        initial_estimate = initial_state + np.array([50.0, 30.0, 20.0, 2.0, 1.0, 0.5])  # Biased initial estimate
        initial_cov = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 5.0])  # Initial uncertainty

        kf = TargetKalmanFilter(
            initial_state=initial_estimate,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track the trajectory
        estimates = []
        covariances = []

        for i in range(len(true_states)):
            # Get true position and add measurement noise
            true_pos = true_states[i, 0:3]
            measurement = true_pos + np.random.normal(0, DEFAULT_MEASUREMENT_NOISE_POS)

            # Kalman filter steps
            if i > 0:  # Skip prediction for first measurement
                kf.predict()
            kf.update(measurement)

            estimates.append(kf.get_state().copy())
            covariances.append(kf.get_covariance().copy())

        estimates = np.array(estimates)
        covariances = np.array(covariances)

        # Check that filter converges to true trajectory
        final_estimate = estimates[-1]
        final_true = true_states[-1]

        # Position should be close (within measurement noise bounds)
        pos_error = np.linalg.norm(final_estimate[0:3] - final_true[0:3])
        assert pos_error < 20.0  # Should be much better than initial 50m error

        # Velocity should be close to true velocity
        vel_error = np.linalg.norm(final_estimate[3:6] - final_true[3:6])
        assert vel_error < 4.0  # Allow some error due to measurement noise (increased tolerance for stochastic test)

        # Covariance should decrease over time
        initial_cov_trace = np.trace(initial_cov)
        final_cov_trace = np.trace(covariances[-1])
        assert final_cov_trace < initial_cov_trace * 0.5

    def test_tracking_with_process_noise(self):
        """Test tracking of trajectory with process noise."""
        initial_state = np.array([100.0, 50.0, 1000.0, 15.0, 8.0, 3.0])
        duration = 3.0

        # Create simulator with process noise
        simulator = TrajectorySimulator(
            initial_state=initial_state,
            process_noise_std=np.array([0.5, 0.5, 0.5]),  # Moderate process noise
            dt=DEFAULT_DT
        )

        # Generate true trajectory
        true_states, _ = simulator.simulate_trajectory(duration)

        # Initialize filter
        kf = TargetKalmanFilter(
            initial_state=initial_state,  # Perfect initial knowledge
            initial_covariance=np.diag([25.0, 25.0, 25.0, 5.0, 5.0, 2.0]),
            process_noise_std=np.array([0.5, 0.5, 0.5]),  # Matching process noise
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track trajectory
        rms_position_errors = []
        rms_velocity_errors = []

        for i in range(len(true_states)):
            true_pos = true_states[i, 0:3]
            measurement = true_pos + np.random.normal(0, DEFAULT_MEASUREMENT_NOISE_POS)

            if i > 0:
                kf.predict()
            kf.update(measurement)

            estimate = kf.get_state()
            pos_error = np.linalg.norm(estimate[0:3] - true_states[i, 0:3])
            vel_error = np.linalg.norm(estimate[3:6] - true_states[i, 3:6])

            rms_position_errors.append(pos_error)
            rms_velocity_errors.append(vel_error)

        # Check RMS errors are reasonable
        mean_pos_error = np.mean(rms_position_errors)
        mean_vel_error = np.mean(rms_velocity_errors)

        # Position error should be better than measurement noise
        assert mean_pos_error < np.mean(DEFAULT_MEASUREMENT_NOISE_POS) * 2.0
        # Velocity error should be reasonable
        assert mean_vel_error < 3.0

    def test_covariance_realism(self):
        """Test that filter covariance reflects true uncertainty."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        duration = 1.0

        # Run multiple Monte Carlo trials
        n_trials = 50
        final_covariances = []

        for trial in range(n_trials):
            # Create simulator with process noise
            simulator = TrajectorySimulator(
                initial_state=initial_state,
                process_noise_std=np.array([1.0, 1.0, 1.0]),
                dt=DEFAULT_DT
            )

            # Generate trajectory
            true_states, _ = simulator.simulate_trajectory(duration)

            # Initialize filter
            kf = TargetKalmanFilter(
                initial_state=initial_state,
                initial_covariance=np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),  # Low initial uncertainty
                process_noise_std=np.array([1.0, 1.0, 1.0]),
                measurement_noise_std=np.array([0.1, 0.1, 0.1]),  # Very accurate measurements
                measurement_type=MeasurementType.POSITION_ONLY,
                dt=DEFAULT_DT
            )

            # Track trajectory with accurate measurements
            for i in range(len(true_states)):
                measurement = true_states[i, 0:3] + np.random.normal(0, np.array([0.1, 0.1, 0.1]))  # Accurate measurement
                if i > 0:
                    kf.predict()
                kf.update(measurement)

            final_covariances.append(kf.get_covariance())

        # Compute average covariance
        avg_covariance = np.mean(final_covariances, axis=0)

        # The filter covariance should reflect the balance between process and measurement noise
        assert np.trace(avg_covariance) > 0.001  # Should be small but non-zero

        # Position variances should reflect measurement accuracy more than initial uncertainty
        # With accurate measurements, position uncertainty should be low
        assert avg_covariance[0, 0] < 1.0  # Should be better than initial 1.0
        assert avg_covariance[1, 1] < 1.0
        assert avg_covariance[2, 2] < 1.0

        # Velocity uncertainties should reflect process noise accumulation
        assert avg_covariance[3, 3] > 0.001  # Should be non-zero due to process noise
        assert avg_covariance[4, 4] > 0.001
        assert avg_covariance[5, 5] > 0.001

    def test_accelerated_motion_tracking(self):
        """Test tracking of accelerated motion."""
        initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

        simulator = TrajectorySimulator(
            initial_state=initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.zeros(3),  # Deterministic for testing
            dt=DEFAULT_DT
        )

        # Generate short trajectory
        duration = 0.5
        true_states, _ = simulator.simulate_trajectory(duration)

        # Initialize filter (constant velocity model for tracking)
        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=np.diag([10.0, 10.0, 10.0, 2.0, 2.0, 1.0]),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track trajectory
        for i in range(len(true_states)):
            measurement = true_states[i, 0:3] + np.random.normal(0, DEFAULT_MEASUREMENT_NOISE_POS * 0.1)  # Low noise

            if i > 0:
                kf.predict()
            kf.update(measurement)

        final_estimate = kf.get_state()
        final_true = true_states[-1]

        # Should still track reasonably well despite model mismatch
        pos_error = np.linalg.norm(final_estimate[0:3] - final_true[0:3])
        assert pos_error < 15.0  # Allow some error due to model mismatch

    def test_divergence_detection(self):
        """Test that filter doesn't diverge with reasonable parameters."""
        initial_state = np.array([0.0, 0.0, 0.0, 20.0, 10.0, 5.0])

        simulator = TrajectorySimulator(
            initial_state=initial_state,
            process_noise_std=np.zeros(3),  # Deterministic trajectory
            dt=DEFAULT_DT
        )

        # Generate long trajectory
        duration = 10.0  # 10 seconds
        true_states, _ = simulator.simulate_trajectory(duration)

        # Initialize filter
        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=np.eye(6) * 100.0,  # High initial uncertainty
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track trajectory
        max_covariance = 0.0

        for i in range(len(true_states)):
            measurement = true_states[i, 0:3] + np.random.normal(0, DEFAULT_MEASUREMENT_NOISE_POS)

            if i > 0:
                kf.predict()
            kf.update(measurement)

            # Check for divergence (covariance explosion)
            cov_trace = np.trace(kf.get_covariance())
            max_covariance = max(max_covariance, cov_trace)

            # Filter should not diverge
            assert cov_trace < 1e6, f"Filter diverged at step {i}"

        # Final covariance should be reasonable
        assert max_covariance < 1e4

    def test_measurement_outage_handling(self):
        """Test filter behavior during measurement outages."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])

        simulator = TrajectorySimulator(
            initial_state=initial_state,
            process_noise_std=np.zeros(3),
            dt=DEFAULT_DT
        )

        # Generate trajectory
        duration = 2.0
        true_states, _ = simulator.simulate_trajectory(duration)

        # Initialize filter
        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track with occasional measurement outages
        outage_period = 5  # Skip every 5th measurement

        for i in range(len(true_states)):
            if i > 0:
                kf.predict()

            # Skip measurement during outages
            if i % outage_period != 0:
                measurement = true_states[i, 0:3] + np.random.normal(0, DEFAULT_MEASUREMENT_NOISE_POS)
                kf.update(measurement)

        # Filter should still maintain reasonable estimates
        final_estimate = kf.get_state()
        final_true = true_states[-1]

        pos_error = np.linalg.norm(final_estimate[0:3] - final_true[0:3])
        vel_error = np.linalg.norm(final_estimate[3:6] - final_true[3:6])

        # Errors should be acceptable despite outages
        assert pos_error < 50.0
        assert vel_error < 5.0

    def test_realistic_radar_scenario(self):
        """Test realistic radar tracking scenario."""
        # Aircraft taking off: initial climb with acceleration
        initial_state = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 10.0])  # 50 m/s runway speed, 10 m/s climb

        simulator = TrajectorySimulator(
            initial_state=initial_state,
            motion_model=MotionModel.CONSTANT_ACCELERATION,
            process_noise_std=np.array([0.2, 0.2, 0.1]),  # Moderate turbulence
            dt=DEFAULT_DT
        )

        # Simulate 30 seconds of flight
        duration = 30.0
        true_states, _ = simulator.simulate_trajectory(duration)

        # Initialize filter with typical radar uncertainties
        kf = TargetKalmanFilter(
            initial_state=initial_state + np.array([100.0, 50.0, 20.0, 5.0, 3.0, 2.0]),  # Initial bias
            initial_covariance=np.diag([200.0, 200.0, 100.0, 20.0, 20.0, 10.0]),  # Radar uncertainties
            process_noise_std=np.array([0.3, 0.3, 0.2]),  # Expected turbulence
            measurement_noise_std=np.array([10.0, 10.0, 15.0]),  # Radar measurement noise
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=DEFAULT_DT
        )

        # Track with realistic radar update rate (2 Hz = every 5 steps at 0.1s dt)
        update_interval = 5

        for i in range(len(true_states)):
            kf.predict()

            # Radar measurement every update_interval steps
            if i % update_interval == 0:
                measurement = true_states[i, 0:3] + np.random.normal(0, np.array([10.0, 10.0, 15.0]))
                kf.update(measurement)

        # Check final tracking performance
        final_estimate = kf.get_state()
        final_true = true_states[-1]

        # Position error should be reasonable for radar tracking
        pos_error = np.linalg.norm(final_estimate[0:3] - final_true[0:3])
        assert pos_error < 100.0  # Within acceptable radar tracking accuracy

        # Velocity estimates should be reasonable
        vel_error = np.linalg.norm(final_estimate[3:6] - final_true[3:6])
        assert vel_error < 10.0

        # Covariance should reflect steady-state uncertainty
        final_cov = kf.get_covariance()
        pos_uncertainty = np.sqrt(np.trace(final_cov[0:3, 0:3]))
        assert pos_uncertainty < 50.0  # Reasonable steady-state position uncertainty