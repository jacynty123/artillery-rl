"""
Unit tests for TargetKalmanFilter class.
"""

import pytest
import numpy as np
from src.kalman_filter import TargetKalmanFilter, MeasurementType

# Test constants
DEFAULT_DT = 0.1
KINEMATIC_ATOL = 1e-10  # Absolute tolerance for kinematic tests
NUMERICAL_ATOL = 1e-3   # Absolute tolerance for numerical integration
DEFAULT_PROCESS_NOISE = np.array([0.1, 0.1, 0.1])
DEFAULT_MEASUREMENT_NOISE_POS = np.array([5.0, 5.0, 5.0])  # Position-only measurement noise
DEFAULT_MEASUREMENT_NOISE_FULL = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])  # Full state measurement noise


class TestTargetKalmanFilter:
    """Test cases for TargetKalmanFilter class."""

    def test_initialization_position_only(self):
        """Test proper initialization with position-only measurements."""
        initial_state = np.array([100.0, 50.0, 1000.0, 20.0, 15.0, 5.0])
        initial_cov = np.eye(6) * 10.0
        dt = DEFAULT_DT

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY,
            dt=dt
        )

        assert np.allclose(kf.get_state(), initial_state)
        assert np.allclose(kf.get_covariance(), initial_cov)
        assert kf.dt == dt
        assert kf.measurement_type == MeasurementType.POSITION_ONLY
        assert kf.H.shape == (3, 6)  # Position-only measurement matrix

    def test_initialization_position_velocity(self):
        """Test proper initialization with full state measurements."""
        initial_state = np.array([100.0, 50.0, 1000.0, 20.0, 15.0, 5.0])
        initial_cov = np.eye(6) * 10.0

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_FULL,
            measurement_type=MeasurementType.POSITION_VELOCITY
        )

        assert kf.measurement_type == MeasurementType.POSITION_VELOCITY
        assert kf.H.shape == (6, 6)  # Full state measurement matrix
        assert np.allclose(kf.H, np.eye(6))

    def test_initialization_invalid_state(self):
        """Test that invalid initial state raises error."""
        initial_cov = np.eye(6)

        with pytest.raises(ValueError, match="Initial state must be 6-element array"):
            TargetKalmanFilter(
                initial_state=np.array([1.0, 2.0, 3.0]),  # Only 3 elements
                initial_covariance=initial_cov,
                process_noise_std=DEFAULT_PROCESS_NOISE,
                measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS
            )

    def test_initialization_invalid_covariance(self):
        """Test that invalid initial covariance raises error."""
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Initial covariance must be 6x6 matrix"):
            TargetKalmanFilter(
                initial_state=initial_state,
                initial_covariance=np.eye(3),  # 3x3 instead of 6x6
                process_noise_std=DEFAULT_PROCESS_NOISE,
                measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS
            )

    def test_predict_constant_velocity(self):
        """Test prediction step with constant velocity motion."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 2.0])  # Moving with constant velocity
        initial_cov = np.eye(6) * 1.0
        dt = DEFAULT_DT

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=np.zeros(3),  # No process noise for deterministic test
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            dt=dt
        )

        # Prediction step
        pred_state, pred_cov = kf.predict()

        # Position should change by velocity * dt
        expected_position = initial_state[0:3] + initial_state[3:6] * dt
        assert np.allclose(pred_state[0:3], expected_position, atol=KINEMATIC_ATOL)

        # Velocity should remain the same (no process noise)
        assert np.allclose(pred_state[3:6], initial_state[3:6], atol=KINEMATIC_ATOL)

        # Covariance should increase due to process noise (even if zero, structure should be correct)
        assert pred_cov.shape == (6, 6)

    def test_predict_with_process_noise(self):
        """Test prediction step with process noise."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        initial_cov = np.eye(6) * 1.0
        process_noise_std = np.array([1.0, 1.0, 1.0])
        dt = DEFAULT_DT

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=process_noise_std,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            dt=dt
        )

        # Store initial covariance
        initial_cov_trace = np.trace(initial_cov)

        # Prediction step
        pred_state, pred_cov = kf.predict()

        # Covariance should increase due to process noise
        pred_cov_trace = np.trace(pred_cov)
        assert pred_cov_trace > initial_cov_trace

        # Check that process noise covariance Q is properly constructed
        # For position components, variance should increase by (dt^2/2)^2 * sigma_a^2
        dt2 = dt * dt
        expected_pos_var_increase = (dt2 * dt2 / 4.0) * process_noise_std[0]**2
        assert pred_cov[0, 0] > initial_cov[0, 0] + expected_pos_var_increase * 0.9  # Allow some tolerance

    def test_update_position_only(self):
        """Test update step with position-only measurement."""
        initial_state = np.array([100.0, 50.0, 1000.0, 10.0, 5.0, 2.0])
        initial_cov = np.eye(6) * 100.0  # High uncertainty

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY
        )

        # True position (slightly different from estimate)
        true_position = np.array([105.0, 55.0, 1005.0])
        measurement = true_position + np.array([1.0, -1.0, 2.0])  # Noisy measurement

        # Update step
        updated_state, updated_cov = kf.update(measurement)

        # State should move closer to measurement
        assert np.allclose(updated_state[0:3], measurement, atol=10.0)  # Should be close but not exact due to Kalman gain

        # Covariance should decrease (more certainty)
        initial_cov_trace = np.trace(initial_cov)
        updated_cov_trace = np.trace(updated_cov)
        assert updated_cov_trace < initial_cov_trace

        # Velocity components should not change significantly with position-only measurements
        # (since we don't have velocity measurements)
        assert np.allclose(updated_state[3:6], initial_state[3:6], atol=1.0)

    def test_update_position_velocity(self):
        """Test update step with full state measurement."""
        initial_state = np.array([100.0, 50.0, 1000.0, 10.0, 5.0, 2.0])
        initial_cov = np.eye(6) * 100.0

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_FULL,
            measurement_type=MeasurementType.POSITION_VELOCITY
        )

        # True full state
        true_state = np.array([105.0, 55.0, 1005.0, 12.0, 6.0, 3.0])
        measurement = true_state + np.array([1.0, -1.0, 2.0, 0.5, -0.5, 1.0])  # Noisy measurement

        # Update step
        updated_state, updated_cov = kf.update(measurement)

        # State should move closer to measurement
        assert np.allclose(updated_state, measurement, atol=10.0)

        # Covariance should decrease significantly
        initial_cov_trace = np.trace(initial_cov)
        updated_cov_trace = np.trace(updated_cov)
        assert updated_cov_trace < initial_cov_trace * 0.5  # Should be more certain

    def test_predict_update_cycle(self):
        """Test a complete predict-update cycle."""
        initial_state = np.array([0.0, 0.0, 0.0, 10.0, 5.0, 0.0])
        initial_cov = np.eye(6) * 1.0

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=np.zeros(3),  # Deterministic for testing
            measurement_noise_std=np.zeros(3),  # Perfect measurements
            measurement_type=MeasurementType.POSITION_ONLY
        )

        # Predict step
        kf.predict()

        # Measurement of predicted position (perfect measurement)
        expected_position = initial_state[0:3] + initial_state[3:6] * DEFAULT_DT
        measurement = expected_position.copy()

        # Update step
        updated_state, updated_cov = kf.update(measurement)

        # Should converge to expected position
        assert np.allclose(updated_state[0:3], expected_position, atol=KINEMATIC_ATOL)
        assert np.allclose(updated_state[3:6], initial_state[3:6], atol=KINEMATIC_ATOL)

    def test_set_state_and_reset(self):
        """Test state setting and reset functionality."""
        initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        new_state = np.array([100.0, 50.0, 25.0, 15.0, 8.0, 3.0])
        new_cov = np.eye(6) * 5.0

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=np.eye(6),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS
        )

        # Change state and covariance
        kf.set_state(new_state, new_cov)
        assert np.allclose(kf.get_state(), new_state)
        assert np.allclose(kf.get_covariance(), new_cov)

        # Reset to original
        kf.reset()
        assert np.allclose(kf.get_state(), initial_state)

        # Reset to different state
        kf.reset(new_state, new_cov)
        assert np.allclose(kf.get_state(), new_state)
        assert np.allclose(kf.get_covariance(), new_cov)

    def test_set_state_invalid(self):
        """Test that setting invalid state raises error."""
        kf = TargetKalmanFilter(
            initial_state=np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
            initial_covariance=np.eye(6),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS
        )

        with pytest.raises(ValueError, match="State must be 6-element array"):
            kf.set_state(np.array([1.0, 2.0, 3.0]))  # Only 3 elements

    def test_measurement_matrix_construction(self):
        """Test that measurement matrices are constructed correctly."""
        # Position-only
        kf_pos = TargetKalmanFilter(
            initial_state=np.zeros(6),
            initial_covariance=np.eye(6),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY
        )

        # Should be 3x6 with identity in position blocks
        expected_H_pos = np.zeros((3, 6))
        expected_H_pos[0, 0] = 1.0
        expected_H_pos[1, 1] = 1.0
        expected_H_pos[2, 2] = 1.0
        assert np.allclose(kf_pos.H, expected_H_pos)

        # Position-velocity
        kf_full = TargetKalmanFilter(
            initial_state=np.zeros(6),
            initial_covariance=np.eye(6),
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_FULL,
            measurement_type=MeasurementType.POSITION_VELOCITY
        )

        # Should be 6x6 identity
        assert np.allclose(kf_full.H, np.eye(6))

    def test_process_noise_covariance(self):
        """Test that process noise covariance matrix is constructed correctly using Van Loan's method."""
        process_noise_std = np.array([1.0, 2.0, 3.0])
        dt = 0.1

        kf = TargetKalmanFilter(
            initial_state=np.zeros(6),
            initial_covariance=np.eye(6),
            process_noise_std=process_noise_std,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            dt=dt
        )

        Q = kf.Q
        assert Q.shape == (6, 6)

        # Check that Q is symmetric and positive semi-definite
        assert np.allclose(Q, Q.T), "Process noise covariance should be symmetric"
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals >= -1e-12), "Process noise covariance should be positive semi-definite"

        # Check that diagonal elements are positive (variances)
        assert np.all(np.diag(Q) > 0), "Diagonal elements should be positive"

        # Check that off-diagonal elements exist (cross-correlations)
        assert Q[0, 3] != 0.0, "Should have position-velocity correlation"
        assert Q[3, 0] == Q[0, 3], "Matrix should be symmetric"

    def test_enhanced_features(self):
        """Test enhanced features: diagnostics and numerical stability."""
        initial_state = np.array([100.0, 50.0, 1000.0, 10.0, 5.0, 2.0])
        initial_cov = np.eye(6) * 10.0

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=DEFAULT_MEASUREMENT_NOISE_POS,
            measurement_type=MeasurementType.POSITION_ONLY
        )

        # Test initial diagnostic values
        innovation = kf.get_innovation()
        assert innovation.shape == (3,)  # Position-only measurement
        assert np.allclose(innovation, np.zeros(3))

        # Test filter status
        status = kf.get_filter_status()
        assert 'condition_number' in status
        assert 'nees' in status
        assert 'nis' in status
        assert 'is_stable' in status
        assert status['is_stable']  # Should be stable initially

        # Perform update to generate innovation
        measurement = np.array([105.0, 55.0, 1005.0])
        kf.update(measurement)

        # Check innovation is non-zero after update
        innovation = kf.get_innovation()
        assert not np.allclose(innovation, np.zeros(3))

        # Test NIS computation
        status = kf.get_filter_status()
        assert status['nis'] >= 0.0  # NIS should be non-negative

    def test_numerical_stability(self):
        """Test numerical stability with challenging conditions."""
        # Start with high uncertainty
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        initial_cov = np.eye(6) * 1000.0  # Very high uncertainty

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise_std=DEFAULT_PROCESS_NOISE,
            measurement_noise_std=np.array([0.01, 0.01, 0.01]),  # Very accurate measurements
            measurement_type=MeasurementType.POSITION_ONLY
        )

        # Perform many predict-update cycles
        for i in range(100):
            kf.predict()
            measurement = np.array([0.0, 0.0, 0.0])  # Perfect measurements
            kf.update(measurement)

            # Check stability
            status = kf.get_filter_status()
            assert status['is_stable'], f"Filter became unstable at step {i}"

            # Covariance should remain positive definite
            cov = kf.get_covariance()
            eigenvals = np.linalg.eigvals(cov)
            assert np.all(np.real(eigenvals) > 1e-12), f"Non-positive definite covariance at step {i}"

        # Final covariance should be much smaller than initial
        final_cov = kf.get_covariance()
        assert np.trace(final_cov) < np.trace(initial_cov) * 0.1
