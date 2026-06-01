"""
Kalman Filter for Target Tracking

This module implements a Kalman filter for tracking targets using noisy measurements.
Supports position-only measurements with configurable process and measurement noise.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from scipy.linalg import expm


class MeasurementType(Enum):
    """Types of measurements supported by the Kalman filter."""
    POSITION_ONLY = "position_only"  # [x, y, z] measurements
    POSITION_VELOCITY = "position_velocity"  # [x, y, z, vx, vy, vz] measurements


class TargetKalmanFilter:
    """
    Kalman filter for target tracking with configurable measurement types.

    State vector: [x, y, z, vx, vy, vz] (position and velocity in 3D)
    Supports position-only measurements (radar-like) and full state measurements.
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_covariance: np.ndarray,
                 process_noise_std: np.ndarray,
                 measurement_noise_std: np.ndarray,
                 measurement_type: MeasurementType = MeasurementType.POSITION_ONLY,
                 dt: float = 0.1):
        """
        Initialize the Kalman filter.

        Args:
            initial_state: Initial state estimate [x, y, z, vx, vy, vz]
            initial_covariance: Initial state covariance matrix (6x6)
            process_noise_std: Process noise standard deviations [ax, ay, az] m/s²
            measurement_noise_std: Measurement noise standard deviations
                                   For POSITION_ONLY: [x, y, z] in meters
                                   For POSITION_VELOCITY: [x, y, z, vx, vy, vz]
            measurement_type: Type of measurements to expect
            dt: Time step for prediction (seconds)
        """
        if initial_state.shape != (6,):
            raise ValueError("Initial state must be 6-element array [x, y, z, vx, vy, vz]")

        if initial_covariance.shape != (6, 6):
            raise ValueError("Initial covariance must be 6x6 matrix")

        self.initial_state = initial_state.copy()
        self.initial_covariance = initial_covariance.copy()
        self.state = initial_state.copy()
        self.covariance = initial_covariance.copy()
        self.process_noise_std = np.array(process_noise_std)
        self.measurement_noise_std = np.array(measurement_noise_std)
        self.measurement_type = measurement_type
        self.dt = dt

        # Build measurement matrix based on measurement type
        self._build_measurement_matrix()

        # Precompute state transition matrix
        self.F = self._build_state_transition_matrix()

        # Precompute frequently used matrices
        self.HT = self.H.T

        # Pre-compute process noise covariance matrix
        self._build_process_noise_covariance()

        # Initialize diagnostic variables
        self.last_innovation = np.zeros(self.H.shape[0])
        self.last_innovation_cov = np.eye(self.H.shape[0])

    def _build_measurement_matrix(self):
        """Build the measurement matrix H based on measurement type."""
        if self.measurement_type == MeasurementType.POSITION_ONLY:
            # Measure only position: [x, y, z]
            self.H = np.zeros((3, 6))
            self.H[0, 0] = 1.0  # x measurement
            self.H[1, 1] = 1.0  # y measurement
            self.H[2, 2] = 1.0  # z measurement
        elif self.measurement_type == MeasurementType.POSITION_VELOCITY:
            # Measure full state: [x, y, z, vx, vy, vz]
            self.H = np.eye(6)
        else:
            raise ValueError(f"Unsupported measurement type: {self.measurement_type}")

    def _build_state_transition_matrix(self) -> np.ndarray:
        """Build the state transition matrix F."""
        F = np.eye(6)
        for i in range(3):
            F[i, i+3] = self.dt  # Position += velocity * dt
        return F

    def _build_process_noise_covariance(self):
        """Build the process noise covariance matrix Q using Van Loan's method."""
        # Use Van Loan's method for more accurate discrete process noise
        n = 6  # State dimension

        # Build continuous-time system matrices
        F_cont = np.zeros((n, n))  # Continuous F is zero for constant velocity
        G = np.zeros((n, 3))  # Input matrix
        for i in range(3):
            G[i, i] = 0.5 * self.dt**2  # Position affected by acceleration
            G[i+3, i] = self.dt         # Velocity affected by acceleration

        # Continuous process noise covariance
        Q_cont = np.diag(self.process_noise_std ** 2)

        # Van Loan's method for discretization
        A = np.zeros((2*n, 2*n))
        A[0:n, 0:n] = -F_cont.T
        A[0:n, n:2*n] = G @ Q_cont @ G.T
        A[n:2*n, n:2*n] = F_cont

        # Matrix exponential
        B = expm(A * self.dt)

        # Extract discrete process noise
        self.Q = B[n:2*n, n:2*n].T @ B[0:n, n:2*n]

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform prediction step (time update).

        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Predict state
        predicted_state = self.F @ self.state

        # Predict covariance
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q

        # Update internal state
        self.state = predicted_state
        self.covariance = predicted_covariance

        return predicted_state.copy(), predicted_covariance.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform update step (measurement update) using Joseph form for numerical stability.

        Args:
            measurement: Measurement vector (size depends on measurement_type)

        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Expected measurement
        expected_measurement = self.H @ self.state

        # Measurement residual (innovation)
        residual = measurement - expected_measurement
        self.last_innovation = residual.copy()

        # Measurement noise covariance R
        if self.measurement_type == MeasurementType.POSITION_ONLY:
            R = np.diag(self.measurement_noise_std ** 2)
        else:  # POSITION_VELOCITY
            R = np.diag(self.measurement_noise_std ** 2)

        # Innovation covariance
        S = self.H @ self.covariance @ self.HT + R
        self.last_innovation_cov = S.copy()

        # Kalman gain with numerical stability check
        try:
            K = self.covariance @ self.HT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Handle singular matrix by using pseudoinverse
            K = self.covariance @ self.HT @ np.linalg.pinv(S)

        # Update state
        updated_state = self.state + K @ residual

        # Update covariance using Joseph form (more numerically stable)
        I = np.eye(6)
        temp = I - K @ self.H
        updated_covariance = temp @ self.covariance @ temp.T + K @ R @ K.T

        # Ensure covariance remains symmetric and positive definite
        updated_covariance = (updated_covariance + updated_covariance.T) / 2.0

        # Update internal state
        self.state = updated_state
        self.covariance = updated_covariance

        return updated_state.copy(), updated_covariance.copy()

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.covariance.copy()

    def set_state(self, state: np.ndarray, covariance: Optional[np.ndarray] = None):
        """Set the current state and optionally covariance."""
        if state.shape != (6,):
            raise ValueError("State must be 6-element array [x, y, z, vx, vy, vz]")

        self.state = state.copy()
        if covariance is not None:
            if covariance.shape != (6, 6):
                raise ValueError("Covariance must be 6x6 matrix")
            self.covariance = covariance.copy()

    def reset(self, initial_state: Optional[np.ndarray] = None,
              initial_covariance: Optional[np.ndarray] = None):
        """Reset filter to initial state."""
        if initial_state is not None:
            self.initial_state = initial_state.copy()
        if initial_covariance is not None:
            self.initial_covariance = initial_covariance.copy()

        # Always reset to stored initial values
        self.state = self.initial_state.copy()
        self.covariance = self.initial_covariance.copy()

    def get_innovation(self) -> np.ndarray:
        """Get the last innovation (measurement residual)."""
        return self.last_innovation.copy()

    def get_innovation_covariance(self) -> np.ndarray:
        """Get the last innovation covariance."""
        return self.last_innovation_cov.copy()

    def get_filter_status(self) -> Dict[str, Any]:
        """Get filter health and diagnostic information."""
        return {
            'condition_number': np.linalg.cond(self.covariance),
            'nees': self._compute_nees(),
            'nis': self._compute_nis(),
            'is_stable': self._check_stability(),
            'covariance_trace': np.trace(self.covariance),
            'max_covariance_eigenvalue': np.max(np.linalg.eigvals(self.covariance))
        }

    def _compute_nees(self) -> float:
        """Compute Normalized Estimation Error Squared (NEES) if true state available."""
        # NEES would require true state - for now return 0
        # In a real implementation, this would compare estimate to true state
        return 0.0

    def _compute_nis(self) -> float:
        """Compute Normalized Innovation Squared (NIS)."""
        if np.linalg.det(self.last_innovation_cov) > 1e-12:
            nis = self.last_innovation.T @ np.linalg.inv(self.last_innovation_cov) @ self.last_innovation
            return float(nis)
        return 0.0

    def _check_stability(self) -> bool:
        """Check if filter appears stable."""
        # Check covariance properties
        eigenvals = np.linalg.eigvals(self.covariance)
        min_eigenval = np.min(np.real(eigenvals))

        # Covariance should be positive definite
        if min_eigenval < 1e-12:
            return False

        # Condition number should not be too large
        cond_num = np.linalg.cond(self.covariance)
        if cond_num > 1e12:
            return False

        return True
