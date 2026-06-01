"""
Trajectory Simulator for Target Motion

This module provides realistic target trajectory generation with configurable
process noise for hit probability analysis scenarios.
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class MotionModel(Enum):
    """Enumeration of supported motion models."""
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"


class TrajectorySimulator:
    """
    Simulates realistic target trajectories with process noise.

    Supports different motion models and generates time-series of target states
    that can be used for tracking and hit probability analysis.
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY,
                 process_noise_std: Optional[np.ndarray] = None,
                 dt: float = 0.1):
        """
        Initialize the trajectory simulator.

        Args:
            initial_state: Initial target state [x, y, z, vx, vy, vz]
            motion_model: Type of motion model to use
            process_noise_std: Standard deviations for process noise [ax, ay, az]
                             If None, defaults to [0.1, 0.1, 0.1] m/s²
            dt: Time step for simulation (seconds)
        """
        if initial_state.shape != (6,):
            raise ValueError("Initial state must be 6-element array [x, y, z, vx, vy, vz]")

        self.initial_state = initial_state.copy()  # Store original state
        self.state = initial_state.copy()
        self.motion_model = motion_model
        self.dt = dt

        # Set default process noise if not provided
        if process_noise_std is None:
            self.process_noise_std = np.array([0.1, 0.1, 0.1])  # m/s²
        else:
            self.process_noise_std = np.array(process_noise_std)

        # Initialize acceleration state for constant acceleration model
        if self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            self.acceleration = np.zeros(3)  # [ax, ay, az]
            self.initial_acceleration = np.zeros(3)  # Store original acceleration

    def step(self) -> np.ndarray:
        """
        Perform one time step of trajectory simulation.

        Returns:
            Current state after the step [x, y, z, vx, vy, vz]
        """
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            self._step_constant_velocity()
        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            self._step_constant_acceleration()
        else:
            raise ValueError(f"Unsupported motion model: {self.motion_model}")

        return self.state.copy()

    def _step_constant_velocity(self):
        """Update state using constant velocity model with acceleration noise."""
        # Generate process noise (acceleration)
        noise = np.random.normal(0, self.process_noise_std)

        # Update velocity with acceleration noise
        self.state[3:6] += noise * self.dt

        # Update position with current velocity
        self.state[0:3] += self.state[3:6] * self.dt

    def _step_constant_acceleration(self):
        """Update state using constant acceleration model with jerk noise."""
        # Generate process noise (jerk - change in acceleration)
        jerk_noise = np.random.normal(0, self.process_noise_std * 0.1)  # Reduced noise for jerk

        # Update acceleration with jerk noise
        self.acceleration += jerk_noise * self.dt

        # Update velocity with current acceleration
        self.state[3:6] += self.acceleration * self.dt

        # Update position with current velocity
        self.state[0:3] += self.state[3:6] * self.dt

    def simulate_trajectory(self, duration: float, include_initial: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a complete trajectory for the specified duration.

        Args:
            duration: Total simulation time (seconds)
            include_initial: Whether to include initial state in the trajectory

        Returns:
            Tuple of (states, times) where:
            - states: Array of shape (n_steps, 6) with state history
            - times: Array of shape (n_steps,) with corresponding times
        """
        n_steps = int(duration / self.dt) + (1 if include_initial else 0)
        states = np.zeros((n_steps, 6))
        times = np.zeros(n_steps)

        # Reset to initial state
        original_state = self.state.copy()
        if self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            original_accel = self.acceleration.copy()

        current_time = 0.0

        for i in range(n_steps):
            if i == 0 and include_initial:
                states[i] = self.state.copy()
                times[i] = current_time
            else:
                states[i] = self.step()
                current_time += self.dt
                times[i] = current_time

        # Restore original state
        self.state = original_state
        if self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            self.acceleration = original_accel

        return states, times

    def get_current_state(self) -> np.ndarray:
        """Get the current state of the simulator."""
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        """Set the current state of the simulator."""
        if state.shape != (6,):
            raise ValueError("State must be 6-element array [x, y, z, vx, vy, vz]")
        self.state = state.copy()

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset the simulator to initial state."""
        if initial_state is not None:
            self.initial_state = initial_state.copy()
            self.set_state(initial_state)
        else:
            self.set_state(self.initial_state)

        if self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            self.acceleration = self.initial_acceleration.copy()