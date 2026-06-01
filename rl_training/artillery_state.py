"""
Defines the state of the artillery environment.
"""
import numpy as np
from typing import List, Optional
from rl_training.curriculum.curriculum_scenarios import ScenarioParameters

class ArtilleryState:
    """Represents the state of the artillery firing environment at a given time."""

    def __init__(
        self,
        scenario: ScenarioParameters,
        time_remaining: float,
        max_episode_steps: int,
        hp_history_length: int = 10,
    ):
        self.scenario = scenario
        self.time_remaining = time_remaining
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps
        self.hp_history_length = hp_history_length

        # State features
        self.current_hit_probability: float = 0.0
        self.covariance_trace: float = 1.0  # Normalized
        self.hp_history: List[float] = [0.0] * hp_history_length
        self.target_position_est: Optional[np.ndarray] = None
        self.target_velocity_est: Optional[np.ndarray] = None

    def to_array(self) -> np.ndarray:
        """Convert state to a NumPy array for the agent."""
        # Normalize features to be roughly in the [0, 1] range
        time_feature = self.time_remaining / self.scenario.tracking_duration
        step_feature = self.episode_step / self.max_episode_steps
        
        # Pad history if it's shorter than required
        padded_history = self.hp_history
        if len(padded_history) < self.hp_history_length:
            padded_history = [0.0] * (self.hp_history_length - len(padded_history)) + padded_history

        features = [
            self.current_hit_probability,
            self.covariance_trace,
            time_feature,
            step_feature,
        ] + padded_history
        
        return np.array(features, dtype=np.float32)

    def __str__(self):
        return (
            f"ArtilleryState(HP={self.current_hit_probability:.3f}, "
            f"CovTrace={self.covariance_trace:.3f}, "
            f"TimeLeft={self.time_remaining:.2f}s, Step={self.episode_step})"
        )
