"""
Training configuration constants for DQN curriculum learning.

Consolidates magic numbers from environment.py and train_dqn_curriculum.py
to improve maintainability without changing behavior.
"""

from dataclasses import dataclass


@dataclass
class TrainingConstants:
    """Configuration constants for training and rewards."""

    # Hit Probability Thresholds (from environment.py)
    HP_EXCELLENT_THRESHOLD: float = 0.85
    HP_GOOD_THRESHOLD: float = 0.70
    HP_MINIMUM_THRESHOLD: float = 0.60
    HP_FAIR_THRESHOLD: float = 0.50
    HP_ACCEPTABLE_THRESHOLD: float = 0.45

    # Reward Values (from environment.py)
    REWARD_EXCELLENT: float = 100.0
    REWARD_GOOD: float = 80.0
    REWARD_MINIMUM: float = 60.0
    REWARD_FAIR: float = 40.0
    REWARD_POOR: float = 20.0
    REWARD_FAILURE: float = -30.0

    # Hold Penalty (from environment.py)
    HOLD_PENALTY: float = -0.6

    # Time Efficiency Parameters (increased bonus for early firing)
    MAX_EPISODE_STEPS: int = 50
    TIME_EFFICIENCY_BASE: float = 30.0
    TIME_PENALTY_FACTOR: float = -100.0  # Much stronger penalty for late firing

    # Training Parameters (from train_dqn_curriculum.py)
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY_STEPS: int = 5000
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 20000
    TARGET_UPDATE_FREQ: int = 200
    LEARNING_STARTS: int = 500
    GAMMA: float = 0.99
    LEARNING_RATE: float = 1e-3

    # Evaluation Parameters
    EVAL_EPISODES_PER_SCENARIO: int = 10
    EVAL_INTERVAL: int = 100

    # Debug Flag
    DEBUG: bool = True

    # Hold Action Thresholds (from environment.py HOLD logic) - lowered to trigger opportunity costs earlier
    HP_HOLD_EXCELLENT_THRESHOLD: float = 0.75
    HP_HOLD_GOOD_THRESHOLD: float = 0.60
    HP_HOLD_ACCEPTABLE_THRESHOLD: float = 0.50

    # Opportunity Costs for Holding (increased to discourage waiting when HP is good)
    OPPORTUNITY_COST_EXCELLENT: float = -30.0
    OPPORTUNITY_COST_GOOD: float = -15.0
    OPPORTUNITY_COST_MINIMUM: float = -8.0

    # Hold Penalties (increased to encourage earlier firing)
    HOLD_PENALTY_BASE: float = -10.0


@dataclass
class RewardConfig:
    """All configurable reward parameters for ArtilleryFiringEnv and training."""
    firing_reward_high: float = 100.0
    time_penalty_factor: float = -100.0
    hold_penalty_high: float = -10.0
    reward_excellent: float = 100.0
    reward_good: float = 80.0
    reward_minimum: float = 60.0
    reward_fair: float = 40.0
    reward_poor: float = 20.0
    reward_failure: float = -30.0
    hold_penalty_base: float = -10.0
    opportunity_cost_excellent: float = -30.0
    opportunity_cost_good: float = -15.0
    opportunity_cost_minimum: float = -8.0