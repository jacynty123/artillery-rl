"""
Training monitoring and logging utilities for DQN curriculum learning.

Provides episode and scenario-level statistics without changing training behavior.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class TrainingMonitor:
    """Monitors training progress and provides statistics."""

    def __init__(self):
        self.episode_stats = defaultdict(list)
        self.scenario_stats = defaultdict(lambda: defaultdict(list))
        self.phase_stats = defaultdict(list)

    def record_episode(self, scenario_name: str, reward: float, final_hp: float,
                      steps: int, fired: bool, phase: int = None):
        """Record statistics for an episode."""
        self.episode_stats['rewards'].append(reward)
        self.episode_stats['hit_probabilities'].append(final_hp)
        self.episode_stats['steps'].append(steps)
        self.episode_stats['fired'].append(fired)

        # Per-scenario tracking
        self.scenario_stats[scenario_name]['rewards'].append(reward)
        self.scenario_stats[scenario_name]['hps'].append(final_hp)
        self.scenario_stats[scenario_name]['steps'].append(steps)

        if phase is not None:
            self.phase_stats[phase].append({
                'scenario': scenario_name,
                'reward': reward,
                'hp': final_hp,
                'steps': steps,
                'fired': fired
            })

    def get_recent_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary statistics for recent episodes."""
        recent_rewards = self.episode_stats['rewards'][-window:]
        recent_hps = self.episode_stats['hit_probabilities'][-window:]
        recent_steps = self.episode_stats['steps'][-window:]

        return {
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'avg_hp': np.mean(recent_hps) if recent_hps else 0.0,
            'avg_steps': np.mean(recent_steps) if recent_steps else 0.0,
            'total_episodes': len(self.episode_stats['rewards'])
        }

    def get_scenario_summary(self, scenario_name: str) -> Dict[str, float]:
        """Get statistics for a specific scenario."""
        if scenario_name not in self.scenario_stats:
            return {}

        rewards = self.scenario_stats[scenario_name]['rewards']
        hps = self.scenario_stats[scenario_name]['hps']
        steps = self.scenario_stats[scenario_name]['steps']

        return {
            'avg_reward': np.mean(rewards),
            'avg_hp': np.mean(hps),
            'avg_steps': np.mean(steps),
            'std_hp': np.std(hps),
            'episodes': len(rewards)
        }

    def get_phase_summary(self, phase: int) -> Dict[str, Any]:
        """Get summary for a training phase."""
        if phase not in self.phase_stats:
            return {}

        phase_data = self.phase_stats[phase]
        rewards = [ep['reward'] for ep in phase_data]
        hps = [ep['hp'] for ep in phase_data]

        return {
            'episodes': len(phase_data),
            'avg_reward': np.mean(rewards),
            'avg_hp': np.mean(hps),
            'min_hp': np.min(hps),
            'max_hp': np.max(hps)
        }

    def reset(self):
        """Reset all statistics."""
        self.episode_stats.clear()
        self.scenario_stats.clear()
        self.phase_stats.clear()