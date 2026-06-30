"""
Tests for evaluation utilities.

Uses lightweight fakes for the environment, Q-network and curriculum so the
evaluation logic can be exercised without torch models or the real heavy env.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from rl_training.utils.evaluation_utils import (
    rollout,
    evaluate_on_scenario,
    evaluate_curriculum,
    check_phase_completion,
)
from rl_training.curriculum.scenario_generator import ScenarioParameters


class FakeEnv:
    """Minimal env: terminates after n_steps and reports a fixed hit probability."""

    def __init__(self, n_steps=3, hp=0.8, max_episode_steps=10):
        self.n_steps = n_steps
        self.hp = hp
        self.max_episode_steps = max_episode_steps
        self._step = 0

    def reset(self, scenario_override=None):
        self._step = 0
        return np.zeros(14, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= self.n_steps
        info = {"hit_probability": self.hp}
        return np.zeros(14, dtype=np.float32), 1.0, terminated, False, info


class FakeQNet:
    """Returns a one-hot Q vector so argmax picks a fixed action."""

    def __init__(self, action=1):
        self.action = action

    def __call__(self, state_tensor):
        q = torch.zeros((1, 2))
        q[0, self.action] = 1.0
        return q


def make_scenario(name="T"):
    s = ScenarioParameters(
        range_m=1000.0,
        target_length=10.0,
        target_width=8.0,
        target_height=4.0,
        target_vx=0.0,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=10.0,
        measurement_noise_std=np.array([0.5, 0.5, 0.5]),
        process_noise_std=np.array([0.5, 0.5, 0.5]),
    )
    s.name = name
    return s


class FakeCurriculum:
    def __init__(self, scenarios):
        self._scenarios = scenarios
        self.difficulty = {s.name: "easy" for s in scenarios}
        self.phase = 1

    def get_all_scenarios(self):
        return self._scenarios

    def get_phase_requirements(self):
        return {
            "min_avg_hp": 0.5,
            "min_hp_per_scenario": 0.4,
            "description": "test phase",
        }


def test_rollout_fires():
    out = rollout(FakeEnv(n_steps=3, hp=0.8), FakeQNet(action=1), make_scenario(), "cpu")
    assert out["steps"] == 3
    assert out["reward"] == pytest.approx(3.0)
    assert out["hp"] == pytest.approx(0.8)
    assert out["fired_at_step"] == 1


def test_rollout_holds():
    out = rollout(FakeEnv(n_steps=2, hp=0.5), FakeQNet(action=0), make_scenario(), "cpu")
    assert out["fired_at_step"] is None
    assert out["steps"] == 2


def test_evaluate_on_scenario():
    res = evaluate_on_scenario(
        FakeEnv(n_steps=3, hp=0.8, max_episode_steps=10),
        FakeQNet(action=1),
        make_scenario(),
        "cpu",
        n_episodes=2,
    )
    assert res["hp"] == pytest.approx(0.8)
    assert res["steps"] == pytest.approx(3.0)
    assert res["range_safe"]  # firing range (1000m) > 200m
    assert "firing_range" in res


def test_evaluate_curriculum():
    curr = FakeCurriculum([make_scenario("A"), make_scenario("B")])
    out = evaluate_curriculum(
        FakeEnv(3, 0.8, 10), FakeQNet(action=1), curr, "cpu", verbose=False
    )
    assert len(out["results"]) == 2
    assert out["avg_hp"] == pytest.approx(0.8)
    assert out["min_hp"] == pytest.approx(0.8)
    assert out["all_safe"]


def test_check_phase_completion_pass():
    curr = FakeCurriculum([make_scenario("A")])
    eval_results = {
        "avg_hp": 0.8,
        "min_hp": 0.7,
        "all_safe": True,
        "results": [{"firing_step": 5.0}, {"firing_step": 6.0}],
    }
    assert check_phase_completion(curr, eval_results)


def test_check_phase_completion_fail_min_hp():
    curr = FakeCurriculum([make_scenario("A")])
    eval_results = {
        "avg_hp": 0.8,
        "min_hp": 0.1,
        "all_safe": True,
        "results": [{"firing_step": 5.0}],
    }
    assert not check_phase_completion(curr, eval_results)
