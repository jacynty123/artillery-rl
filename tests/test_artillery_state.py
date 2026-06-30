"""
Tests for ArtilleryState.

Covers default initialisation, the to_array feature vector (length, values,
history padding) and the __str__ representation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from rl_training.artillery_state import ArtilleryState
from rl_training.curriculum.scenario_generator import ScenarioParameters


def make_scenario(tracking_duration=10.0):
    return ScenarioParameters(
        range_m=1000.0,
        target_length=10.0,
        target_width=8.0,
        target_height=4.0,
        target_vx=0.0,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=tracking_duration,
        measurement_noise_std=np.array([0.5, 0.5, 0.5]),
        process_noise_std=np.array([0.5, 0.5, 0.5]),
    )


def test_defaults():
    s = ArtilleryState(make_scenario(), time_remaining=5.0, max_episode_steps=50)
    assert s.episode_step == 0
    assert s.current_hit_probability == 0.0
    assert s.covariance_trace == 1.0
    assert s.hp_history == [0.0] * 10
    assert s.target_position_est is None
    assert s.target_velocity_est is None


def test_to_array_length_and_values():
    s = ArtilleryState(
        make_scenario(tracking_duration=10.0), time_remaining=5.0, max_episode_steps=50
    )
    arr = s.to_array()
    assert arr.shape == (14,)
    assert arr.dtype == np.float32
    assert arr[0] == pytest.approx(0.0)  # hit probability
    assert arr[1] == pytest.approx(1.0)  # covariance trace
    assert arr[2] == pytest.approx(0.5)  # time_remaining / tracking_duration
    assert arr[3] == pytest.approx(0.0)  # episode_step / max_episode_steps


def test_to_array_reflects_updates():
    s = ArtilleryState(make_scenario(), time_remaining=5.0, max_episode_steps=50)
    s.current_hit_probability = 0.7
    s.covariance_trace = 0.3
    s.episode_step = 10
    arr = s.to_array()
    assert arr[0] == pytest.approx(0.7)
    assert arr[1] == pytest.approx(0.3)
    assert arr[3] == pytest.approx(10 / 50)


def test_history_padding():
    s = ArtilleryState(
        make_scenario(), time_remaining=5.0, max_episode_steps=50, hp_history_length=5
    )
    s.hp_history = [0.1, 0.2]  # shorter than hp_history_length
    arr = s.to_array()
    assert arr.shape == (4 + 5,)
    assert list(arr[4:]) == pytest.approx([0.0, 0.0, 0.0, 0.1, 0.2])


def test_str():
    s = ArtilleryState(make_scenario(), time_remaining=5.0, max_episode_steps=50)
    s.current_hit_probability = 0.5
    text = str(s)
    assert text.startswith("ArtilleryState(")
    assert "HP=0.500" in text
