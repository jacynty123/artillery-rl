"""
Branch / edge-case tests for the artillery firing environment helpers.

Targets the static scenario factory, EnvironmentState.to_array, dynamic
max-step calculation, cache keys, expected-HP lookup, scenario classifiers,
the firing-reward tiers, render and close.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from rl_training.environment import (
    ArtilleryFiringEnv,
    EnvironmentState,
    get_static_scenario,
)
from rl_training.curriculum.scenario_generator import ScenarioParameters


def make_scenario(range_m=800.0, vx=0.0, vy=0.0, meas=0.5, tracking=10.0):
    return ScenarioParameters(
        range_m=range_m,
        target_length=10.0,
        target_width=8.0,
        target_height=4.0,
        target_vx=vx,
        target_vy=vy,
        target_vz=0.0,
        tracking_duration=tracking,
        measurement_noise_std=np.array([meas, meas, meas]),
        process_noise_std=np.array([0.5, 0.5, 0.5]),
    )


@pytest.fixture
def env():
    return ArtilleryFiringEnv(max_episode_steps=50)


def test_get_static_scenario():
    s = get_static_scenario()
    assert s.range_m == 800.0
    assert s.target_vx == 0.0


def test_environment_state_to_array():
    s = EnvironmentState(
        scenario=make_scenario(1000.0),
        time_remaining=5.0,
        current_hit_probability=0.5,
        episode_step=3,
    )
    arr = s.to_array()
    assert arr.shape == (14,)
    assert arr.dtype == np.float32


def test_calculate_max_steps(env):
    assert env._calculate_max_steps(make_scenario(3000.0, vx=-50.0, meas=0.2)) == 25
    assert env._calculate_max_steps(make_scenario(4000.0, vx=-80.0, meas=0.2)) == 40
    assert env._calculate_max_steps(make_scenario(3200.0, vx=0.0)) == 100
    assert env._calculate_max_steps(make_scenario(2600.0, vx=0.0)) == 90
    assert env._calculate_max_steps(make_scenario(800.0, vx=0.0)) == 40


def test_get_cache_key_deterministic(env):
    s = make_scenario(800.0)
    k1 = env._get_cache_key(s, seed=1)
    k2 = env._get_cache_key(s, seed=1)
    k3 = env._get_cache_key(s, seed=2)
    assert k1 == k2
    assert k1 != k3
    assert isinstance(k1, str) and len(k1) == 32


def test_get_expected_max_hp(env):
    assert env._get_expected_max_hp(500.0) == 0.95
    assert env._get_expected_max_hp(1500.0) == 0.75
    assert env._get_expected_max_hp(2500.0) == 0.55
    assert env._get_expected_max_hp(3500.0) == 0.35
    assert env._get_expected_max_hp(4500.0) == 0.20


def test_is_fast_approaching_scenario(env):
    env.current_state = EnvironmentState(
        scenario=make_scenario(3000.0, vx=-50.0, meas=0.2),
        time_remaining=10.0,
        current_hit_probability=0.0,
        episode_step=0,
    )
    assert env._is_fast_approaching_scenario()

    env.current_state = EnvironmentState(
        scenario=get_static_scenario(),
        time_remaining=10.0,
        current_hit_probability=0.0,
        episode_step=0,
    )
    assert not env._is_fast_approaching_scenario()

    env.current_state = None
    assert not env._is_fast_approaching_scenario()


def test_is_hp_plateau(env):
    env.current_state = None
    assert not env._is_hp_plateau()

    short = EnvironmentState(
        scenario=make_scenario(1000.0),
        time_remaining=5.0,
        current_hit_probability=0.6,
        episode_step=5,
    )
    short.hp_history = [0.6] * 3  # not enough history
    env.current_state = short
    assert not env._is_hp_plateau()

    flat = EnvironmentState(
        scenario=make_scenario(1000.0),
        time_remaining=5.0,
        current_hit_probability=0.6,
        episode_step=20,
    )
    flat.hp_history = [0.6] * 12  # enough history, no improvement, HP healthy
    env.current_state = flat
    assert env._is_hp_plateau()


def test_calculate_firing_reward_tiers(env):
    env.current_state = EnvironmentState(
        scenario=get_static_scenario(),
        time_remaining=5.0,
        current_hit_probability=0.0,
        episode_step=0,
    )
    env.current_state.hp_history = []  # avoid plateau bonus

    r_exc = env._calculate_firing_reward(0.9, episode_step=5)
    r_good = env._calculate_firing_reward(0.65, episode_step=5)
    r_min = env._calculate_firing_reward(0.55, episode_step=5)
    r_fail = env._calculate_firing_reward(0.05, episode_step=5)

    assert r_exc > r_good > r_min > r_fail


def test_render(env, capsys):
    env.current_state = None
    env.render()
    assert "not initialized" in capsys.readouterr().out.lower()

    env.current_state = EnvironmentState(
        scenario=make_scenario(1000.0),
        time_remaining=5.0,
        current_hit_probability=0.5,
        episode_step=2,
    )
    env.render()
    assert "Range" in capsys.readouterr().out


def test_close(env):
    assert env.close() is None
