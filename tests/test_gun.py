"""
Tests for the gun abstraction (src/gun.py).

Verifies that GunParameters carries a position, that solve_firing_solution threads the
gun's parameters into find_optimal_firing_angles (and reproduces it for a gun at the
origin), and that hit_probability_for_gun is correct (origin-equivalent and
translation-invariant in the gun frame).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch

from src.gun import (
    GunParameters,
    FiringSolution,
    Battery,
    solve_firing_solution,
    hit_probability_for_gun,
)
from src.find_optimal_firing_angles import find_optimal_firing_angles
from src.hit_probability import HitProbabilityCalculator


_UNC = np.diag(np.array([10, 0.005, 0.005, 10, 10, 5, 5, 2, 2], dtype=float) ** 2)


def _calc():
    return HitProbabilityCalculator(
        projectile_velocity=800.0, target_dimensions=(5.0, 2.0, 2.0), ammo_type="tpt"
    )


# ---------------------------------------------------------------------------
# GunParameters
# ---------------------------------------------------------------------------

def test_gun_defaults():
    g = GunParameters()
    assert g.gun_id == 0
    assert np.allclose(g.position, [0.0, 0.0, 0.0])
    assert g.position.shape == (3,)
    assert g.ammo_type == "tpt"


def test_default_at_origin():
    g = GunParameters.default_at_origin()
    assert np.allclose(g.position, np.zeros(3))
    assert g.ammo_type == "tpt"


def test_position_coerced_to_array():
    g = GunParameters(position=[100.0, 0.0, 0.0])
    assert isinstance(g.position, np.ndarray)
    assert g.position.shape == (3,)
    assert np.allclose(g.position, [100.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# solve_firing_solution
# ---------------------------------------------------------------------------

def test_solve_firing_solution_passes_gun_params():
    gun = GunParameters(position=[10.0, 0.0, 0.0], muzzle_velocity=900.0, ammo_type="tpt")
    with patch("src.gun.find_optimal_firing_angles",
               return_value=(0.1, 0.0, 1.5, np.array([1000.0, 0.0, 0.0]), 0.3)) as m:
        sol = solve_firing_solution(gun, [1000.0, 0.0, 0.0], [0.0, 0.0, 0.0], max_time=20.0)
    args, kwargs = m.call_args
    assert np.allclose(args[0], [10.0, 0.0, 0.0])  # shooter_pos = gun.position
    assert args[3] == 900.0                         # projectile_speed = muzzle_velocity
    assert args[4] == "tpt"                          # ammo_type
    assert kwargs["max_time"] == 20.0
    assert isinstance(sol, FiringSolution)
    assert sol.elevation == 0.1
    assert sol.feasible


def test_solve_firing_solution_above_max_range():
    gun = GunParameters(max_range_m=500.0)
    with patch("src.gun.find_optimal_firing_angles",
               return_value=(0.0, 0.0, 1.0, np.array([1000.0, 0.0, 0.0]), 0.5)):
        sol = solve_firing_solution(gun, [1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not sol.feasible
    assert "range" in sol.infeasible_reason


def test_solve_firing_solution_below_min_range():
    gun = GunParameters(min_range_m=500.0)
    with patch("src.gun.find_optimal_firing_angles",
               return_value=(0.0, 0.0, 1.0, np.array([100.0, 0.0, 0.0]), 0.5)):
        sol = solve_firing_solution(gun, [100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not sol.feasible
    assert "min" in sol.infeasible_reason


def test_solve_firing_solution_azimuth_outside_traverse():
    gun = GunParameters(traverse_limits=(np.radians(-10.0), np.radians(10.0)))
    with patch("src.gun.find_optimal_firing_angles",
               return_value=(0.0, np.radians(45.0), 1.0, np.array([1000.0, 500.0, 0.0]), 0.5)):
        sol = solve_firing_solution(gun, [1000.0, 500.0, 0.0], [0.0, 0.0, 0.0])
    assert not sol.feasible
    assert "azimuth" in sol.infeasible_reason


def test_solve_firing_solution_elevation_outside_limits():
    gun = GunParameters(elevation_limits=(np.radians(0.0), np.radians(10.0)))
    with patch("src.gun.find_optimal_firing_angles",
               return_value=(np.radians(45.0), 0.0, 1.0, np.array([1000.0, 0.0, 0.0]), 0.5)):
        sol = solve_firing_solution(gun, [1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not sol.feasible
    assert "elevation" in sol.infeasible_reason


def test_solve_firing_solution_matches_find_optimal_at_origin():
    """A gun at the origin reproduces find_optimal_firing_angles exactly."""
    gun = GunParameters(position=[0.0, 0.0, 0.0], muzzle_velocity=1000.0, ammo_type="tpt")
    target = np.array([1000.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    sol = solve_firing_solution(gun, target, vel)
    elev, azim, t, hit, min_dist = find_optimal_firing_angles(
        np.array([0.0, 0.0, 0.0]), target, vel, 1000.0, "tpt"
    )
    assert np.isclose(sol.elevation, elev, atol=1e-6)
    assert np.isclose(sol.azimuth, azim, atol=1e-6)
    assert np.isclose(sol.miss_distance, min_dist, atol=1e-6)
    assert sol.feasible


# ---------------------------------------------------------------------------
# hit_probability_for_gun
# ---------------------------------------------------------------------------

def test_hit_probability_for_gun_origin_matches_direct():
    calc = _calc()
    gun = GunParameters(position=[0.0, 0.0, 0.0])
    np.random.seed(0)
    direct = calc.calculate_hit_probability(
        (1000.0, 0.0, 100.0), (20.0, 0.0, 0.0), _UNC, n_samples=200
    )
    np.random.seed(0)
    via_gun = hit_probability_for_gun(
        calc, gun, (1000.0, 0.0, 100.0), (20.0, 0.0, 0.0),
        measurement_uncertainty=_UNC, n_samples=200,
    )
    assert direct == via_gun


def test_hit_probability_for_gun_translation_invariant():
    calc = _calc()
    g0 = GunParameters(position=[0.0, 0.0, 0.0])
    g1 = GunParameters(position=[100.0, 0.0, 0.0])
    np.random.seed(0)
    hp0 = hit_probability_for_gun(
        calc, g0, (1000.0, 0.0, 100.0), (20.0, 0.0, 0.0),
        measurement_uncertainty=_UNC, n_samples=200,
    )
    np.random.seed(0)
    hp1 = hit_probability_for_gun(
        calc, g1, (1100.0, 0.0, 100.0), (20.0, 0.0, 0.0),
        measurement_uncertainty=_UNC, n_samples=200,
    )
    assert hp0 == hp1


# ---------------------------------------------------------------------------
# Battery (number of guns as a parameter)
# ---------------------------------------------------------------------------

def test_battery_line_count():
    b = Battery.line(n_guns=4)
    assert b.n_guns == 4
    assert len(b) == 4
    assert [g.gun_id for g in b] == [0, 1, 2, 3]


def test_battery_line_single_at_origin():
    b = Battery.line(n_guns=1)
    assert b.n_guns == 1
    assert np.allclose(b[0].position, [0.0, 0.0, 0.0])


def test_battery_line_centered_spacing():
    b = Battery.line(n_guns=3, spacing=50.0, axis="y")
    ys = [g.position[1] for g in b]
    assert np.allclose(ys, [-50.0, 0.0, 50.0])


def test_battery_line_axis_x():
    b = Battery.line(n_guns=2, spacing=100.0, axis="x")
    xs = [g.position[0] for g in b]
    assert np.allclose(xs, [-50.0, 50.0])


def test_battery_from_positions():
    b = Battery.from_positions([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], ammo_type="tpt")
    assert b.n_guns == 2
    assert np.allclose(b[1].position, [100.0, 0.0, 0.0])


def test_battery_invalid_count():
    with pytest.raises(ValueError):
        Battery.line(n_guns=0)
