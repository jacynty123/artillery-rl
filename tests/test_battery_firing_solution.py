"""
Tests for scripts/battery_firing_solution.py.

The per-gun ballistics (solve_firing_solution / hit_probability_for_gun) are mocked so
the tests stay fast and focus on the battery orchestration and the combined salvo
kill-probability formula.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from unittest.mock import patch, MagicMock

import scripts.battery_firing_solution as bfs
from src.gun import Battery, FiringSolution


def _sol(feasible=True, reason=""):
    return FiringSolution(
        elevation=0.1, azimuth=0.0, time_to_impact=1.0,
        hit_point=np.zeros(3), miss_distance=0.5,
        feasible=feasible, infeasible_reason=reason,
    )


def test_parse_args_defaults():
    args = bfs.parse_args([])
    assert args.num_guns == 1
    assert args.range_m == 1500.0


def test_parse_args_overrides():
    args = bfs.parse_args(["--num_guns", "5", "--range", "2000", "--axis", "x"])
    assert args.num_guns == 5
    assert args.range_m == 2000.0
    assert args.axis == "x"


def test_engage_combined_pkill():
    battery = Battery.line(n_guns=2)
    with patch("scripts.battery_firing_solution.solve_firing_solution", return_value=_sol()), \
         patch("scripts.battery_firing_solution.hit_probability_for_gun", return_value=0.5):
        rows, p_kill = bfs.engage(battery, [1500.0, 0.0, 100.0], [0.0, 0.0, 0.0],
                                  MagicMock(), n_samples=10)
    assert len(rows) == 2
    assert all(r["feasible"] for r in rows)
    # 1 - (1 - 0.5)(1 - 0.5) = 0.75
    assert abs(p_kill - 0.75) < 1e-9


def test_engage_infeasible_guns_excluded():
    battery = Battery.line(n_guns=2)
    with patch("scripts.battery_firing_solution.solve_firing_solution",
               return_value=_sol(feasible=False, reason="out of range")), \
         patch("scripts.battery_firing_solution.hit_probability_for_gun", return_value=0.5):
        rows, p_kill = bfs.engage(battery, [9999.0, 0.0, 100.0], [0.0, 0.0, 0.0],
                                  MagicMock(), n_samples=10)
    assert all(not r["feasible"] for r in rows)
    assert p_kill == 0.0


def test_main_runs_with_num_guns(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--num_guns", "3"])
    with patch("scripts.battery_firing_solution.solve_firing_solution", return_value=_sol()), \
         patch("scripts.battery_firing_solution.hit_probability_for_gun", return_value=0.4):
        rows, p_kill = bfs.main()
    assert len(rows) == 3
    out = capsys.readouterr().out
    assert "3 gun" in out
    assert "Combined salvo kill probability" in out
