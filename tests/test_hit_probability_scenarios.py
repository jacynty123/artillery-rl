"""
Tests for scripts/hit_probability_scenarios.py.

Covers the pure-logic scenario builder and the summary/visualisation helpers.
The heavy `analyze_scenario` / `main` (real firing-angle optimisation + Kalman
Monte-Carlo) remain deferred.
"""

import sys
import os

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import scripts.hit_probability_scenarios as hps


def test_create_scenario_basic():
    s = hps.create_scenario(1000.0, 12.0, 6.0, 3.0, 150.0, 15.0, 0.0)
    assert s["range"] == 1000.0
    assert s["target_dims"] == (12.0, 6.0, 3.0)
    assert s["projectile_type"] == "tpt"
    assert s["use_dynamic_covariance"] is True
    assert np.allclose(s["target_velocity"], [150.0, 15.0, 0.0])
    assert "projectile_covariance" in s
    assert s["measurement_noise_std"].shape == (3,)
    assert s["process_noise_std"].shape == (3,)


def test_create_scenario_process_noise_branches():
    slow = hps.create_scenario(1000.0, 10, 8, 4, 10.0, 0.0, 0.0)    # speed 10 -> slow
    medium = hps.create_scenario(1000.0, 10, 8, 4, 60.0, 0.0, 0.0)  # speed 60 -> medium
    fast = hps.create_scenario(1000.0, 10, 8, 4, 150.0, 0.0, 0.0)   # speed 150 -> fast
    assert np.allclose(slow["process_noise_std"], [0.3, 0.3, 0.1])
    assert np.allclose(medium["process_noise_std"], [1.0, 1.0, 0.5])
    assert np.allclose(fast["process_noise_std"], [2.0, 2.0, 1.0])


def test_create_scenario_explicit_noise_overrides():
    meas = np.array([1.0, 1.0, 1.0])
    proc = np.array([0.5, 0.5, 0.5])
    s = hps.create_scenario(
        1000.0, 10, 8, 4, 10.0, 0.0, 0.0,
        measurement_noise_std=meas, process_noise_std=proc,
    )
    assert np.allclose(s["measurement_noise_std"], meas)
    assert np.allclose(s["process_noise_std"], proc)


def _fake_result(hp=0.5, unc=(1.0, 2.0, 3.0)):
    return {
        "scenario": hps.create_scenario(1000.0, 12.0, 6.0, 3.0, 150.0, 15.0, 0.0),
        "hit_probability_analytical": hp,
        "uncertainty_std": np.array(unc),
        "intersection_point": np.array([1000.0, 0.0, 50.0]),
        "intersection_covariance": np.eye(3),
        "uncertainty_volume": 100.0,
    }


def test_create_summary_table(capsys):
    hps.create_summary_table([_fake_result(0.5), _fake_result(0.6)], n_iterations=2)
    out = capsys.readouterr().out
    assert "HIT PROBABILITY ANALYSIS SUMMARY" in out
    assert "Hit Probability" in out


def test_create_visualization_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(hps, "RESULTS_DIR", tmp_path)
    results = [_fake_result(0.5), _fake_result(0.6), _fake_result(0.55)]
    hps.create_visualization(results, n_iterations=3)
    assert (tmp_path / "hit_probability_analysis.png").exists()
