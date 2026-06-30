"""
Tests for parameter_sweep.py

Covers: run_training — stdout parsing, config file lifecycle,
        non-zero return-code handling, and missing-metric handling.
"""

import json
import os
import sys
import subprocess
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import matplotlib

matplotlib.use("Agg")

from parameter_sweep import run_training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    epsilon_decay_steps=500,
    firing_reward_high=100,
    time_penalty_factor=-100.0,
    hold_penalty_high=-10.0,
    eval_interval=50,
    reward_excellent=100.0,
    reward_good=80.0,
    reward_minimum=60.0,
    reward_fair=40.0,
    reward_poor=20.0,
    reward_failure=-30.0,
    hold_penalty_base=-10.0,
    opportunity_cost_excellent=-30.0,
    opportunity_cost_good=-15.0,
    opportunity_cost_minimum=-8.0,
    timesteps=100,
)

_FAKE_STDOUT = (
    "Training phase 1...\n"
    "Average HP: 0.75 (target: 0.70)\n"
    "Min HP: 0.60\n"
    "Timing efficiency (avg step ≤ 35): 28.0 ✓\n"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunTraining:

    def _run(self, stdout="", returncode=0, **overrides):
        """Call run_training with a mocked subprocess.run."""
        fake_result = MagicMock()
        fake_result.stdout = stdout
        fake_result.returncode = returncode

        kwargs = {**_BASE_KWARGS, **overrides}
        with patch("parameter_sweep.subprocess.run", return_value=fake_result) as mock_run:
            result = run_training(**kwargs)
        return result, mock_run

    def test_parses_avg_hp(self):
        """avg_hp is extracted from 'Average HP:' line."""
        result, _ = self._run(stdout=_FAKE_STDOUT)
        assert abs(result.avg_hp - 0.75) < 1e-9

    def test_parses_min_hp(self):
        """min_hp is extracted from 'Min HP:' line."""
        result, _ = self._run(stdout=_FAKE_STDOUT)
        assert abs(result.min_hp - 0.60) < 1e-9

    def test_parses_avg_steps(self):
        """avg_steps is extracted from the timing efficiency line."""
        result, _ = self._run(stdout=_FAKE_STDOUT)
        assert abs(result.avg_steps - 28.0) < 1e-9

    def test_success_true_when_hp_above_threshold(self):
        """success=True when returncode==0 and avg_hp >= 0.7."""
        result, _ = self._run(stdout=_FAKE_STDOUT, returncode=0)
        assert result.success is True

    def test_success_false_on_nonzero_returncode(self):
        """success=False when subprocess exits with non-zero code."""
        result, _ = self._run(stdout=_FAKE_STDOUT, returncode=1)
        assert result.success is False

    def test_success_false_when_hp_below_threshold(self):
        """success=False when avg_hp < 0.7."""
        low_hp_stdout = _FAKE_STDOUT.replace("Average HP: 0.75", "Average HP: 0.65")
        result, _ = self._run(stdout=low_hp_stdout)
        assert result.success is False

    def test_missing_metrics_return_none(self):
        """avg_hp, min_hp, avg_steps are None when stdout has no matching lines."""
        result, _ = self._run(stdout="Training complete.\n")
        assert result.avg_hp is None
        assert result.min_hp is None
        assert result.avg_steps is None

    def test_config_file_written_and_deleted(self):
        """run_training writes a temp JSON config and deletes it afterwards."""
        written_paths = []
        original_unlink = os.unlink

        def capture_unlink(path):
            written_paths.append(path)
            original_unlink(path)

        fake_result = MagicMock()
        fake_result.stdout = _FAKE_STDOUT
        fake_result.returncode = 0

        with patch("parameter_sweep.subprocess.run", return_value=fake_result):
            with patch("parameter_sweep.os.unlink", side_effect=capture_unlink):
                run_training(**_BASE_KWARGS)

        assert len(written_paths) == 1
        assert not os.path.exists(written_paths[0])

    def test_config_file_contains_correct_values(self):
        """The temp config JSON contains the exact parameters passed to run_training."""
        captured_config = {}

        original_run = subprocess.run

        def capture_cmd(cmd, **kwargs):
            # Find the --config argument and read the file
            config_idx = cmd.index("--config") + 1
            config_path = cmd[config_idx]
            with open(config_path) as f:
                captured_config.update(json.load(f))
            fake = MagicMock()
            fake.stdout = _FAKE_STDOUT
            fake.returncode = 0
            return fake

        with patch("parameter_sweep.subprocess.run", side_effect=capture_cmd):
            run_training(**_BASE_KWARGS)

        assert captured_config["epsilon_decay_steps"] == 500
        assert captured_config["reward_excellent"] == 100.0
        assert captured_config["time_penalty_factor"] == -100.0


import parameter_sweep


class TestMainDispatch:

    def test_grid_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "grid"])
        with patch("parameter_sweep.run_grid_search") as m:
            parameter_sweep.main()
        m.assert_called_once()

    def test_optuna_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "optuna", "--trials", "3"])
        with patch("parameter_sweep.run_general_optuna") as m:
            parameter_sweep.main()
        m.assert_called_once()

    def test_reward_optuna_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "reward_optuna", "--trials", "2"])
        with patch("parameter_sweep.run_optuna_reward_optimization") as m:
            parameter_sweep.main()
        m.assert_called_once()


def _ok_result():
    return parameter_sweep.TrainingResult(
        avg_hp=0.75, min_hp=0.6, avg_steps=20.0, hp_std=0.15, success=True
    )


class TestRunGridSearch:

    def test_grid_search_runs_and_saves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(parameter_sweep, "RESULTS_DIR", tmp_path)
        with patch("parameter_sweep.run_training", return_value=_ok_result()) as rt:
            parameter_sweep.run_grid_search()
        assert rt.call_count == 6 * 3 * 4 * 3 * 3
        assert (tmp_path / "parameter_sweep_results.csv").exists()
        assert (tmp_path / "parameter_sweep_scatter.png").exists()
        assert (tmp_path / "parameter_sweep_top10_bar.png").exists()


class TestOptunaDrivers:

    def test_run_general_optuna(self, tmp_path, monkeypatch):
        monkeypatch.setattr(parameter_sweep, "CONFIG_DIR", tmp_path)
        with patch("parameter_sweep.run_training", return_value=_ok_result()):
            parameter_sweep.run_general_optuna(n_trials=1, study_name="t_general")
        assert (tmp_path / "best_t_general_params.json").exists()

    def test_run_optuna_reward(self, tmp_path, monkeypatch):
        monkeypatch.setattr(parameter_sweep, "CONFIG_DIR", tmp_path)
        with patch("parameter_sweep.run_training", return_value=_ok_result()):
            parameter_sweep.run_optuna_reward_optimization(n_trials=1, study_name="t_reward")
        assert (tmp_path / "best_t_reward_params.json").exists()
