"""
Tests for scripts/generate_training_summary.py.

Uses the matplotlib Agg backend and writes artefacts to tmp dirs (via
monkeypatching REPORTS_DIR) so no files leak into the repo.
"""

import sys
import os

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import scripts.generate_training_summary as gts


def make_df():
    return pd.DataFrame(
        {
            "scenario_name": ["a", "b", "c"],
            "initial_range": [300.0, 1500.0, 3500.0],
            "steps": [5, 10, 8],
            "fired_at_step": [3.0, None, 6.0],
            "final_hp": [0.8, 0.4, 0.6],
        }
    )


def write_csv(tmp_path):
    p = tmp_path / "eval.csv"
    make_df().to_csv(p, index=False)
    return p


def test_load_data_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        gts.load_data(tmp_path / "nope.csv")


def test_load_data_adds_fired(tmp_path):
    df = gts.load_data(write_csv(tmp_path))
    assert "fired" in df.columns
    assert df["fired"].tolist() == [True, False, True]


def test_compute_metrics_fired(tmp_path):
    m = gts.compute_metrics(gts.load_data(write_csv(tmp_path)))
    assert m["n_trajectories"] == 3
    assert m["fired_step_mean"] is not None
    assert 0.0 <= m["firing_rate"] <= 1.0
    assert m["hp_by_range_bucket"] is not None


def test_compute_metrics_no_fire():
    df = pd.DataFrame(
        {
            "scenario_name": ["a", "b"],
            "initial_range": [300.0, 1500.0],
            "steps": [5, 6],
            "fired_at_step": [None, None],
            "final_hp": [0.3, 0.4],
        }
    )
    df["fired"] = df["fired_at_step"].notnull()
    m = gts.compute_metrics(df)
    assert m["fired_step_mean"] is None


def test_plots_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(gts, "REPORTS_DIR", tmp_path)
    df = gts.load_data(write_csv(tmp_path))
    m = gts.compute_metrics(df)

    p1 = gts.plot_hp_distribution(df)
    p2 = gts.plot_hp_vs_range(df)
    p3 = gts.plot_firing_step_hist(df)
    p4 = gts.plot_hp_by_range_bucket(m)

    assert p1.exists() and p2.exists() and p4.exists()
    assert p3 is not None and p3.exists()

    assets = {
        "hp_distribution": str(p1),
        "hp_vs_range": str(p2),
        "firing_step_histogram": str(p3),
        "hp_by_range_bucket": str(p4),
    }
    md = gts.write_markdown_report(m, assets)
    assert md.exists()
    assert "DQN Training Evaluation Summary" in md.read_text()


def test_plot_firing_step_hist_empty():
    df = pd.DataFrame(
        {
            "scenario_name": ["a"],
            "initial_range": [300.0],
            "steps": [5],
            "fired_at_step": [None],
            "final_hp": [0.3],
        }
    )
    df["fired"] = df["fired_at_step"].notnull()
    assert gts.plot_firing_step_hist(df) is None


def test_main(tmp_path, monkeypatch):
    monkeypatch.setattr(gts, "INPUT_CSV", write_csv(tmp_path))
    monkeypatch.setattr(gts, "REPORTS_DIR", tmp_path)
    gts.main()
    assert (tmp_path / "training_summary.md").exists()
