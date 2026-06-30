"""
Tests for TrainingMonitor.

Covers episode recording and the recent / scenario / phase summaries,
including the empty and unknown-key fall-through paths.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from rl_training.utils.training_monitor import TrainingMonitor


def test_empty_summaries():
    m = TrainingMonitor()
    s = m.get_recent_summary()
    assert s["total_episodes"] == 0
    assert s["avg_reward"] == 0.0
    assert s["avg_hp"] == 0.0
    assert s["avg_steps"] == 0.0
    assert m.get_scenario_summary("missing") == {}
    assert m.get_phase_summary(1) == {}


def test_record_and_summaries():
    m = TrainingMonitor()
    m.record_episode("A", reward=10.0, final_hp=0.8, steps=5, fired=True, phase=1)
    m.record_episode("A", reward=20.0, final_hp=0.6, steps=7, fired=False, phase=1)

    s = m.get_recent_summary()
    assert s["total_episodes"] == 2
    assert s["avg_reward"] == pytest.approx(15.0)
    assert s["avg_hp"] == pytest.approx(0.7)
    assert s["avg_steps"] == pytest.approx(6.0)

    sc = m.get_scenario_summary("A")
    assert sc["episodes"] == 2
    assert sc["avg_reward"] == pytest.approx(15.0)
    assert sc["avg_hp"] == pytest.approx(0.7)

    ph = m.get_phase_summary(1)
    assert ph["episodes"] == 2
    assert ph["avg_hp"] == pytest.approx(0.7)
    assert ph["min_hp"] == pytest.approx(0.6)
    assert ph["max_hp"] == pytest.approx(0.8)


def test_recent_summary_window():
    m = TrainingMonitor()
    m.record_episode("A", 10.0, 0.8, 5, True, phase=1)
    m.record_episode("A", 20.0, 0.6, 7, False, phase=1)
    s = m.get_recent_summary(window=1)
    assert s["avg_reward"] == pytest.approx(20.0)
    assert s["total_episodes"] == 2


def test_phase_summary_requires_phase():
    m = TrainingMonitor()
    m.record_episode("A", 10.0, 0.8, 5, True)  # phase defaults to None
    assert m.get_phase_summary(1) == {}


def test_reset():
    m = TrainingMonitor()
    m.record_episode("A", 10.0, 0.8, 5, True, phase=1)
    m.reset()
    assert m.get_recent_summary()["total_episodes"] == 0
    assert m.get_scenario_summary("A") == {}
    assert m.get_phase_summary(1) == {}
