"""
Tests for CurriculumScenarios.

Covers per-phase scenario counts, round-robin iteration, indexed access,
difficulty assignment, phase requirements and the info printout.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.curriculum.scenario_generator import ScenarioParameters


def test_phase_scenario_counts():
    assert CurriculumScenarios(phase=1).num_scenarios() == 4
    assert CurriculumScenarios(phase=2).num_scenarios() == 8
    assert CurriculumScenarios(phase=3).num_scenarios() == 12
    assert CurriculumScenarios(phase=4).num_scenarios() == 15


def test_get_next_scenario_round_robin():
    c = CurriculumScenarios(phase=1)
    n = c.num_scenarios()
    seq = [c.get_next_scenario().name for _ in range(n)]
    assert seq[0] == "Static_Close"
    assert len(set(seq)) == n  # one full unique cycle
    assert c.get_next_scenario().name == seq[0]  # wraps around


def test_get_scenario_by_index():
    c = CurriculumScenarios(phase=1)
    s = c.get_scenario_by_index(0)
    assert isinstance(s, ScenarioParameters)
    assert s.name == "Static_Close"
    assert s.range_m == 800.0


def test_get_all_scenarios():
    c = CurriculumScenarios(phase=2)
    alls = c.get_all_scenarios()
    assert len(alls) == c.num_scenarios()
    assert all(isinstance(s, ScenarioParameters) for s in alls)


def test_difficulty_assignment():
    c = CurriculumScenarios(phase=3)
    assert len(c.difficulty) == c.num_scenarios()
    assert all(v in ("easy", "hard") for v in c.difficulty.values())


def test_phase_requirements_differ():
    r1 = CurriculumScenarios(phase=1).get_phase_requirements()
    r4 = CurriculumScenarios(phase=4).get_phase_requirements()
    assert r1["min_avg_hp"] == 0.70
    assert r4["min_avg_hp"] == 0.55
    assert "description" in r1 and "description" in r4


def test_print_curriculum_info(capsys):
    CurriculumScenarios(phase=1).print_curriculum_info()
    out = capsys.readouterr().out
    assert "CURRICULUM PHASE 1" in out
    assert "Static_Close" in out
