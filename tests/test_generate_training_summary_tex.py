"""
Tests for scripts/generate_training_summary_tex.py.

Covers tex_escape, make_scenario_section (fired / no-fire branches),
build_tex and load_details (including the missing-file path).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest

import scripts.generate_training_summary_tex as gtex


def make_record(fired=2, n=6):
    return {
        "scenario_name": "Test_Scenario",
        "fired_at_step": fired,
        "final_hp": 0.5,
        "steps": n,
        "initial_range": 1500.0,
        "target_positions": [[1000.0 + i, 0.0, 0.0] for i in range(n)],
        "hp_trace": [0.1 * i for i in range(n)],
        "ranges": [1500.0 - i for i in range(n)],
    }


def test_tex_escape():
    assert gtex.tex_escape("a_b") == "a\\_b"
    assert gtex.tex_escape("100%") == "100\\%"
    assert gtex.tex_escape("x&y") == "x\\&y"
    assert gtex.tex_escape("a#b$c") == "a\\#b\\$c"


def test_make_scenario_section_fired():
    sec = gtex.make_scenario_section(make_record(fired=2))
    assert "\\subsection" in sec
    assert "\\item" in sec
    assert "FIRE" in sec


def test_make_scenario_section_no_fire():
    sec = gtex.make_scenario_section(make_record(fired=None))
    assert "NOT fire" in sec
    assert "\\item" in sec


def test_build_tex():
    tex = gtex.build_tex([make_record(fired=2), make_record(fired=None)])
    assert "\\documentclass" in tex
    assert "\\begin{document}" in tex
    assert "\\end{document}" in tex
    assert "Firing Statistics" in tex


def test_load_details_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        gtex.load_details(tmp_path / "missing.json")


def test_load_details(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([make_record(fired=2)]))
    data = gtex.load_details(p)
    assert isinstance(data, list)
    assert data[0]["scenario_name"] == "Test_Scenario"


def test_main(tmp_path, monkeypatch):
    details = [make_record(fired=2), make_record(fired=None)]
    p = tmp_path / "details.json"
    p.write_text(json.dumps(details))
    monkeypatch.setattr(gtex, "DETAILS_JSON", p)
    monkeypatch.setattr(gtex, "REPORTS_DIR", tmp_path)
    gtex.main()
    assert (tmp_path / "training_summary.tex").exists()
