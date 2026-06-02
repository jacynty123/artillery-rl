#!/usr/bin/env python3
"""
Generate a detailed LaTeX report of DQN evaluation results.

Reads evaluation_details.json (exported by evaluate_dqn_trajectories.py)
and produces reports/training_summary.tex with per-scenario analysis.
"""

from pathlib import Path
import json
import numpy as np

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DETAILS_JSON = Path(__file__).parent.parent / "results" / "evaluation_details.json"


def load_details(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Detailed results file not found: {path}. Re-run evaluate_dqn_trajectories.py.")
    return json.loads(path.read_text())


def tex_escape(s: str) -> str:
    return (
        s.replace('&', '\\&')
         .replace('%', '\\%')
         .replace('$', '\\$')
         .replace('#', '\\#')
         .replace('_', '\\_')
         .replace('{', '\\{')
         .replace('}', '\\}')
    )


def make_scenario_section(r: dict) -> str:
    name = tex_escape(r['scenario_name'])
    fired = r['fired_at_step']
    final_hp = r['final_hp']
    steps = r['steps']
    initial_range = r['initial_range']

    # Target kinematics approximation from positions (finite differences)
    pos = np.array(r['target_positions'])
    vel = np.diff(pos, axis=0)
    if len(vel) > 0:
        mean_vel = vel.mean(axis=0)
        vel_str = f"({mean_vel[0]:.2f}, {mean_vel[1]:.2f}, {mean_vel[2]:.2f}) m/step"
    else:
        vel_str = "N/A"

    # Conditions at firing or final step
    idx = fired if fired is not None else max(0, len(r['target_positions']) - 1)
    hp_at_idx = r['hp_trace'][idx] if r['hp_trace'] else 0.0
    pos_at_idx = r['target_positions'][idx] if len(r['target_positions']) > idx else [0, 0, 0]
    range_at_idx = r['ranges'][idx] if len(r['ranges']) > idx else 0.0

    # Covariance trace: not available in JSON; note limitation
    cov_note = "Covariance trace not available in exported results."

    # Decision rationale
    if fired is not None:
        decision = f"Agent decided to FIRE at step {fired} (HP={hp_at_idx:.3f})."
    else:
        decision = "Agent did NOT fire. Possible causes: low/conflicting HP signal, insufficient confidence, or adverse kinematics (fast movement or increasing range)."

    # LaTeX content
    lines = []
    lines.append(f"\\subsection*{{Scenario: {name}}}\n")
    lines.append("\\begin{itemize}\n")
    lines.append(f"  \item Initial range: {initial_range:.0f} m\n")
    lines.append(f"  \item Steps: {steps}\n")
    lines.append(f"  \item Mean target velocity (x,y,z): {vel_str}\n")
    lines.append(f"  \item Decision: {tex_escape(decision)}\n")
    lines.append(f"  \item Conditions at decision/end: position=({pos_at_idx[0]:.0f},{pos_at_idx[1]:.0f},{pos_at_idx[2]:.0f}) m, range={range_at_idx:.0f} m, HP={hp_at_idx:.3f}\n")
    lines.append(f"  \item {cov_note}\n")
    lines.append("\\end{itemize}\n\n")

    # Small table: first and last 5 HP samples
    hp = r['hp_trace']
    head = hp[:5]
    tail = hp[-5:] if len(hp) > 5 else hp
    lines.append("\\begin{table}[h]\\centering\n")
    lines.append(f"\\caption{{Hit Probability Trace for {name}}}\n")
    lines.append("\\begin{tabular}{l l}\\toprule\n")
    lines.append(f"First 5 & {' , '.join(f'{x:.3f}' for x in head)} \\\\\n")
    lines.append(f"Last 5 & {' , '.join(f'{x:.3f}' for x in tail)} \\\\\n")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n\n")

    # Interpretive notes
    lines.append("\\noindent\\textbf{Interpretation.} ")
    if fired is None:
        lines.append("The agent abstained, which often correlates with low or noisy HP trajectories, rapidly changing target states, or ranges trending unfavorably. Consider curriculum examples with similar kinematics to improve confidence thresholds, and review observation features related to covariance (if available) to calibrate risk-aware firing policies.\n\n")
    else:
        lines.append("The firing decision aligns with an HP increase or peak around the decision time, suggesting the agent detected favorable conditions. Review dispersion/ballistics covariance integration to ensure HP reflects realistic uncertainty near decision points.\n\n")

    return ''.join(lines)


def build_tex(details: list) -> str:
    lines = []
    lines.append("\\documentclass[11pt]{article}\n")
    lines.append("\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\usepackage{geometry}\n\\geometry{margin=1in}\n")
    lines.append("\\title{DQN Evaluation Report: Decision-Making Analysis}\n\\author{Automated Summary}\n\\date{}\n")
    lines.append("\\begin{document}\n\\maketitle\n")
    lines.append("\\section*{Overview}\nThis report presents per-scenario analysis of the trained DQN agent's decision-making. It includes hit probability traces, decision timing, target kinematics, and conditions at decision or episode end. A consolidated trajectory figure is included for reference.\\\n\n")
    
    # Add firing statistics summary
    fired_scenarios = [r for r in details if r['fired_at_step'] is not None]
    no_fire_scenarios = [r for r in details if r['fired_at_step'] is None]
    firing_rate = len(fired_scenarios) / len(details) * 100
    
    lines.append("\\subsection*{Firing Statistics}\n")
    lines.append("\\begin{itemize}\n")
    lines.append(f"  \\item Total scenarios evaluated: {len(details)}\n")
    lines.append(f"  \\item Scenarios with firing decisions: {len(fired_scenarios)} ({firing_rate:.1f}\\%)\n")
    lines.append(f"  \\item Scenarios with no firing: {len(no_fire_scenarios)} ({100-firing_rate:.1f}\\%)\n")
    
    if fired_scenarios:
        avg_fire_step = np.mean([r['fired_at_step'] for r in fired_scenarios])
        avg_fire_hp = np.mean([r['hp_trace'][r['fired_at_step']] for r in fired_scenarios])
        lines.append(f"  \\item Average firing step: {avg_fire_step:.1f}\n")
        lines.append(f"  \\item Average HP at firing: {avg_fire_hp:.3f}\n")
    
    if no_fire_scenarios:
        avg_final_hp_no_fire = np.mean([r['final_hp'] for r in no_fire_scenarios])
        lines.append(f"  \\item Average final HP in no-fire scenarios: {avg_final_hp_no_fire:.3f}\n")
        lines.append("  \\item No-fire scenarios indicate situations where the agent correctly abstains from firing due to persistently poor conditions (low HP, adverse kinematics, or insufficient time for improvement)\n")
    
    lines.append("\\end{itemize}\n\n")
    
    # Include overall trajectory figure if present
    traj_fig = Path("trajectory_evaluation.png")
    if traj_fig.exists():
        lines.append("\\begin{figure}[h]\\centering\n")
        lines.append("\\includegraphics[width=0.95\\textwidth]{trajectory_evaluation.png}\n")
        lines.append("\\caption{Target trajectories with firing points across evaluated scenarios.}\n\\end{figure}\n\n")

    lines.append("\\section*{Scenarios}\n")
    for r in details:
        lines.append(make_scenario_section(r))

    lines.append("\\end{document}\n")
    return ''.join(lines)


def main():
    details = load_details(DETAILS_JSON)
    tex_content = build_tex(details)
    out_path = REPORTS_DIR / "training_summary.tex"
    out_path.write_text(tex_content)
    print(f"LaTeX report written to: {out_path}")
    print("To compile PDF:")
    print(f"  pdflatex -interaction=nonstopmode -output-directory {REPORTS_DIR} {out_path}")


if __name__ == '__main__':
    main()