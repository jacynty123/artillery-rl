#!/usr/bin/env python3
"""
Evaluate the trained DQN model on artillery firing trajectories.

This script loads the trained DQN model and evaluates its performance
on various trajectory scenarios, showing hit probability traces and firing decisions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_training.environment import ArtilleryFiringEnv
from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.agents.dqn_components import DQN
from rl_training.utils.evaluation_utils import evaluate_curriculum


def load_dqn_model(model_path, state_dim, action_dim, device):
    """Load the trained DQN model."""
    model = DQN(state_dim, action_dim).to(device)
    if model_path.exists():
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")


def evaluate_trajectory(env, q_network, scenario, device, max_steps=100, seed=None):
    """Evaluate a single trajectory and return detailed results."""
    state, _ = env.reset(scenario_override=scenario, seed=seed)
    hp_trace = []
    actions = []
    rewards = []
    ranges = []
    target_positions = []  # Track full target position (x,y,z)
    cov_traces = []  # Track covariance trace
    done = False
    step_count = 0
    fired_at_step = None
    initial_range = scenario.range_m
    initial_y = 0.0  # Target starts at y=0
    initial_z = 50.0  # Target starts at z=50m

    # Record initial HP and covariance
    hp_trace.append(env.current_state.current_hit_probability)
    cov_traces.append(env.current_state.covariance_trace)
    ranges.append(initial_range)
    target_positions.append((initial_range, initial_y, initial_z))

    while not done and step_count < max_steps:
        # Get action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = q_network(state_tensor).argmax().item()

        # Record firing decision
        if action == 1 and fired_at_step is None:  # FIRE action
            fired_at_step = step_count

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record trajectory data
        actions.append(action)
        rewards.append(reward)

        # Calculate current range and full target position
        elapsed_time = step_count * (scenario.tracking_duration / env.max_episode_steps)
        current_x = initial_range + scenario.target_vx * elapsed_time
        current_y = initial_y + scenario.target_vy * elapsed_time
        current_z = initial_z + scenario.target_vz * elapsed_time
        ranges.append(current_x)
        target_positions.append((current_x, current_y, current_z))

        # Record HP and covariance after step
        hp_trace.append(info["hit_probability"])
        cov_traces.append(info["covariance_trace"])

        state = next_state
        step_count += 1

    return {
        'hp_trace': hp_trace,
        'actions': actions,
        'rewards': rewards,
        'ranges': ranges,
        'target_positions': target_positions,
        'cov_traces': cov_traces,
        'fired_at_step': fired_at_step,
        'final_hp': hp_trace[-1] if hp_trace else 0.0,
        'scenario_name': scenario.name,
        'initial_range': initial_range,
        'steps': step_count
    }


def plot_trajectory_results(results, save_path=None):
    """Plot hit probability traces with firing decisions and additional analyses."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('DQN Model Evaluation on Artillery Trajectories', fontsize=16)

    # Create subplots: 2x3 grid (HP traces + analyses)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # HP trace plots (first 4 scenarios)
    hp_axes = []
    for i in range(4):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        hp_axes.append(ax)

    colors = ['blue', 'red', 'green', 'orange']

    for i, result in enumerate(results[:4]):  # Plot first 4 trajectories
        ax = hp_axes[i]

        # Plot HP trace
        steps = range(len(result['hp_trace']))
        ax.plot(steps, result['hp_trace'], 'b-', linewidth=2, label='Hit Probability')

        # Mark firing decision
        if result['fired_at_step'] is not None:
            fire_step = result['fired_at_step']
            fire_hp = result['hp_trace'][fire_step]
            ax.scatter(fire_step, fire_hp, color='red', s=100, marker='x',
                      linewidth=3, label=f'FIRE (HP={fire_hp:.3f})')

        # Add scenario info
        ax.set_title(f"{result['scenario_name']}\nRange: {result['initial_range']:.0f}m")
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hit Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set reasonable y-axis limits
        ax.set_ylim(0, 1.0)

    # Covariance Trace vs Time
    ax_cov_time = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # HP trace plots (first 4 scenarios)
    hp_axes = []
    for i in range(4):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        hp_axes.append(ax)

    colors = ['blue', 'red', 'green', 'orange']

    for i, result in enumerate(results[:4]):  # Plot first 4 trajectories
        ax = hp_axes[i]

        # Plot HP trace
        steps = range(len(result['hp_trace']))
        ax.plot(steps, result['hp_trace'], 'b-', linewidth=2, label='Hit Probability')

        # Mark firing decision
        if result['fired_at_step'] is not None:
            fire_step = result['fired_at_step']
            fire_hp = result['hp_trace'][fire_step]
            ax.scatter(fire_step, fire_hp, color='red', s=100, marker='x',
                      linewidth=3, label=f'FIRE (HP={fire_hp:.3f})')

        # Add scenario info
        ax.set_title(f"{result['scenario_name']}\nRange: {result['initial_range']:.0f}m")
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hit Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set reasonable y-axis limits
        ax.set_ylim(0, 1.0)

    # Covariance Trace vs Time
    ax_cov_time = fig.add_subplot(gs[:, 2])
    for i, result in enumerate(results):
        if 'cov_traces' in result and result['cov_traces']:
            steps = range(len(result['cov_traces']))
            ax_cov_time.scatter(steps, result['cov_traces'], label=result['scenario_name'], alpha=0.7, marker='o', s=10)
    ax_cov_time.set_xlabel('Time Step')
    ax_cov_time.set_ylabel('Covariance Trace')
    ax_cov_time.set_title('Covariance Trace vs Time')
    ax_cov_time.legend()
    ax_cov_time.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plots to {save_path}")
    else:
        plt.show()


def plot_individual_scenarios(results, plots_dir="scenario_plots"):
    """Create individual plots for each scenario and save to a directory."""
    import os
    os.makedirs(plots_dir, exist_ok=True)
    
    for result in results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"DQN Evaluation: {result['scenario_name']}", fontsize=14)
        
        scenario_name = result['scenario_name']
        steps = range(len(result['hp_trace']))
        
        # HP Trace
        ax1.plot(steps, result['hp_trace'], 'b-', linewidth=2, label='Hit Probability')
        if result['fired_at_step'] is not None:
            fire_step = result['fired_at_step']
            fire_hp = result['hp_trace'][fire_step]
            ax1.scatter(fire_step, fire_hp, color='red', s=100, marker='x',
                       linewidth=3, label=f'FIRE (HP={fire_hp:.3f})')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Hit Probability')
        ax1.set_title('Hit Probability Trace')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        
        # Range vs Time
        if result['ranges']:
            ax2.plot(steps, result['ranges'], 'g-', linewidth=2)
            if result['fired_at_step'] is not None:
                fire_step = result['fired_at_step']
                fire_range = result['ranges'][fire_step] if fire_step < len(result['ranges']) else result['ranges'][-1]
                ax2.scatter(fire_step, fire_range, color='red', s=100, marker='x', linewidth=3)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Range (m)')
            ax2.set_title('Target Range vs Time')
            ax2.grid(True, alpha=0.3)
        
        # Covariance Trace
        if result['cov_traces'] and any(c is not None for c in result['cov_traces']):
            valid_cov = [(i, c) for i, c in enumerate(result['cov_traces']) if c is not None]
            if valid_cov:
                cov_steps, cov_values = zip(*valid_cov)
                ax3.plot(cov_steps, cov_values, 'purple', linewidth=2, marker='o', markersize=3)
                if result['fired_at_step'] is not None:
                    fire_step = result['fired_at_step']
                    if fire_step < len(result['cov_traces']) and result['cov_traces'][fire_step] is not None:
                        ax3.scatter(fire_step, result['cov_traces'][fire_step], color='red', s=100, marker='x', linewidth=3)
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Covariance Trace')
                ax3.set_title('Kalman Filter Covariance Trace')
                ax3.grid(True, alpha=0.3)
        
        # Actions and Rewards
        ax4_twin = ax4.twinx()
        # Actions are recorded per step, so they have one less element than hp_trace
        action_steps = range(len(result['actions']))
        ax4.step(action_steps, result['actions'], 'orange', linewidth=2, where='post', label='Action')
        ax4_twin.plot(action_steps, result['rewards'], 'cyan', linewidth=2, marker='s', markersize=3, label='Reward')
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Action (0=HOLD, 1=FIRE)', color='orange')
        ax4_twin.set_ylabel('Reward', color='cyan')
        ax4.set_title('Actions and Rewards')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['HOLD', 'FIRE'])
        ax4.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text = f"Initial Range: {result['initial_range']:.0f}m\n"
        summary_text += f"Steps: {result['steps']}\n"
        summary_text += f"Final HP: {result['final_hp']:.3f}\n"
        if result['fired_at_step'] is not None:
            summary_text += f"Fired at step: {result['fired_at_step']}\n"
        else:
            summary_text += "No firing decision\n"
        
        fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        safe_name = scenario_name.replace(' ', '_').replace('/', '_')
        plot_path = os.path.join(plots_dir, f"{safe_name}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Individual scenario plots saved to {plots_dir}/ directory")


def print_trajectory_summary(results):
    """Print detailed summary of trajectory evaluations."""
    print("\n" + "="*100)
    print("TRAJECTORY EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Scenario':<25} {'Initial Range':<12} {'Steps':<6} {'Fired At':<9} {'Final HP':<9} {'Range@Fire':<12} {'Position@Fire (x,y,z)':<20}")
    print("-"*100)

    for result in results:
        fired_step = result['fired_at_step']
        if fired_step is not None and fired_step < len(result['ranges']):
            fire_range = result['ranges'][fired_step]
            fire_pos = result['target_positions'][fired_step]
            pos_str = f"({fire_pos[0]:.0f},{fire_pos[1]:.0f},{fire_pos[2]:.0f})"
        else:
            fire_range = result['ranges'][-1] if result['ranges'] else 0.0
            fire_pos = result['target_positions'][-1] if result['target_positions'] else (0.0, 0.0, 0.0)
            pos_str = f"({fire_pos[0]:.0f},{fire_pos[1]:.0f},{fire_pos[2]:.0f})"

        print(f"{result['scenario_name']:<25} "
              f"{result['initial_range']:<12.0f} "
              f"{result['steps']:<6} "
              f"{fired_step if fired_step is not None else '-':<9} "
              f"{result['final_hp']:<9.3f} "
              f"{fire_range:<12.0f} "
              f"{pos_str:<20}")

    # Summary statistics
    fired_count = sum(1 for r in results if r['fired_at_step'] is not None)
    avg_final_hp = np.mean([r['final_hp'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])

    print("-"*100)
    print(f"Total trajectories: {len(results)}")
    print(f"Trajectories with firing: {fired_count}")
    print(f"Average final HP:         {avg_final_hp:.3f}")
    print(f"Average steps:            {avg_steps:.1f}")
    print("="*100)


def main(num_runs: int = 1):
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=num_runs,
                        help="Number of independent evaluation runs (each with a different seed)")
    args = parser.parse_args()
    num_runs = args.num_runs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment and curriculum
    env = ArtilleryFiringEnv()
    curriculum = CurriculumScenarios(phase=4)  # Use Phase 4 for comprehensive evaluation

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load the trained model (Phase 4)
    model_path = Path("rl_training/models/checkpoints/dqn_curriculum_phase4.pth")
    try:
        q_network = load_dqn_model(model_path, state_dim, action_dim, device)
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Available checkpoints:")
        checkpoints_dir = Path("rl_training/models/checkpoints")
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.pth"):
                print(f"  {f.name}")
        return

    scenarios_to_test = [
        curriculum.get_scenario_by_index(0),   # Static_Close
        curriculum.get_scenario_by_index(4),   # Fast_Approach_Long
        curriculum.get_scenario_by_index(6),   # Very_Fast_Approach
        curriculum.get_scenario_by_index(11),  # ExtremeLong_Medium_Slow
        curriculum.get_scenario_by_index(13),  # ExtremeLong_Medium_Approaching
    ]

    def perturb_scenario(scenario, rng):
        """Return a copy of scenario with small random perturbations.

        Perturbations (±5–10 %) simulate real-world uncertainty in initial
        conditions, giving each run a genuinely different physics trajectory.
        """
        import copy
        s = copy.copy(scenario)
        s.range_m           = scenario.range_m           * rng.uniform(0.95, 1.05)
        s.target_vx         = scenario.target_vx         * rng.uniform(0.90, 1.10) if scenario.target_vx != 0 else 0.0
        s.target_vy         = scenario.target_vy         * rng.uniform(0.90, 1.10) if scenario.target_vy != 0 else 0.0
        s.target_vz         = scenario.target_vz         * rng.uniform(0.90, 1.10) if scenario.target_vz != 0 else 0.0
        s.measurement_noise_std = scenario.measurement_noise_std * rng.uniform(0.80, 1.20)
        return s

    all_results = []  # collects results from every run
    for run_idx in range(num_runs):
        seed = run_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n{'='*60}\nRUN {run_idx+1}/{num_runs}  (seed={seed})\n{'='*60}")
        results = []
        for scenario in scenarios_to_test:
            perturbed = perturb_scenario(scenario, rng)
            print(f"  Evaluating: {perturbed.name}  (range={perturbed.range_m:.0f}m)")
            result = evaluate_trajectory(env, q_network, perturbed, device,
                                         seed=seed)
            results.append(result)
        all_results.append(results)

    # Print detailed summary
    print_trajectory_summary(results)

    import pandas as pd

    # Flatten all results into a single list with run/seed columns
    flat_rows = []
    for run_idx, run_results in enumerate(all_results):
        for r in run_results:
            flat_rows.append({
                'run': run_idx,
                'seed': run_idx,
                'scenario_name': r['scenario_name'],
                'final_hp': float(r['final_hp']),
                'fired_at_step': r['fired_at_step'],
                'steps': r['steps'],
                'initial_range': float(r['initial_range']),
            })
    df_all = pd.DataFrame(flat_rows)

    # Per-run summary (last run for single-run mode)
    last_results = all_results[-1]
    if num_runs == 1:
        print_trajectory_summary(last_results)

    # Multi-run statistics table
    if num_runs > 1:
        stats = (
            df_all.groupby('scenario_name')
            .agg(
                mean_final_hp  =('final_hp', 'mean'),
                std_final_hp   =('final_hp', 'std'),
                min_final_hp   =('final_hp', 'min'),
                max_final_hp   =('final_hp', 'max'),
                mean_fired_step=('fired_at_step', 'mean'),
                std_fired_step =('fired_at_step', 'std'),
                firing_rate    =('fired_at_step', lambda x: x.notna().mean() * 100),
            )
            .reset_index()
        )
        # Coefficient of variation (%) — std / mean * 100, meaningful robustness metric
        stats['cv_pct'] = (stats['std_final_hp'] / stats['mean_final_hp'] * 100).round(1)

        # ---- pretty console table ----
        col_w = [28, 8, 8, 7, 7, 10, 10, 8, 7]
        headers = ['Scenario', 'Mean HP', 'Std HP', 'Min', 'Max',
                   'Mean Step', 'Std Step', 'Fire%', 'CV%']
        sep = '+' + '+'.join('-' * w for w in col_w) + '+'
        hdr = '|' + '|'.join(h.center(w) for h, w in zip(headers, col_w)) + '|'

        print(f"\n{'='*70}")
        print(f"ROBUSTNESS STATISTICS  ({num_runs} runs, seeds 0–{num_runs-1})")
        print(f"Parameter perturbations: range ±5%, velocity ±10%, noise ±20%")
        print(f"{'='*70}")
        print(sep); print(hdr); print(sep)
        for _, row in stats.iterrows():
            fired_step_mean = f"{row['mean_fired_step']:.1f}" if pd.notna(row['mean_fired_step']) else '—'
            fired_step_std  = f"{row['std_fired_step']:.1f}"  if pd.notna(row['std_fired_step'])  else '—'
            vals = [
                row['scenario_name'][:col_w[0]-2],
                f"{row['mean_final_hp']:.3f}",
                f"{row['std_final_hp']:.3f}",
                f"{row['min_final_hp']:.3f}",
                f"{row['max_final_hp']:.3f}",
                fired_step_mean,
                fired_step_std,
                f"{row['firing_rate']:.0f}",
                f"{row['cv_pct']:.1f}",
            ]
            print('|' + '|'.join(v.center(w) for v, w in zip(vals, col_w)) + '|')
        print(sep)
        print("\nNote: This is a robustness (sensitivity) analysis, not a generalization")
        print("test. Perturbations simulate sensor/model uncertainty (cf. Tobin et al.,")
        print("2017, Domain Randomization; Rajeswaran et al., 2017, EPOpt). CV% is the")
        print("coefficient of variation (std/mean × 100) — lower values indicate a more")
        print("robust policy. For out-of-distribution generalization, held-out scenario")
        print("families would be required (Cobbe et al., 2019; Packer et al., 2018).")

        stats.to_csv("evaluation_summary_multiple_runs.csv", index=False)
        print("\nPer-run statistics saved to evaluation_summary_multiple_runs.csv")

    # Save flat CSV with all individual results
    df_all.to_csv("evaluation_summary.csv", index=False)
    print("All results saved to evaluation_summary.csv")

    # Plots based on last run
    plot_trajectory_results(last_results, save_path="trajectory_evaluation.png")
    plot_individual_scenarios(last_results, plots_dir="scenario_plots")

    print("\n🎯 Evaluation complete!")
    print("📊 Trajectory plots saved to: trajectory_evaluation.png")
    print("📊 Individual scenario plots saved to: scenario_plots/ directory")
if __name__ == "__main__":
    main()