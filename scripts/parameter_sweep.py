"""
Parameter Sweeping Script for DQN Curriculum Training

Sweeps over hyperparameters to find optimal values for firing policy.
Supports both grid search and Optuna Bayesian optimization.
"""

import subprocess
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile
import optuna
import argparse

# Project root is one level up from this scripts/ folder
ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Add imports for statistics and plotting
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class TrainingResult:
    """Structured output from a single run_training() call."""
    avg_hp: float
    min_hp: float
    avg_steps: float
    hp_std: float
    success: bool

def run_training(epsilon_decay_steps, firing_reward_high, time_penalty_factor, hold_penalty_high, eval_interval, 
                reward_excellent, reward_good, reward_minimum, reward_fair, reward_poor, reward_failure,
                hold_penalty_base, opportunity_cost_excellent, opportunity_cost_good, opportunity_cost_minimum,
                timesteps=5000):
    """Run training with given parameters and return evaluation results."""
    
    # Create temporary config file
    config = {
        "epsilon_decay_steps": epsilon_decay_steps,
        "firing_reward_high": firing_reward_high,
        "time_penalty_factor": time_penalty_factor,
        "hold_penalty_high": hold_penalty_high,
        "eval_interval": eval_interval,
        "reward_excellent": reward_excellent,
        "reward_good": reward_good,
        "reward_minimum": reward_minimum,
        "reward_fair": reward_fair,
        "reward_poor": reward_poor,
        "reward_failure": reward_failure,
        "hold_penalty_base": hold_penalty_base,
        "opportunity_cost_excellent": opportunity_cost_excellent,
        "opportunity_cost_good": opportunity_cost_good,
        "opportunity_cost_minimum": opportunity_cost_minimum
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        # Run the training script with modified parameters
        cmd = [
            sys.executable, "-m", "rl_training.train.train_dqn_curriculum",
            "--start_phase", "1", "--end_phase", "1", 
            "--timesteps", str(timesteps),
            "--config", config_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        
        # Parse the output for evaluation results
        lines = result.stdout.split('\n')
        avg_hp = None
        min_hp = None
        avg_steps = None
        
        for line in lines:
            if "Average HP:" in line:
                avg_hp = float(line.split(":")[1].strip().split()[0])
            elif "Min HP:" in line:
                min_hp = float(line.split(":")[1].strip().split()[0])
            elif "Timing efficiency" in line and "≤" in line:
                # Extract avg step from "avg step ≤ 35): 40.0 ✗"
                parts = line.split(":")
                if len(parts) > 1:
                    avg_steps = float(parts[1].strip().split()[0])
        
        hp_std = (avg_hp - min_hp) if avg_hp is not None and min_hp is not None else 0.0
        
        return TrainingResult(
            avg_hp=avg_hp,
            min_hp=min_hp,
            avg_steps=avg_steps,
            hp_std=hp_std,
            success=result.returncode == 0 and avg_hp is not None and avg_hp >= 0.7
        )
        
    finally:
        os.unlink(config_file)


def _save_optuna_plots(study, study_name: str) -> None:
    """Save Optuna HTML visualizations if plotly is available."""
    try:
        import plotly
        print("\nGenerating visualizations...")

        fig = optuna.visualization.plot_param_importances(study)
        out1 = CONFIG_DIR / f"{study_name}_param_importances.html"
        fig.write_html(str(out1))
        print(f"Parameter importance plot saved to {out1}")

        fig2 = optuna.visualization.plot_optimization_history(study)
        out2 = CONFIG_DIR / f"{study_name}_optimization_history.html"
        fig2.write_html(str(out2))
        print(f"Optimization history plot saved to {out2}")

        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        out3 = CONFIG_DIR / f"{study_name}_parallel_coordinate.html"
        fig3.write_html(str(out3))
        print(f"Parallel coordinate plot saved to {out3}")

    except ImportError:
        print("Install plotly for visualizations: pip install plotly")


def _define_reward_search_space(trial) -> dict:
    """Define the reward parameter search space for Optuna trials."""
    return {
        "epsilon_decay_steps": trial.suggest_categorical("epsilon_decay_steps", [250, 500, 1000, 2000, 4000, 8000]),
        "firing_reward_high": trial.suggest_categorical("firing_reward_high", [100, 150]),
        "time_penalty_factor": trial.suggest_categorical("time_penalty_factor", [-100, -120, -150]),
        "hold_penalty_high": trial.suggest_categorical("hold_penalty_high", [-10, -12]),
        "eval_interval": trial.suggest_categorical("eval_interval", [50, 100]),
        "reward_excellent": trial.suggest_categorical("reward_excellent", [80, 100, 120, 150]),
        "reward_good": trial.suggest_categorical("reward_good", [60, 80, 100, 120]),
        "reward_minimum": trial.suggest_categorical("reward_minimum", [40, 60, 80, 100]),
        "reward_fair": trial.suggest_categorical("reward_fair", [20, 40, 60, 80]),
        "reward_poor": trial.suggest_categorical("reward_poor", [10, 20, 30, 40]),
        "reward_failure": trial.suggest_categorical("reward_failure", [-50, -30, -10, 0]),
        "hold_penalty_base": trial.suggest_categorical("hold_penalty_base", [-15, -10, -5, 0]),
        "opportunity_cost_excellent": trial.suggest_categorical("opportunity_cost_excellent", [-40, -30, -20, -10]),
        "opportunity_cost_good": trial.suggest_categorical("opportunity_cost_good", [-20, -15, -10, -5]),
        "opportunity_cost_minimum": trial.suggest_categorical("opportunity_cost_minimum", [-10, -8, -5, 0]),
    }


def run_optuna_reward_optimization(n_trials=50, timesteps_per_trial=10000, study_name="dqn_reward_optimization"):
    """Run comprehensive Optuna optimization focused on reward parameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study with persistent storage
    db_path = CONFIG_DIR / f"{study_name}.db"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True
    )

    def reward_objective(trial):
        """Objective function focused on reward parameter optimization."""
        params = _define_reward_search_space(trial)

        result = run_training(
            params["epsilon_decay_steps"], params["firing_reward_high"], params["time_penalty_factor"],
            params["hold_penalty_high"], params["eval_interval"],
            params["reward_excellent"], params["reward_good"], params["reward_minimum"],
            params["reward_fair"], params["reward_poor"], params["reward_failure"],
            params["hold_penalty_base"], params["opportunity_cost_excellent"],
            params["opportunity_cost_good"], params["opportunity_cost_minimum"],
            timesteps=timesteps_per_trial
        )

        if result.success:
            # Enhanced scoring for reward optimization
            hp_score = result.avg_hp * 10  # Weight HP heavily
            timing_score = -0.05 * ((result.avg_steps or 0) - 20)  # Penalize late firing
            consistency_score = -0.05 * result.hp_std  # Penalize inconsistency
            score = hp_score + timing_score + consistency_score
            return score
        else:
            return -100.0  # Lower penalty for reward optimization

    print(f"Starting comprehensive reward parameter optimization...")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Timesteps per trial: {timesteps_per_trial}")

    # Run optimization
    study.optimize(reward_objective, n_trials=n_trials)

    # Print comprehensive results
    print("\n" + "="*80)
    print("REWARD PARAMETER OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best parameters
    best_params_file = CONFIG_DIR / f"best_{study_name}_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_params_file}")

    # Print optimization statistics
    print(f"\nOptimization Statistics:")
    print(f"Total trials: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.value is not None and t.value != -100.0]
    print(f"Completed trials: {len(completed_trials)}")
    if completed_trials:
        scores = [t.value for t in completed_trials]
        print(f"Mean score: {sum(scores)/len(scores):.4f}")
        print(f"Score std: {pd.Series(scores).std():.4f}")

    # Generate visualizations if plotly is available
    _save_optuna_plots(study, study_name)

    return study


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Parameter sweeping and optimization for DQN training")
    parser.add_argument("--mode", choices=["grid", "optuna", "reward_optuna"],
                       default="grid", help="Optimization mode")
    parser.add_argument("--trials", type=int, default=50,
                       help="Number of Optuna trials (for optuna modes)")
    parser.add_argument("--timesteps", type=int, default=5000,
                       help="Timesteps per training run")
    parser.add_argument("--study_name", type=str, default="dqn_parameter_optimization",
                       help="Name for Optuna study")

    args = parser.parse_args()

    if args.mode == "grid":
        run_grid_search()
    elif args.mode == "optuna":
        run_general_optuna(args.trials, args.timesteps, args.study_name)
    elif args.mode == "reward_optuna":
        run_optuna_reward_optimization(args.trials, args.timesteps, args.study_name)


def run_grid_search():
    """Run traditional grid search over hyperparameters."""
    # Parameter ranges (using default reward values for grid search)
    epsilon_decays = [250, 500, 1000, 2000, 4000, 8000]
    firing_rewards = [50, 100, 150]
    time_penalties = [-80, -100, -120, -150]
    hold_penalties = [-8, -10, -12]
    eval_intervals = [50, 100, 200]

    results = []

    total_combinations = len(epsilon_decays) * len(firing_rewards) * len(time_penalties) * len(hold_penalties) * len(eval_intervals)
    print(f"Starting grid search with {total_combinations} combinations...")

    for eps_decay in epsilon_decays:
        for fire_reward in firing_rewards:
            for time_pen in time_penalties:
                for hold_pen in hold_penalties:
                    for eval_int in eval_intervals:
                        print(f"Testing: eps_decay={eps_decay}, fire_reward={fire_reward}, time_pen={time_pen}, hold_pen={hold_pen}, eval_int={eval_int}")

                        result = run_training(eps_decay, fire_reward, time_pen, hold_pen, eval_int,
                                             100.0, 80.0, 60.0, 40.0, 20.0, -30.0, -10.0, -30.0, -15.0, -8.0)

                        if result.success:
                            score = result.avg_hp - 0.1 * ((result.avg_steps or 0) - 20) - 0.1 * result.hp_std
                            results.append({
                                "params": {
                                    "epsilon_decay_steps": eps_decay,
                                    "firing_reward_high": fire_reward,
                                    "time_penalty_factor": time_pen,
                                    "hold_penalty_high": hold_pen,
                                    "eval_interval": eval_int
                                },
                                "avg_hp": result.avg_hp,
                                "min_hp": result.min_hp,
                                "avg_steps": result.avg_steps,
                                "hp_std": result.hp_std,
                                "score": score
                            })
                            print(f"  SUCCESS: HP={result.avg_hp:.3f}, MinHP={result.min_hp:.3f}, Steps={result.avg_steps:.1f}, Std={result.hp_std:.3f}, Score={score:.3f}")
                        else:
                            print(f"  FAILED: HP={result.avg_hp}, Steps={result.avg_steps}")

    # Sort by score (higher HP, earlier firing)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Tabular summary using pandas
    df = pd.DataFrame(results)
    param_cols = ["epsilon_decay_steps", "firing_reward_high", "time_penalty_factor", "hold_penalty_high", "eval_interval"]
    df_params = df["params"].apply(pd.Series)
    df_full = pd.concat([df_params, df.drop(columns=["params"])], axis=1)

    print("\nTop 10 parameter combinations (tabular summary):")
    print(df_full.head(10).to_string(index=False, float_format="{:.3f}".format))

    # Statistics
    print("\nGrid Search Statistics:")
    print(f"Total runs: {len(df_full)}")
    print(f"Mean HP: {df_full['avg_hp'].mean():.3f}")
    print(f"Std HP: {df_full['avg_hp'].std():.3f}")
    print(f"Mean Steps: {df_full['avg_steps'].mean():.1f}")
    print(f"Best HP: {df_full['avg_hp'].max():.3f}")
    print(f"Best Score: {df_full['score'].max():.3f}")

    # Save tabular summary to CSV
    csv_out = RESULTS_DIR / "parameter_sweep_results.csv"
    df_full.to_csv(csv_out, index=False)
    print(f"Tabular results saved to {csv_out}")

    # Plotting
    plt.figure(figsize=(10,6))
    plt.scatter(df_full['score'], df_full['avg_hp'], c=df_full['avg_steps'], cmap='viridis', s=60, alpha=0.7)
    plt.colorbar(label='Avg Steps')
    plt.xlabel('Sweep Score')
    plt.ylabel('Average Hit Probability')
    plt.title('Parameter Sweep: Score vs. Hit Probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_out = RESULTS_DIR / 'parameter_sweep_scatter.png'
    plt.savefig(scatter_out, dpi=200)
    print(f"Figure saved to {scatter_out}")

    # Bar plot for top 10
    top10 = df_full.head(10)
    plt.figure(figsize=(12,6))
    plt.bar(range(10), top10['avg_hp'], tick_label=[f"{i+1}" for i in range(10)])
    plt.xlabel('Top 10 Parameter Sets')
    plt.ylabel('Average Hit Probability')
    plt.title('Top 10 Parameter Sets by HP')
    plt.tight_layout()
    bar_out = RESULTS_DIR / 'parameter_sweep_top10_bar.png'
    plt.savefig(bar_out, dpi=200)
    print(f"Bar plot saved to {bar_out}")


def run_general_optuna(n_trials=50, timesteps_per_trial=5000, study_name="dqn_general_optimization"):
    """Run general Optuna optimization over all hyperparameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Starting general Optuna optimization with {n_trials} trials...")

    # Create study
    db_path = CONFIG_DIR / f"{study_name}.db"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Print results
    print("\nGeneral Optuna Optimization Results:")
    print(f"Best score: {study.best_value:.3f}")
    print(f"Best parameters: {study.best_params}")

    # Save best parameters
    best_params_file = CONFIG_DIR / f"best_{study_name}_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best parameters saved to {best_params_file}")


def objective(trial):
    """General Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    epsilon_decay_steps = trial.suggest_categorical("epsilon_decay_steps", [250, 500, 1000, 2000, 4000, 8000])
    firing_reward_high = trial.suggest_categorical("firing_reward_high", [50, 100, 150])
    time_penalty_factor = trial.suggest_categorical("time_penalty_factor", [-80, -100, -120, -150])
    hold_penalty_high = trial.suggest_categorical("hold_penalty_high", [-8, -10, -12])
    eval_interval = trial.suggest_categorical("eval_interval", [50, 100, 200])

    # Suggest reward parameters
    reward_excellent = trial.suggest_categorical("reward_excellent", [80, 100, 120])
    reward_good = trial.suggest_categorical("reward_good", [60, 80, 100])
    reward_minimum = trial.suggest_categorical("reward_minimum", [40, 60, 80])
    reward_fair = trial.suggest_categorical("reward_fair", [20, 40, 60])
    reward_poor = trial.suggest_categorical("reward_poor", [10, 20, 30])
    reward_failure = trial.suggest_categorical("reward_failure", [-50, -30, -10])
    hold_penalty_base = trial.suggest_categorical("hold_penalty_base", [-15, -10, -5])
    opportunity_cost_excellent = trial.suggest_categorical("opportunity_cost_excellent", [-40, -30, -20])
    opportunity_cost_good = trial.suggest_categorical("opportunity_cost_good", [-20, -15, -10])
    opportunity_cost_minimum = trial.suggest_categorical("opportunity_cost_minimum", [-10, -8, -5])

    # Run training
    result = run_training(epsilon_decay_steps, firing_reward_high, time_penalty_factor, hold_penalty_high, eval_interval,
                         reward_excellent, reward_good, reward_minimum, reward_fair, reward_poor, reward_failure,
                         hold_penalty_base, opportunity_cost_excellent, opportunity_cost_good, opportunity_cost_minimum)

    if result.success:
        # Calculate score (higher is better)
        score = result.avg_hp - 0.1 * ((result.avg_steps or 0) - 20) - 0.1 * result.hp_std
        return score
    else:
        # Return very low score for failed runs
        return -10.0


if __name__ == "__main__":
    main()