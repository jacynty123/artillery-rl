# Advanced Ballistics Hit Probability Calculator

A Python implementation of hit probability calculation for projectiles against moving targets, featuring a 6-DOF ballistics model, Jacobian-based error propagation, Monte Carlo estimation, and a DQN-based RL training pipeline.

> The `mc2/` directory contains legacy MATLAB code. All active development uses `src/` and `rl_training/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac

pip install numpy scipy matplotlib pytest
pip install gymnasium torch tensorboard  # for RL training
pip install seaborn pandas optuna        # for parameter sweep and reports
```

## Pipeline

### Step 1: Activate environment

```bash
source .venv/bin/activate
```

### Step 2: Hyperparameter optimization (optional)

```bash
# Reward structure optimization — recommended before training
python scripts/parameter_sweep.py --mode reward_optuna --trials 50 --timesteps 10000 --study_name dqn_reward_optimization

# General hyperparameter optimization
python scripts/parameter_sweep.py --mode optuna --trials 50 --timesteps 5000

# Traditional grid search
python scripts/parameter_sweep.py --mode grid
```

Outputs written to `config/`:
- `best_<study_name>_params.json` — best parameters found
- `<study_name>.db` — Optuna study database
- `*_param_importances.html`, `*_optimization_history.html`, `*_parallel_coordinate.html`

### Step 3: Train DQN with curriculum learning

```bash
# All 4 phases with optimized parameters
python -m rl_training.train.train_dqn_curriculum \
    --start_phase 1 --end_phase 4 \
    --timesteps 15000 \
    --config config/best_dqn_reward_optimization_params.json

# Specific phases only
python -m rl_training.train.train_dqn_curriculum \
    --start_phase 1 --end_phase 2 \
    --timesteps 15000

# Without config (default parameters)
python -m rl_training.train.train_dqn_curriculum \
    --start_phase 1 --end_phase 4 \
    --timesteps 15000
```

Training phases:
- **Phase 1**: Static targets, slow motion
- **Phase 2**: Fast approaches, medium targets
- **Phase 3**: Complex 3D motion, maneuvering targets
- **Phase 4**: Extreme long-range scenarios (2.5–3 km)

Checkpoints saved to `rl_training/models/checkpoints/dqn_curriculum_phase{1..4}.pth`.

### Step 4: Evaluate trained model

**Single run** — evaluates 5 curriculum scenarios with a single seed:

```bash
python scripts/evaluate_dqn_trajectories.py
```

Outputs:
- `results/trajectory_evaluation.png` — HP traces and 3D trajectories
- `results/evaluation_summary.csv` — per-scenario tabular results
- `results/scenario_plots/` — individual scenario plots

**Multi-run robustness analysis** — repeats evaluation across N seeds with small random perturbations (range ±5%, velocity ±10%, noise ±20%) to measure policy stability:

```bash
python scripts/evaluate_dqn_trajectories.py --num_runs 10
```

Additional outputs when `--num_runs > 1`:
- `results/evaluation_summary_multiple_runs.csv` — per-scenario statistics across all runs
- Console table with mean HP, std HP, min/max HP, mean/std firing step, firing rate %, and CV% (coefficient of variation — lower means more robust policy)

### Step 5: Generate reports

Requires `results/evaluation_summary.csv` from Step 4.

```bash
# Markdown report with plots
python scripts/generate_training_summary.py
```

Outputs to `reports/`: `training_summary.md`, `hp_distribution.png`, `hp_vs_range.png`, `hp_by_range_bucket.png`, `firing_step_histogram.png`.

```bash
# LaTeX report (requires results/evaluation_details.json from Step 4)
python scripts/generate_training_summary_tex.py
```

Output: `reports/training_summary.tex` with per-scenario trajectory details.

### Step 6: Hit probability scenario analysis

Analyzes engagement scenarios (500–2000m, multiple target and ammo types) using the ballistics model directly, without the RL agent:

```bash
python scripts/hit_probability_scenarios.py
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir=./logs/
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/
```

## Project structure

```
src/                              # Core ballistics library
rl_training/
├── environment.py
├── agents/                       # DQN, SAC implementations
├── curriculum/                   # Scenario generation
├── train/                        # train_dqn_curriculum.py, train_sac_her.py
└── infrastructure/               # Config management, monitoring
scripts/
├── parameter_sweep.py            # Optuna / grid hyperparameter search
├── evaluate_dqn_trajectories.py  # Model evaluation (single & multi-run)
├── generate_training_summary.py  # Markdown report
├── generate_training_summary_tex.py  # LaTeX report
└── hit_probability_scenarios.py  # Standalone ballistics scenario analysis
config/                           # Saved best parameter sets
results/                          # Evaluation CSVs and plots
reports/                          # Generated reports
docs/                             # LaTeX papers and documentation
mc2/                              # Legacy MATLAB code (not maintained)
```