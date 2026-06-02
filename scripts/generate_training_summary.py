#!/usr/bin/env python3
"""
Generate a concise summary of DQN training/evaluation results.

Reads the CSV produced by evaluate_dqn_trajectories.py and creates:
- Metrics printed to console
- Plots saved to reports/
- A Markdown report summarizing findings
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

INPUT_CSV = Path(__file__).parent.parent / "results" / "evaluation_summary.csv"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}. Run evaluate_dqn_trajectories.py first.")
    df = pd.read_csv(csv_path)
    # Expected columns: scenario_name, initial_range, steps, fired_at_step, final_hp
    # Ensure numeric types
    df['initial_range'] = pd.to_numeric(df['initial_range'], errors='coerce')
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
    df['final_hp'] = pd.to_numeric(df['final_hp'], errors='coerce')
    # fired_at_step can be NaN
    df['fired'] = df['fired_at_step'].notnull()
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    metrics['n_trajectories'] = len(df)
    metrics['firing_rate'] = df['fired'].mean() if len(df) else 0.0
    metrics['mean_final_hp'] = df['final_hp'].mean()
    metrics['std_final_hp'] = df['final_hp'].std()
    metrics['median_final_hp'] = df['final_hp'].median()
    metrics['min_final_hp'] = df['final_hp'].min()
    metrics['max_final_hp'] = df['final_hp'].max()
    metrics['mean_steps'] = df['steps'].mean()

    # HP by range buckets
    bins = [0, 500, 1000, 2000, 4000, np.inf]
    labels = ["<=500m", "500-1k", "1k-2k", "2k-4k", ">4k"]
    df['range_bucket'] = pd.cut(df['initial_range'], bins=bins, labels=labels, include_lowest=True)
    hp_by_bucket = df.groupby('range_bucket')['final_hp'].agg(['mean', 'count']).reset_index()
    metrics['hp_by_range_bucket'] = hp_by_bucket

    # Firing step statistics where fired
    fired_steps = df.loc[df['fired'], 'fired_at_step']
    if not fired_steps.empty:
        metrics['fired_step_mean'] = fired_steps.mean()
        metrics['fired_step_std'] = fired_steps.std()
        metrics['fired_step_min'] = fired_steps.min()
        metrics['fired_step_max'] = fired_steps.max()
    else:
        metrics['fired_step_mean'] = metrics['fired_step_std'] = metrics['fired_step_min'] = metrics['fired_step_max'] = None

    return metrics


def plot_hp_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df['final_hp'], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel('Final Hit Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Hit Probability')
    fig.tight_layout()
    out = REPORTS_DIR / 'hp_distribution.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_hp_vs_range(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df['initial_range'], df['final_hp'], c=np.where(df['fired'], 'tab:red', 'tab:blue'), alpha=0.8)
    ax.set_xlabel('Initial Range (m)')
    ax.set_ylabel('Final Hit Probability')
    ax.set_title('Final HP vs Initial Range')
    # Trend line (simple linear fit)
    if len(df) >= 2 and df['initial_range'].notnull().all():
        x = df['initial_range'].values
        y = df['final_hp'].values
        # Guard against NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            xp = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xp, coeffs[0]*xp + coeffs[1], color='gray', linestyle='--', label='Linear trend')
            ax.legend()
    fig.tight_layout()
    out = REPORTS_DIR / 'hp_vs_range.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_firing_step_hist(df: pd.DataFrame) -> Path | None:
    fired_steps = df['fired_at_step'].dropna()
    if fired_steps.empty:
        return None
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(fired_steps, bins=10, color='salmon', edgecolor='black')
    ax.set_xlabel('Firing Step')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Firing Steps')
    fig.tight_layout()
    out = REPORTS_DIR / 'firing_step_histogram.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_hp_by_range_bucket(metrics: dict) -> Path:
    hpb = metrics['hp_by_range_bucket']
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(hpb['range_bucket'].astype(str), hpb['mean'], color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Initial Range Bucket')
    ax.set_ylabel('Mean Final HP')
    ax.set_title('Mean Final HP by Initial Range Bucket')
    fig.tight_layout()
    out = REPORTS_DIR / 'hp_by_range_bucket.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def write_markdown_report(metrics: dict, assets: dict) -> Path:
    md = REPORTS_DIR / 'training_summary.md'
    lines = []
    lines.append('# DQN Training Evaluation Summary\n')
    lines.append('This report summarizes decision-making performance based on evaluation results.\n')

    lines.append('## Key Metrics\n')
    lines.append(f"- Total trajectories: {metrics['n_trajectories']}\n")
    lines.append(f"- Firing rate: {metrics['firing_rate']*100:.1f}%\n")
    lines.append(f"- Mean final HP: {metrics['mean_final_hp']:.3f}\n")
    lines.append(f"- Std final HP: {metrics['std_final_hp']:.3f}\n")
    lines.append(f"- Median final HP: {metrics['median_final_hp']:.3f}\n")
    lines.append(f"- Min/Max final HP: {metrics['min_final_hp']:.3f} / {metrics['max_final_hp']:.3f}\n")
    lines.append(f"- Mean episode steps: {metrics['mean_steps']:.1f}\n")

    if metrics['fired_step_mean'] is not None:
        lines.append('## Firing Timing\n')
        lines.append(f"- Mean firing step: {metrics['fired_step_mean']:.1f}\n")
        lines.append(f"- Std firing step: {metrics['fired_step_std']:.1f}\n")
        lines.append(f"- Min/Max firing step: {metrics['fired_step_min']:.0f} / {metrics['fired_step_max']:.0f}\n")

    lines.append('## Plots\n')
    lines.append(f"![HP Distribution]({assets['hp_distribution']})\n")
    lines.append(f"![HP vs Range]({assets['hp_vs_range']})\n")
    if assets.get('firing_step_histogram'):
        lines.append(f"![Firing Step Histogram]({assets['firing_step_histogram']})\n")
    lines.append(f"![HP by Range Bucket]({assets['hp_by_range_bucket']})\n")

    # Add table for HP by range bucket
    hpb = metrics['hp_by_range_bucket']
    lines.append('\n## Mean Final HP by Range Bucket\n')
    lines.append('| Range Bucket | Mean Final HP | Samples |\n')
    lines.append('|---|---:|---:|\n')
    for _, row in hpb.iterrows():
        lines.append(f"| {row['range_bucket']} | {row['mean']:.3f} | {int(row['count'])} |\n")

    md.write_text(''.join(lines))
    return md


def main():
    df = load_data(INPUT_CSV)
    metrics = compute_metrics(df)

    # Generate plots
    assets = {}
    assets['hp_distribution'] = str(plot_hp_distribution(df))
    assets['hp_vs_range'] = str(plot_hp_vs_range(df))
    fs_hist = plot_firing_step_hist(df)
    if fs_hist:
        assets['firing_step_histogram'] = str(fs_hist)
    assets['hp_by_range_bucket'] = str(plot_hp_by_range_bucket(metrics))

    # Write report
    md_path = write_markdown_report(metrics, assets)

    # Console summary
    print('Summary metrics:')
    print(f"- Trajectories: {metrics['n_trajectories']}")
    print(f"- Firing rate: {metrics['firing_rate']*100:.1f}%")
    print(f"- Mean final HP: {metrics['mean_final_hp']:.3f} (std {metrics['std_final_hp']:.3f})")
    print(f"- Median final HP: {metrics['median_final_hp']:.3f}")
    print(f"- Min/Max final HP: {metrics['min_final_hp']:.3f} / {metrics['max_final_hp']:.3f}")
    print(f"- Mean steps: {metrics['mean_steps']:.1f}")
    if metrics['fired_step_mean'] is not None:
        print(f"- Firing step mean: {metrics['fired_step_mean']:.1f} (std {metrics['fired_step_std']:.1f})")
    print(f"Report written to: {md_path}")
    print(f"Assets in: {REPORTS_DIR}")


if __name__ == '__main__':
    main()