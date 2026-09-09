#!/usr/bin/env python3
"""
Benchmark single-sample CPU inference latency for the trained DQN firing policy.

Measures forward-pass latency of QNetwork / DQN (14-128-128-2) on CPU,
reporting mean, median, P95, P99, and worst-case tail latency across N repetitions.
Exports results to results/inference_latency_benchmark.json.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rl_training.agents.dqn_components import DQN


def benchmark_dqn_latency(
    checkpoint_path: Path,
    num_warmup: int = 2000,
    num_runs: int = 50000,
    batch_size: int = 1,
    output_path: Path = None,
):
    print("=" * 60)
    print("DQN Policy Inference Latency Benchmark")
    print("=" * 60)

    device = torch.device("cpu")
    model = DQN(state_dim=14, action_dim=2).to(device)

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model.eval()

    # Synthetic single-state input tensor
    x = torch.randn(batch_size, 14, device=device)

    # Warmup runs to prime cache and JIT/CPU optimizations
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)

    # Timed benchmarking runs
    print(f"Executing {num_runs} timed inference runs (batch_size={batch_size})...")
    latencies_us = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    latencies_us = np.array(latencies_us)

    stats = {
        "hardware": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "platform": platform.platform(),
        },
        "software": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        },
        "benchmark_config": {
            "checkpoint": str(checkpoint_path.relative_to(project_root)),
            "batch_size": batch_size,
            "warmup_iterations": num_warmup,
            "timed_repetitions": num_runs,
            "input_dimension": 14,
            "output_dimension": 2,
            "total_parameters": sum(p.numel() for p in model.parameters()),
        },
        "latency_microseconds": {
            "mean": float(np.mean(latencies_us)),
            "std": float(np.std(latencies_us)),
            "median": float(np.median(latencies_us)),
            "p95": float(np.percentile(latencies_us, 95)),
            "p99": float(np.percentile(latencies_us, 99)),
            "min": float(np.min(latencies_us)),
            "max": float(np.max(latencies_us)),
        },
        "latency_milliseconds": {
            "mean": float(np.mean(latencies_us) / 1000.0),
            "median": float(np.median(latencies_us) / 1000.0),
            "p95": float(np.percentile(latencies_us, 95) / 1000.0),
            "p99": float(np.percentile(latencies_us, 99) / 1000.0),
            "max": float(np.max(latencies_us) / 1000.0),
        },
    }

    print("\nBenchmark Results:")
    print(f"  Model Parameters: {stats['benchmark_config']['total_parameters']:,}")
    print(f"  Mean Latency:     {stats['latency_microseconds']['mean']:.2f} µs ({stats['latency_milliseconds']['mean']:.4f} ms)")
    print(f"  Median Latency:   {stats['latency_microseconds']['median']:.2f} µs ({stats['latency_milliseconds']['median']:.4f} ms)")
    print(f"  95th Percentile:  {stats['latency_microseconds']['p95']:.2f} µs ({stats['latency_milliseconds']['p95']:.4f} ms)")
    print(f"  99th Percentile:  {stats['latency_microseconds']['p99']:.2f} µs ({stats['latency_milliseconds']['p99']:.4f} ms)")
    print(f"  Max Tail Latency: {stats['latency_microseconds']['max']:.2f} µs ({stats['latency_milliseconds']['max']:.4f} ms)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved benchmark metrics to: {output_path}")

    return stats


if __name__ == "__main__":
    ckpt = project_root / "rl_training" / "models" / "checkpoints" / "dqn_curriculum_phase4.pth"
    out = project_root / "results" / "inference_latency_benchmark.json"
    benchmark_dqn_latency(ckpt, output_path=out)
