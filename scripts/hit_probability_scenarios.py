#!/usr/bin/env python3
"""
Hit Probability Analysis for Multiple Engagement Scenarios

This script analyzes hit probability for various realistic engagement scenarios
ranging from 500m to 2000m, with different target types, projectile types,
and uncertainty levels.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Ensure project root is on the path so src/ imports resolve
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hit_probability import HitProbabilityCalculator
# Use find_optimal_firing_angles to get best firing solution
from src.find_optimal_firing_angles import (
        find_optimal_firing_angles,
    )
# Import trajectory simulator and Kalman filter for dynamic covariance
from src.trajectory_simulator import TrajectorySimulator
from src.kalman_filter import TargetKalmanFilter, MeasurementType


def create_scenario(range_m, target_length, target_width, target_height,
                   target_vx, target_vy, target_vz, projectile_type="tpt",
                   projectile_velocity=1180.0, tracking_duration=10.0,
                   measurement_noise_std=None, process_noise_std=None):
    """
    Create a single scenario with specified parameters.
    Always uses dynamic covariance calculation.

    Args:
        range_m: Engagement range in meters
        target_length, target_width, target_height: Target dimensions in meters
        target_vx, target_vy, target_vz: Target velocity components in m/s
        projectile_type: Type of projectile ("tpt")
        projectile_velocity: Projectile muzzle velocity in m/s
        tracking_duration: Tracking duration in seconds (for dynamic covariance)
        measurement_noise_std: Measurement noise std [x,y,z] in meters
        process_noise_std: Process noise std [ax,ay,az] in m/s²

    Returns:
        Scenario dictionary
    """
    # Baseline projectile uncertainties (always used)
    baseline_projectile_covariance = np.array([
        [6.25, 0.0, 0.0],  # Velocity uncertainty: 2.5 m/s
        [0.0, 1.9e-7, 0.0],  # Elevation uncertainty: ~0.025°
        [0.0, 0.0, 1.9e-7],  # Azimuth uncertainty: ~0.025°
    ])

    # Note: baseline_target_covariance removed - only dynamic covariance used

    # Default dynamic covariance parameters if not provided
    if measurement_noise_std is None:
        # More realistic radar measurement noise at range
        noise_scale = range_m / 1000.0  # Normalize to 1000m baseline
        measurement_noise_std = np.array([2.0, 2.0, 3.0]) * noise_scale  # Better accuracy

    if process_noise_std is None:
        # Default process noise based on target speed
        speed = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
        if speed > 100:  # Fast target (fighter)
            process_noise_std = np.array([2.0, 2.0, 1.0])  # Reduced from [5,5,2]
        elif speed > 50:  # Medium speed (bomber/helicopter)
            process_noise_std = np.array([1.0, 1.0, 0.5])  # Reduced from [2,2,1]
        else:  # Slow target (transport)
            process_noise_std = np.array([0.3, 0.3, 0.1])  # Reduced from [0.5,0.5,0.2]

    scenario = {
        "name": f"Custom Scenario {range_m}m",
        "description": f"{range_m}m engagement with custom target parameters",
        "range": float(range_m),
        "target_dims": (float(target_length), float(target_width), float(target_height)),
        "target_velocity": np.array([float(target_vx), float(target_vy), float(target_vz)]),
        "projectile_type": projectile_type,
        "projectile_velocity": float(projectile_velocity),
        "elevation_angle": np.radians(5.0),  # Default elevation
        "azimuth_angle": np.radians(2.0),    # Default azimuth
        "projectile_covariance": baseline_projectile_covariance,
        # target_covariance removed - only dynamic covariance used
        "use_dynamic_covariance": True,  # Always use dynamic covariance
        "tracking_duration": float(tracking_duration),
        "measurement_noise_std": measurement_noise_std,
        "process_noise_std": process_noise_std,
    }

    return scenario


def compute_dynamic_target_covariance(scenario, impact_time):
    """
    Compute dynamic target covariance using trajectory simulation and Kalman filtering.

    Args:
        scenario: Scenario dictionary with tracking parameters
        impact_time: Time of projectile impact (seconds)

    Returns:
        Dynamic covariance matrix (6x6) representing target state uncertainty at impact time
    """
    # Extract scenario parameters
    tracking_duration = scenario["tracking_duration"]
    measurement_noise_std = scenario["measurement_noise_std"]
    process_noise_std = scenario["process_noise_std"]
    target_init_pos = np.array([scenario["range"], 0.0, 50.0])
    target_init_vel = scenario["target_velocity"]

    # Initialize target state [x, y, z, vx, vy, vz]
    initial_state = np.concatenate([target_init_pos, target_init_vel])

    # Initialize trajectory simulator
    trajectory_sim = TrajectorySimulator(
        initial_state=initial_state,
        process_noise_std=process_noise_std,
        dt=0.1  # 10 Hz simulation
    )

    # Initialize Kalman filter
    # More realistic initial uncertainty for radar detection at range
    # Position uncertainty: ~3m std (9m²), Velocity uncertainty: ~2m/s std (4m²/s²)
    initial_covariance = np.zeros((6, 6))
    initial_covariance[0:3, 0:3] = np.eye(3) * 9.0    # Position: 3m std
    initial_covariance[3:6, 3:6] = np.eye(3) * 4.0    # Velocity: 2m/s std

    kf = TargetKalmanFilter(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
        measurement_type=MeasurementType.POSITION_ONLY,  # Radar position measurements
        dt=0.1
    )

    # Simulate tracking history
    # Start tracking some time before impact
    tracking_start_time = max(0.0, impact_time - tracking_duration)
    current_time = tracking_start_time

    print(f"  Simulating {tracking_duration:.1f}s of target tracking...")

    while current_time < impact_time:
        # Simulate true target movement
        true_state = trajectory_sim.step()

        # Generate noisy measurement (radar detection)
        measurement_noise = np.random.normal(0, measurement_noise_std)
        measurement = true_state[:3] + measurement_noise  # Position-only measurement

        # Kalman filter prediction and update
        kf.predict()
        kf.update(measurement)

        current_time += 0.1

    # Extract final covariance at impact time
    dynamic_covariance = kf.get_covariance()

    # Get filter diagnostics
    diagnostics = kf.get_filter_status()
    print(f"  Filter diagnostics: NIS={diagnostics['nis']:.2f}, cond#={diagnostics['condition_number']:.1f}")

    return dynamic_covariance


def analyze_scenario(scenario):
    """
    Analyze a single engagement scenario and calculate hit probability.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {scenario['name']}")
    print(f"{'='*60}")
    print(f"Description: {scenario['description']}")
    print(f"Range: {scenario['range']} m")
    print(f"Target dimensions: {scenario['target_dims']} m (L×W×H)")
    print(f"Target velocity: {scenario['target_velocity']} m/s")
    print(
        f"Projectile: {scenario['projectile_type'].upper()} at {scenario['projectile_velocity']} m/s"
    )

    # Check if dynamic covariance is enabled (should always be true now)
    use_dynamic_covariance = scenario.get("use_dynamic_covariance", True)
    if use_dynamic_covariance:
        print(f"Dynamic covariance enabled: {scenario['tracking_duration']}s tracking, "
              f"measurement noise {scenario['measurement_noise_std']}m")

    # Shooter at origin, target at desired range, 0, 50
    shooter_pos = np.array([0.0, 0.0, 0.0])
    target_init = np.array([scenario["range"], 0.0, 50.0])
    target_vel = scenario["target_velocity"]
    projectile_speed = scenario["projectile_velocity"]
    ammo_type = scenario["projectile_type"]

    elev, azim, impact_time, intersection_point, min_dist = find_optimal_firing_angles(
        shooter_pos, target_init, target_vel, projectile_speed, ammo_type, max_time=30.0
    )

    print(
        f"Optimal firing solution: elev={np.degrees(elev):.2f}°, azim={np.degrees(azim):.2f}°, t_impact={impact_time:.2f}s, min_dist={min_dist:.2f}m"
    )
    print(
        f"Intersection point: [{intersection_point[0]:.1f}, {intersection_point[1]:.1f}, {intersection_point[2]:.1f}] m"
    )

    # Initialize analytical calculator

    calc = HitProbabilityCalculator(
        projectile_velocity=projectile_speed,
        ammo_type=ammo_type,
        target_dimensions=scenario["target_dims"],
    )

    projectile_cov = scenario["projectile_covariance"]  # (v0, elev, azim)

    # Compute dynamic target covariance (always used now)
    print(f"Computing dynamic target covariance...")
    target_cov = compute_dynamic_target_covariance(scenario, impact_time)
    print(f"Dynamic covariance computed at impact time {impact_time:.2f}s")

    # Directly use scenario's initial target position and velocity, passing precomputed impact_time and intersection_point
    analytical_prob = calc.calculate_hit_probability_analytical(
        tuple(target_init),
        tuple(target_vel),
        projectile_param_cov=projectile_cov,
        target_param_cov=target_cov,
        elevation_angle=elev,
        azimuth_angle=azim,
        use_diagonal_approx=True,
        mc_refine_samples=1500,
        impact_time=impact_time,
        intersection_point=intersection_point,
    )

    # Propagate individual covariances
    error_prop = calc.error_propagation
    proj_state_cov = np.zeros((8, 8))
    proj_state_cov[3:6, 3:6] = projectile_cov
    Cp = error_prop.propagate_projectile_covariance(
        proj_state_cov,
        scenario["projectile_velocity"],
        scenario["elevation_angle"],
        scenario["azimuth_angle"],
        impact_time,
    )
    Ct = error_prop.propagate_target_covariance(
        target_cov, impact_time, include_vertical_velocity=True
    )

    # Use Cp for projectile position covariance; Ct for target position covariance
    intersection_covariance = Cp
    cov_target = Ct
    cov_projectile = intersection_covariance

    print(f"\nCovariance Matrices at Hit Point:")
    print(f"Target Position Covariance (m²):")
    print(f"  {cov_target[0]}")
    print(f"  {cov_target[1]}")
    print(f"  {cov_target[2]}")
    print(f"Projectile Position Covariance at Intersection (m²):")
    print(f"  {cov_projectile[0]}")
    print(f"  {cov_projectile[1]}")
    print(f"  {cov_projectile[2]}")

    # Analyze uncertainties
    eigvals, eigvecs = np.linalg.eigh(intersection_covariance)
    std_devs = np.sqrt(np.maximum(eigvals, 0))

    print(f"\nUncertainty Analysis:")
    print(
        f"Intersection point uncertainties: σx={std_devs[0]:.1f}m, σy={std_devs[1]:.1f}m, σz={std_devs[2]:.1f}m"
    )
    volume = (4 / 3) * np.pi * np.prod(std_devs)
    print(f"3D uncertainty ellipsoid volume: {volume:.0f} m³")

    print(f"\nHit Probability Results:")
    print(
        f"Analytical (diag+refine) hit probability: {analytical_prob:.6f} ({analytical_prob*100:.3f}%)"
    )

    # Calculate probability density at target center

    try:
        mv = multivariate_normal(
            mean=np.zeros(3),
            cov=intersection_covariance + cov_target,
            allow_singular=True,
        )
        density_at_center = mv.pdf(np.zeros(3))
        print(f"Probability density at relative center: {density_at_center:.2e} m⁻³")
    except Exception:
        print("Could not calculate probability density (singular combined covariance)")

    return {
        "scenario": scenario,
        "intersection_point": intersection_point,
        "intersection_covariance": intersection_covariance,
        "hit_probability_analytical": analytical_prob,
        "uncertainty_std": std_devs,
        "uncertainty_volume": volume,
    }


def create_summary_table(results, n_iterations):
    """
    Create a summary table of scenario results for multiple iterations.
    """
    print(f"\n{'='*95}")
    print(f"HIT PROBABILITY ANALYSIS SUMMARY ({n_iterations} iterations)")
    print(f"{'='*95}")

    if results:
        scenario = results[0]["scenario"]
        print(f"Scenario: {scenario['name']}")
        print(f"Range: {scenario['range']}m, Target: {scenario['target_dims'][0]:.0f}×{scenario['target_dims'][1]:.0f}×{scenario['target_dims'][2]:.0f}m")
        print(f"Target velocity: {scenario['target_velocity']} m/s")
        print(f"Dynamic covariance: Always enabled")
        if scenario.get('use_dynamic_covariance', False):
            print(f"Tracking duration: {scenario['tracking_duration']}s")
        print()

    # Calculate statistics across iterations
    probabilities = [r["hit_probability_analytical"] for r in results]
    uncertainties = [np.sqrt(np.sum(r["uncertainty_std"] ** 2)) for r in results]

    print(f"Results across {len(results)} iterations:")
    print(f"Hit Probability: mean={np.mean(probabilities)*100:.3f}%, std={np.std(probabilities)*100:.3f}%, min={np.min(probabilities)*100:.3f}%, max={np.max(probabilities)*100:.3f}%")
    print(f"Total Uncertainty: mean={np.mean(uncertainties):.2f}m, std={np.std(uncertainties):.2f}m")


def create_visualization(results, n_iterations):
    """
    Create visualization of hit probability distribution across iterations.
    """
    probabilities = [r["hit_probability_analytical"] * 100 for r in results]
    uncertainties = [np.sqrt(np.sum(r["uncertainty_std"] ** 2)) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Hit probability histogram
    ax1.hist(probabilities, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(probabilities):.3f}%')
    ax1.set_xlabel("Hit Probability (%)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Hit Probability Distribution ({n_iterations} iterations)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Uncertainty histogram
    ax2.hist(uncertainties, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(np.mean(uncertainties), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(uncertainties):.2f}m')
    ax2.set_xlabel("Total Position Uncertainty (m)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Position Uncertainty Distribution ({n_iterations} iterations)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "hit_probability_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as '{out}'")


def main(n_iterations=10, range_m=1000.0, target_length=12.0, target_width=6., target_height=3.,
         target_vx=150.0, target_vy=15.0, target_vz=0.0, projectile_type="tpt",
         projectile_velocity=1180.0, tracking_duration=10.0,
         measurement_noise_std=None, process_noise_std=None):
    """
    Main analysis function with configurable parameters.
    Always uses dynamic covariance calculation.

    Args:
        n_iterations: Number of Monte Carlo iterations to run
        range_m: Engagement range in meters
        target_length, target_width, target_height: Target dimensions in meters
        target_vx, target_vy, target_vz: Target velocity components in m/s
        projectile_type: Type of projectile
        projectile_velocity: Projectile muzzle velocity in m/s
        tracking_duration: Tracking duration in seconds (for dynamic covariance)
        measurement_noise_std: Measurement noise std [x,y,z] in meters (auto-scaled if None)
        process_noise_std: Process noise std [ax,ay,az] in m/s² (auto-scaled if None)
    """
    print("HIT PROBABILITY ANALYSIS WITH DYNAMIC COVARIANCE")
    print(f"Running {n_iterations} iterations for single scenario")
    print(f"Range: {range_m}m, Target: {target_length:.0f}×{target_width:.0f}×{target_height:.0f}m")
    print(f"Target velocity: [{target_vx:.1f}, {target_vy:.1f}, {target_vz:.1f}] m/s")
    print(f"Dynamic covariance: Always enabled")

    # Create single scenario with specified parameters
    scenario = create_scenario(
        range_m=range_m,
        target_length=target_length,
        target_width=target_width,
        target_height=target_height,
        target_vx=target_vx,
        target_vy=target_vy,
        target_vz=target_vz,
        projectile_type=projectile_type,
        projectile_velocity=projectile_velocity,
        tracking_duration=tracking_duration,
        measurement_noise_std=measurement_noise_std,
        process_noise_std=process_noise_std
    )

    # Run analysis for n iterations
    results = []
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        result = analyze_scenario(scenario)
        results.append(result)

    # Create summary table
    create_summary_table(results, n_iterations)

    # Create visualization
    create_visualization(results, n_iterations)

    print(f"\nAnalysis complete! {n_iterations} iterations completed.")
    print(f"Results saved to '{RESULTS_DIR / 'hit_probability_analysis.png'}'")

    return results


if __name__ == "__main__":
    # Default parameters - can be modified or passed as command line arguments
    main(
        n_iterations=5,  # Reduced for testing, increase for production
        range_m=1000.0,
        target_length=12.0,
        target_width=6.,
        target_height=3.,
        target_vx=150.0,
        target_vy=15.0,
        target_vz=0.0
    )
