#!/usr/bin/env python3
"""
Multi-gun firing-solution demo.

Builds a battery of ``--num_guns`` guns and, against a single target, prints each gun's
firing solution (elevation / azimuth / feasibility) and hit probability, plus the
combined salvo kill probability over the feasible guns. Pure ballistics — no RL agent.

Examples
--------
    python scripts/battery_firing_solution.py --num_guns 4 --range 1500
    python scripts/battery_firing_solution.py --num_guns 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gun import Battery, solve_firing_solution, hit_probability_for_gun
from src.hit_probability import HitProbabilityCalculator

TARGET_DIMENSIONS = (10.0, 8.0, 4.0)  # L, W, H (m) — realistic vehicle-sized target


def _default_measurement_uncertainty():
    """A representative 9x9 input covariance for the hit-probability Monte Carlo.

    Order: [muzzle_velocity, elevation, azimuth, tx, ty, tz, tvx, tvy, tvz].
    Uses modern-radar-grade position noise so hit probabilities are representative.
    """
    sigmas = np.array([10, 0.002, 0.002, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0], dtype=float)
    return np.diag(sigmas ** 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Battery firing-solution demo (N guns vs one target).")
    p.add_argument("--num_guns", type=int, default=1, help="Number of guns in the battery")
    p.add_argument("--spacing", type=float, default=50.0, help="Spacing between guns (m)")
    p.add_argument("--axis", choices=["x", "y", "z"], default="y", help="Battery line axis")
    p.add_argument("--range", dest="range_m", type=float, default=1500.0,
                   help="Target downrange distance / x (m)")
    p.add_argument("--target_y", type=float, default=0.0, help="Target y (m)")
    p.add_argument("--target_z", type=float, default=100.0, help="Target z / altitude (m)")
    p.add_argument("--target_vx", type=float, default=0.0, help="Target vx (m/s)")
    p.add_argument("--target_vy", type=float, default=0.0, help="Target vy (m/s)")
    p.add_argument("--target_vz", type=float, default=0.0, help="Target vz (m/s)")
    p.add_argument("--ammo", default="tpt", help="Ammunition type")
    p.add_argument("--muzzle_velocity", type=float, default=1000.0, help="Muzzle velocity (m/s)")
    p.add_argument("--n_samples", type=int, default=500, help="Hit-probability Monte Carlo samples")
    return p.parse_args(argv)


def engage(battery, target_pos, target_vel, calc, n_samples=500, measurement_uncertainty=None):
    """Compute per-gun firing solutions + hit probabilities and the combined salvo P_kill."""
    if measurement_uncertainty is None:
        measurement_uncertainty = _default_measurement_uncertainty()
    target_pos = np.asarray(target_pos, dtype=float)
    rows = []
    for gun in battery:
        sol = solve_firing_solution(gun, target_pos, target_vel)
        hp = hit_probability_for_gun(
            calc, gun, target_pos, target_vel,
            measurement_uncertainty=measurement_uncertainty,
            elevation_angle=sol.elevation, azimuth_angle=sol.azimuth,
            n_samples=n_samples,
        )
        rng = float(np.linalg.norm(target_pos - gun.position))
        rows.append({
            "gun_id": gun.gun_id,
            "position": gun.position,
            "range_m": rng,
            "elevation": sol.elevation,
            "azimuth": sol.azimuth,
            "feasible": sol.feasible,
            "reason": sol.infeasible_reason,
            "hp": float(hp),
        })
    feasible_hps = [r["hp"] for r in rows if r["feasible"]]
    p_kill = 1.0 - float(np.prod([1.0 - hp for hp in feasible_hps])) if feasible_hps else 0.0
    return rows, p_kill


def _print_report(battery, target_pos, target_vel, rows, p_kill):
    print("=" * 78)
    print(f"Battery engagement: {battery.n_guns} gun(s) vs target at "
          f"[{target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f}] m, "
          f"vel [{target_vel[0]:.0f}, {target_vel[1]:.0f}, {target_vel[2]:.0f}] m/s")
    print("=" * 78)
    print(f"{'Gun':<4} {'Position(x,y,z)':<20} {'Range':<8} {'Elev':<8} "
          f"{'Azim':<8} {'Feasible':<9} {'HP':<6}")
    print("-" * 78)
    for r in rows:
        pos = r["position"]
        pos_str = f"({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})"
        print(f"{r['gun_id']:<4} {pos_str:<20} {r['range_m']:<8.0f} "
              f"{np.degrees(r['elevation']):<8.1f} {np.degrees(r['azimuth']):<8.1f} "
              f"{('yes' if r['feasible'] else 'no'):<9} {r['hp']:<6.3f}")
    print("-" * 78)
    print(f"Combined salvo kill probability (feasible guns): {p_kill:.3f}")
    print("=" * 78)


def main(argv=None):
    args = parse_args(argv)
    battery = Battery.line(
        n_guns=args.num_guns, spacing=args.spacing, ammo_type=args.ammo,
        muzzle_velocity=args.muzzle_velocity, axis=args.axis,
    )
    target_pos = np.array([args.range_m, args.target_y, args.target_z])
    target_vel = np.array([args.target_vx, args.target_vy, args.target_vz])
    calc = HitProbabilityCalculator(
        projectile_velocity=args.muzzle_velocity,
        target_dimensions=TARGET_DIMENSIONS, ammo_type=args.ammo,
    )
    rows, p_kill = engage(battery, target_pos, target_vel, calc, n_samples=args.n_samples)
    _print_report(battery, target_pos, target_vel, rows, p_kill)
    return rows, p_kill


if __name__ == "__main__":
    main()
