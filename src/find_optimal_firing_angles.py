import numpy as np
from scipy.optimize import minimize
from src.advanced_ballistics import (
    AdvancedProjectileMotion,
    AmmoParameters,
)


def find_optimal_firing_angles(
    shooter_pos,
    target_init,
    target_vel,
    projectile_speed,
    ammo_type,
    max_time=30.0,
    adaptive_mode="two_phase",  # Options: "fixed", "two_phase", "distance_adaptive"
):
    """
    Find the optimal elevation and azimuth for the projectile to hit a moving target.
    Uses full ODE integration for the projectile and CV for the target.

    Args:
        shooter_pos: Shooter position [x, y, z]
        target_init: Initial target position [x, y, z]
        target_vel: Target velocity [vx, vy, vz]
        projectile_speed: Initial projectile velocity (m/s)
        ammo_type: Ammunition type ("tpt", etc.)
        max_time: Maximum simulation time (s)
        adaptive_mode: Stepping strategy
            - "fixed": Original fixed time step
            - "two_phase": Coarse then fine stepping (default)
            - "distance_adaptive": Step size adapts to distance rate of change

    Returns: (elevation [rad], azimuth [rad], time_to_impact [s], hit_point [3], min_distance)
    """
    if ammo_type == "tpt":
        ammo = AmmoParameters.create_tpt_ammo()
    else:
        raise ValueError("Unsupported ammo_type")
    ballistics = AdvancedProjectileMotion(ammo)

    def closest_approach_for_angles(angles):
        elevation, azimuth = angles
        # print(
        #     f"\n[DEBUG] Trying angles: elevation={np.degrees(elevation):.3f} deg, azimuth={np.degrees(azimuth):.3f} deg"
        # )

        if adaptive_mode == "fixed":
            return _fixed_step_approach(angles)
        elif adaptive_mode == "two_phase":
            return _two_phase_adaptive_approach(angles)
        elif adaptive_mode == "distance_adaptive":
            return _distance_adaptive_approach(angles)
        else:
            return _fixed_step_approach(angles)

    def _fixed_step_approach(angles):
        """Original fixed time step approach."""
        elevation, azimuth = angles

        # Integrate projectile trajectory with fixed step
        traj_t, traj = ballistics.calculate_trajectory(
            initial_position=shooter_pos,
            initial_velocity=projectile_speed,
            elevation_angle=elevation,
            azimuth_angle=azimuth,
            max_time=max_time,
            time_step=0.005,  # Fixed fine step
        )

        return _find_minimum_distance(traj_t, traj)

    def _two_phase_adaptive_approach(angles):
        """Two-phase approach: coarse scan then fine refinement."""
        elevation, azimuth = angles

        # Phase 1: Coarse trajectory scan for rough minimum
        coarse_step = max(0.02, max_time / 1000)  # Adaptive based on max_time
        traj_t, traj = ballistics.calculate_trajectory(
            initial_position=shooter_pos,
            initial_velocity=projectile_speed,
            elevation_angle=elevation,
            azimuth_angle=azimuth,
            max_time=max_time,
            time_step=coarse_step,
        )

        # Find rough minimum with coarse steps
        min_dist, best_t, best_p = _find_minimum_distance(traj_t, traj)
        rough_min_time = best_t

        # Phase 2: Fine-grained search around minimum if found
        if rough_min_time is not None:
            # Define refined search window
            time_window = max(
                0.5, rough_min_time * 0.2
            )  # 20% of impact time or 0.5s minimum
            fine_start = max(0.0, rough_min_time - time_window)
            fine_end = min(max_time, rough_min_time + time_window)
            fine_step = coarse_step / 10  # 10x finer resolution

            # Recalculate with fine steps in the region of interest
            fine_traj_t, fine_traj = ballistics.calculate_trajectory(
                initial_position=shooter_pos,
                initial_velocity=projectile_speed,
                elevation_angle=elevation,
                azimuth_angle=azimuth,
                max_time=fine_end,
                time_step=fine_step,
            )

            # Refined search in the critical region
            fine_min_dist, fine_best_t, fine_best_p = _find_minimum_distance(
                fine_traj_t, fine_traj, time_range=(fine_start, fine_end)
            )

            # Use finer result if it's better
            if fine_min_dist < min_dist:
                min_dist, best_t, best_p = fine_min_dist, fine_best_t, fine_best_p

        return min_dist, best_t, best_p

    def _distance_adaptive_approach(angles):
        """Distance-adaptive approach: step size adapts to distance change rate."""
        elevation, azimuth = angles

        # Start with coarse step to get trajectory shape
        base_step = max(0.01, max_time / 2000)
        traj_t, traj = ballistics.calculate_trajectory(
            initial_position=shooter_pos,
            initial_velocity=projectile_speed,
            elevation_angle=elevation,
            azimuth_angle=azimuth,
            max_time=max_time,
            time_step=base_step,
        )

        # Calculate distance change rates
        distances = []
        for i, t in enumerate(traj_t):
            projectile_pos = np.array([traj[i, 1], traj[i, 2], traj[i, 0]])
            target_pos = target_init + target_vel * t
            dist = np.linalg.norm(projectile_pos - target_pos)
            distances.append(dist)

        # Find regions of rapid distance change
        min_dist_idx = np.argmin(distances)
        min_dist_time = traj_t[min_dist_idx]

        # Adaptive refinement around critical region
        critical_window = max(0.3, min_dist_time * 0.15)
        critical_start = max(0.0, min_dist_time - critical_window)
        critical_end = min(max_time, min_dist_time + critical_window)

        # Ultra-fine step in critical region
        ultra_fine_step = base_step / 20
        critical_traj_t, critical_traj = ballistics.calculate_trajectory(
            initial_position=shooter_pos,
            initial_velocity=projectile_speed,
            elevation_angle=elevation,
            azimuth_angle=azimuth,
            max_time=critical_end,
            time_step=ultra_fine_step,
        )

        return _find_minimum_distance(
            critical_traj_t, critical_traj, time_range=(critical_start, critical_end)
        )

    def _find_minimum_distance(traj_t, traj, time_range=None):
        """Helper function to find minimum distance in trajectory."""
        min_dist = np.inf
        best_t = None
        best_p = None
        best_target_pos = None

        for i, t in enumerate(traj_t):
            # Skip if outside time range
            if time_range is not None:
                if t < time_range[0] or t > time_range[1]:
                    continue

            # FIXED: Correct coordinate extraction - traj[i, 0:3] contains [z, x, y]
            projectile_pos = np.array([traj[i, 1], traj[i, 2], traj[i, 0]])  # x, y, z
            target_pos = target_init + target_vel * t
            dist = np.linalg.norm(projectile_pos - target_pos)

            if dist < min_dist:
                min_dist = dist
                best_t = t
                best_p = projectile_pos.copy()
                best_target_pos = target_pos.copy()

        # print(f"[DEBUG] Closest approach: min_dist={min_dist:.3f} m at t={best_t:.3f} s")
        # print(f"[DEBUG] Projectile: {best_p}, Target: {best_target_pos}")
        return min_dist, best_t, best_p

    def objective(angles):
        min_dist, _, _ = closest_approach_for_angles(angles)
        # Add penalty for solutions that don't reach the target area
        penalty = 0
        if min_dist > 100:  # If we're very far, add distance penalty
            penalty = (min_dist - 100) * 0.1
        return min_dist + penalty

    # Better initial guess that accounts for target motion
    rel_pos = target_init - shooter_pos
    distance = np.linalg.norm(rel_pos)

    if distance > 1e-8:
        # Estimate time to target (account for target motion)
        closing_speed = projectile_speed - np.dot(target_vel, rel_pos) / distance
        if closing_speed > 0:
            estimated_time = distance / closing_speed
        else:
            estimated_time = distance / projectile_speed

        # Predict target position at estimated time
        predicted_target = target_init + target_vel * estimated_time
        rel_predicted = predicted_target - shooter_pos
        r_predicted = np.linalg.norm(rel_predicted)

        # Basic elevation accounting for gravity (very rough estimate)
        g = 9.81
        elev0 = np.arctan2(rel_predicted[2], np.linalg.norm(rel_predicted[:2]))
        # Add gravity drop compensation
        elev0 += (0.5 * g * estimated_time**2) / (projectile_speed * estimated_time)

        azim0 = np.arctan2(rel_predicted[1], rel_predicted[0])
    else:
        elev0 = 0.0
        azim0 = 0.0

    x0 = [elev0, azim0]
    # print(
    #     f"[DEBUG] Initial guess: elev={np.degrees(elev0):.3f}°, azim={np.degrees(azim0):.3f}°"
    # )

    # Try multiple optimization strategies
    best_result = None
    best_min_dist = np.inf

    # Strategy 1: Direct optimization with wider bounds
    bounds = [
        (np.radians(-15), np.radians(45)),  # Wider elevation bounds
        # print(f"[DEBUG] Closest approach: min_dist={min_dist:.3f} m at t={best_t:.3f} s")
        # print(f"[DEBUG] Projectile: {best_p}, Target: {best_target_pos}")
    ]

    methods = ["L-BFGS-B", "Powell", "Nelder-Mead"]

    for method in methods:
        # print(f"\n[DEBUG] Trying optimization method: {method}")
        try:
            res = minimize(
                objective,
                x0=x0,
                bounds=bounds if method == "L-BFGS-B" else None,
                method=method,
                options=(
                    {"ftol": 1e-2, "maxiter": 50}
                    if method != "Nelder-Mead"
                    else {"maxiter": 100}
                ),
            )

            min_dist, t_impact, hit_point = closest_approach_for_angles(res.x)
            # print(f"[DEBUG] Method {method}: min_dist={min_dist:.3f}m")

            if min_dist < best_min_dist:
                best_min_dist = min_dist
                best_result = (res.x[0], res.x[1], t_impact, hit_point, min_dist)

            if min_dist < 1.0:  # Good enough solution
                break

        except Exception as e:
            # print(f"[DEBUG] Method {method} failed: {e}")
            continue

    if best_result is None:
        # Fallback: use initial guess
        # print("[DEBUG] All optimizations failed, using initial guess")
        min_dist, t_impact, hit_point = closest_approach_for_angles(x0)
        best_result = (x0[0], x0[1], t_impact, hit_point, min_dist)

    elev, azim, t_impact, hit_point, min_dist = best_result
    # print(
    #     f"[DEBUG] Final: elevation={np.degrees(elev):.3f}°, azimuth={np.degrees(azim):.3f}°, "
    #     f"t_impact={t_impact:.3f}s, min_dist={min_dist:.3f}m"
    # )

    return elev, azim, t_impact, hit_point, min_dist
