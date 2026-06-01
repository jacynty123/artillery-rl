import numpy as np
from src.hit_probability import HitProbabilityCalculator, ErrorPropagation
from src.advanced_ballistics import AmmoParameters, AdvancedProjectileMotion


def test_covariance_hit_probability_basic():
    ammo = AmmoParameters.create_tpt_ammo()
    motion = AdvancedProjectileMotion(ammo)
    calc = HitProbabilityCalculator(projectile_velocity=ammo.v0)

    target_position = (1500.0, 100.0, 0.0)
    target_velocity = (40.0, 5.0, 0.0)

    # Projectile param covariance (v0, elev, azim)
    projectile_cov = np.diag([5.0**2, (0.01)**2, (0.01)**2])

    # Target param covariance (x0,y0,z0,vx,vy,vz)
    target_cov = np.diag([2.0**2, 2.0**2, 1.0**2, 1.0**2, 0.5**2, 0.2**2])

    p_hit = calc.calculate_hit_probability_analytical(
        target_position,
        target_velocity,
        projectile_cov,
        target_cov,
        elevation_angle=0.12,
        azimuth_angle=0.03,
        use_diagonal_approx=True,
        mc_refine_samples=0
    )

    assert 0.0 <= p_hit <= 1.0


def test_covariance_hit_probability_mc_refine():
    ammo = AmmoParameters.create_fapdst_ammo()
    calc = HitProbabilityCalculator(projectile_velocity=ammo.v0, ammo_type="fapdst")

    target_position = (1000.0, 50.0, 0.0)
    target_velocity = (20.0, 3.0, 0.0)

    projectile_cov = np.diag([10.0**2, (0.02)**2, (0.02)**2])
    target_cov = np.diag([3.0**2, 3.0**2, 1.0**2, 2.0**2, 1.0**2, 0.3**2])

    p_hit = calc.calculate_hit_probability_analytical(
        target_position,
        target_velocity,
        projectile_cov,
        target_cov,
        elevation_angle=0.1,
        azimuth_angle=0.02,
        use_diagonal_approx=False,
        mc_refine_samples=5000
    )

    assert 0.0 <= p_hit <= 1.0
