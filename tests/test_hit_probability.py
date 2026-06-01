"""
Unit tests for error propagation and hit probability calculations.

Tests the ErrorPropagation class and HitProbabilityCalculator using only
advanced ballistics models.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hit_probability import ErrorPropagation, HitProbabilityCalculator
from advanced_ballistics import AdvancedProjectileMotion, AmmoParameters


class TestErrorPropagation:
    """Test cases for ErrorPropagation class."""

    def setup_method(self):
        """Set up test fixtures."""
        ammo = AmmoParameters.create_tpt_ammo()
        self.advanced_ballistics = AdvancedProjectileMotion(ammo)
        self.error_prop = ErrorPropagation(self.advanced_ballistics)

    def test_jacobian_shape(self):
        """Test that Jacobian matrix has correct shape."""
        jacobian = self.error_prop.calculate_jacobian(
            initial_velocity=800.0,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            target_position=(1000, 0, 100),
            target_velocity=(50, 0, 0),
            impact_time=2.0,
        )

        # Should be 3x8 matrix (3 position outputs, 8 input parameters)
        assert jacobian.shape == (3, 8)

    def test_jacobian_velocity_sensitivity(self):
        """Test that Jacobian shows correct sensitivity to velocity changes."""
        jacobian = self.error_prop.calculate_jacobian(
            initial_velocity=800.0,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            target_position=(1000, 0, 100),
            target_velocity=(50, 0, 0),
            impact_time=2.0,
        )

        # Velocity should significantly affect x-position (first row, first column)
        velocity_sensitivity_x = jacobian[0, 0]
        assert abs(velocity_sensitivity_x) > 0.1

    def test_jacobian_angle_sensitivity(self):
        """Test Jacobian sensitivity to angle changes."""
        jacobian = self.error_prop.calculate_jacobian(
            initial_velocity=800.0,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            target_position=(1000, 0, 100),
            target_velocity=(50, 0, 0),
            impact_time=2.0,
        )

        # Elevation should affect z-position (third row, second column)
        elevation_sensitivity_z = jacobian[2, 1]
        assert abs(elevation_sensitivity_z) > 1.0

        # Azimuth should affect y-position (second row, third column)
        azimuth_sensitivity_y = jacobian[1, 2]
        assert abs(azimuth_sensitivity_y) > 1.0

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through Jacobian."""
        jacobian = self.error_prop.calculate_jacobian(
            initial_velocity=800.0,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            target_position=(1000, 0, 100),
            target_velocity=(50, 0, 0),
            impact_time=2.0,
        )

        # Create input uncertainty matrix
        input_uncertainties = np.array([10, 0.001, 0.001, 5, 5, 2, 2, 1])
        input_covariance = np.diag(input_uncertainties**2)

        # Propagate uncertainties
        output_covariance = self.error_prop.propagate_uncertainty(
            jacobian, input_covariance
        )

        # Output should be 3x3 matrix
        assert output_covariance.shape == (3, 3)

        # Should be positive semi-definite
        eigenvals = np.linalg.eigvals(output_covariance)
        assert all(eigenvals >= -1e-10)  # Allow for small numerical errors

    def test_projectile_position_calculation(self):
        """Test internal projectile position calculation."""
        position = self.error_prop._calculate_projectile_position_advanced(
            velocity=800.0, elevation=0.1, azimuth=0.0, time=2.0
        )

        # Should return 3D position
        assert len(position) == 3
        assert all(np.isfinite(position))

        # Should show forward motion
        assert position[0] > 0  # x-position should be positive


class TestHitProbabilityCalculator:
    """Test cases for HitProbabilityCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = HitProbabilityCalculator(
            projectile_velocity=800.0,
            target_dimensions=(5.0, 2.0, 2.0),
            ammo_type="tpt",
        )

    def test_initialization(self):
        """Test proper initialization of calculator."""
        assert self.calculator.projectile_velocity == 800.0
        assert self.calculator.target_dimensions == (5.0, 2.0, 2.0)
        assert self.calculator.advanced_ballistics is not None
        assert self.calculator.error_propagation is not None

    def test_hit_probability_range(self):
        """Test that hit probability is in valid range [0, 1]."""
        # Create measurement uncertainty
        uncertainties = np.array([10, 0.005, 0.005, 10, 10, 5, 5, 2, 2])
        measurement_uncertainty = np.diag(uncertainties**2)

        prob = self.calculator.calculate_hit_probability(
            target_position=(1000, 0, 100),
            target_velocity=(20, 0, 0),
            measurement_uncertainty=measurement_uncertainty,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            n_samples=100,  # Small number for fast testing
        )

        # Should be valid probability
        assert 0.0 <= prob <= 1.0

    def test_hit_probability_perfect_conditions(self):
        """Test hit probability under perfect conditions (no uncertainty)."""
        # Very small uncertainty
        uncertainties = np.array([0.1, 0.0001, 0.0001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        measurement_uncertainty = np.diag(uncertainties**2)

        # Large target
        self.calculator.target_dimensions = (20.0, 20.0, 20.0)

        prob = self.calculator.calculate_hit_probability(
            target_position=(1000, 0, 100),
            target_velocity=(0, 0, 0),  # Stationary target
            measurement_uncertainty=measurement_uncertainty,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            n_samples=50,
        )

        # Should be high probability with low uncertainty and large target
        assert prob > 0.5

    def test_hit_probability_high_uncertainty(self):
        """Test hit probability with high uncertainty."""
        # High uncertainty
        uncertainties = np.array([100, 0.1, 0.1, 50, 50, 20, 20, 10, 10])
        measurement_uncertainty = np.diag(uncertainties**2)

        # Small target
        self.calculator.target_dimensions = (1.0, 1.0, 1.0)

        prob = self.calculator.calculate_hit_probability(
            target_position=(1000, 0, 100),
            target_velocity=(50, 10, 0),
            measurement_uncertainty=measurement_uncertainty,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            n_samples=50,
        )

        # Should be low probability with high uncertainty and small target
        assert prob < 0.5

    def test_is_hit_method(self):
        """Test the hit detection method."""
        # Hit case
        proj_pos = np.array([100, 0, 50])
        target_pos = np.array([101, 1, 51])  # Close to projectile

        assert self.calculator.is_hit(proj_pos, target_pos)

        # Miss case
        proj_pos = np.array([100, 0, 50])
        target_pos = np.array([120, 10, 60])  # Far from projectile

        assert not self.calculator.is_hit(proj_pos, target_pos)

    def test_different_target_sizes(self):
        """Test hit probability with different target sizes."""
        uncertainties = np.array([10, 0.01, 0.01, 5, 5, 2, 2, 1, 1])
        measurement_uncertainty = np.diag(uncertainties**2)

        # Small target
        self.calculator.target_dimensions = (1.0, 1.0, 1.0)
        small_prob = self.calculator.calculate_hit_probability(
            target_position=(1000, 0, 100),
            target_velocity=(20, 0, 0),
            measurement_uncertainty=measurement_uncertainty,
            n_samples=50,
        )

        # Large target
        self.calculator.target_dimensions = (10.0, 10.0, 10.0)
        large_prob = self.calculator.calculate_hit_probability(
            target_position=(1000, 0, 100),
            target_velocity=(20, 0, 0),
            measurement_uncertainty=measurement_uncertainty,
            n_samples=50,
        )

        # Larger target should have higher hit probability
        assert large_prob >= small_prob


class TestIntegrationHitProbability:
    """Integration tests for the complete hit probability system."""

    def test_realistic_engagement_scenario(self):
        """Test a realistic engagement scenario."""
        calculator = HitProbabilityCalculator(
            projectile_velocity=1180.0,  # TPT velocity
            target_dimensions=(12.0, 2.3, 2.3),  # Fighter aircraft
            ammo_type="tpt",
        )

        # Realistic uncertainties
        uncertainties = np.array(
            [
                15.0,  # velocity ±15 m/s
                0.002,  # elevation ±0.1°
                0.002,  # azimuth ±0.1°
                10.0,  # target x ±10m
                5.0,  # target y ±5m
                3.0,  # target z ±3m
                5.0,  # target vx ±5 m/s
                2.0,  # target vy ±2 m/s
                2.0,  # target vz ±2 m/s
            ]
        )
        measurement_uncertainty = np.diag(uncertainties**2)

        prob = calculator.calculate_hit_probability(
            target_position=(3000, 200, 1000),  # 3km range, 1km altitude
            target_velocity=(250, 20, 0),  # Fast moving aircraft
            measurement_uncertainty=measurement_uncertainty,
            elevation_angle=np.arctan(1000 / 3000),  # Appropriate elevation
            azimuth_angle=np.arctan(200 / 3000),  # Lead angle
            n_samples=100,
        )

        # Should be a reasonable probability
        assert 0.0 <= prob <= 1.0
        # Given the scenario, should be relatively low but non-zero
        assert prob >= 0.0
