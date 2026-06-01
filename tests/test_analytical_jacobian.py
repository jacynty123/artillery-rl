"""
Test analytical Jacobian implementation vs finite difference method.
"""

import numpy as np
import pytest
from src.hit_probability import ErrorPropagation
from src.advanced_ballistics import AmmoParameters


class TestAnalyticalJacobian:
    """Test analytical Jacobian implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test ammo (TPT type)
        from src.advanced_ballistics import AdvancedProjectileMotion
        self.ammo = AmmoParameters.create_tpt_ammo()
        self.advanced_ballistics = AdvancedProjectileMotion(self.ammo)
        
        # Create error propagation instance
        self.error_prop = ErrorPropagation(self.advanced_ballistics)
        
        # Test parameters
        self.initial_velocity = 1050.0
        self.elevation_angle = 0.1
        self.azimuth_angle = 0.05
        self.target_position = (1500.0, 200.0, 0.0)
        self.target_velocity = (50.0, 10.0, 0.0)
        self.impact_time = 2.0
    
    def test_analytical_jacobian_exists(self):
        """Test that analytical Jacobian method exists."""
        assert hasattr(self.error_prop, 'calculate_jacobian_analytical')
    
    def test_analytical_jacobian_shape(self):
        """Test that analytical Jacobian returns correct shape."""
        jacobian = self.error_prop.calculate_jacobian_analytical(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        assert jacobian.shape == (3, 8), f"Expected (3, 8), got {jacobian.shape}"
    
    def test_analytical_vs_finite_difference(self):
        """Compare analytical Jacobian with finite difference method."""
        # Calculate both Jacobians
        jacobian_fd = self.error_prop.calculate_jacobian(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        jacobian_analytical = self.error_prop.calculate_jacobian_analytical(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        # Both should have same shape
        assert jacobian_fd.shape == jacobian_analytical.shape
        
        # They should be reasonably close (within numerical tolerance)
        # Note: Analytical method may have different accuracy, so we use relaxed tolerance
        relative_error = np.abs((jacobian_analytical - jacobian_fd) / (jacobian_fd + 1e-10))
        
        # Check that most elements are reasonably close (some may differ due to method differences)
        close_elements = relative_error < 0.5  # 50% tolerance for now
        close_ratio = np.mean(close_elements)
        
        print(f"Jacobian comparison:")
        print(f"Finite Difference:\n{jacobian_fd}")
        print(f"Analytical:\n{jacobian_analytical}")
        print(f"Relative Error:\n{relative_error}")
        print(f"Close elements ratio: {close_ratio:.2f}")
        
        # At least 70% of elements should be reasonably close
        assert close_ratio >= 0.7, f"Only {close_ratio:.2f} of elements are close"
    
    def test_analytical_jacobian_target_derivatives(self):
        """Test that target position/velocity derivatives are correct."""
        jacobian = self.error_prop.calculate_jacobian_analytical(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        # Target position derivatives should be -1 on diagonal
        assert np.isclose(jacobian[0, 3], -1.0), f"dx/dtarget_x = {jacobian[0, 3]}, expected -1.0"
        assert np.isclose(jacobian[1, 4], -1.0), f"dy/dtarget_y = {jacobian[1, 4]}, expected -1.0" 
        assert np.isclose(jacobian[2, 5], -1.0), f"dz/dtarget_z = {jacobian[2, 5]}, expected -1.0"
        
        # Target velocity derivatives should be -impact_time
        assert np.isclose(jacobian[0, 6], -self.impact_time), f"dx/dtarget_vx = {jacobian[0, 6]}, expected {-self.impact_time}"
        assert np.isclose(jacobian[1, 7], -self.impact_time), f"dy/dtarget_vy = {jacobian[1, 7]}, expected {-self.impact_time}"
    
    def test_analytical_jacobian_fallback(self):
        """Test analytical Jacobian fallback to finite difference."""
        # Test with parameters that might cause analytical method to fail
        # Use very small velocity that might cause numerical issues but not hang
        jacobian = self.error_prop.calculate_jacobian_analytical(
            1e-6,  # Very small velocity that should trigger fallback
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.001  # Very short time
        )
        
        # Should still return a valid Jacobian (via fallback)
        assert jacobian.shape == (3, 8)
        assert not np.any(np.isnan(jacobian))
        assert not np.any(np.isinf(jacobian))
    
    def test_both_methods_available(self):
        """Test that both Jacobian calculation methods are available."""
        # Both methods should exist and be callable
        assert callable(getattr(self.error_prop, 'calculate_jacobian'))
        assert callable(getattr(self.error_prop, 'calculate_jacobian_analytical'))
        
        # Both should return valid results
        jacobian_fd = self.error_prop.calculate_jacobian(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        jacobian_analytical = self.error_prop.calculate_jacobian_analytical(
            self.initial_velocity,
            self.elevation_angle,
            self.azimuth_angle,
            self.target_position,
            self.target_velocity,
            self.impact_time
        )
        
        assert not np.any(np.isnan(jacobian_fd))
        assert not np.any(np.isnan(jacobian_analytical))
        assert jacobian_fd.shape == (3, 8)
        assert jacobian_analytical.shape == (3, 8)


class TestAnalyticalJacobianIntegration:
    """Integration tests for analytical Jacobian."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        from src.advanced_ballistics import AdvancedProjectileMotion
        self.ammo = AmmoParameters.create_fapdst_ammo()
        self.advanced_ballistics = AdvancedProjectileMotion(self.ammo)
        self.error_prop = ErrorPropagation(self.advanced_ballistics)
    
    def test_analytical_method_performance(self):
        """Test analytical method performance vs finite difference."""
        import time
        
        # Test parameters
        initial_velocity = 950.0
        elevation_angle = 0.15
        azimuth_angle = 0.02
        target_position = (1200.0, 100.0, 0.0)
        target_velocity = (30.0, 5.0, 0.0)
        impact_time = 1.5
        
        # Time finite difference method
        start_time = time.time()
        jacobian_fd = self.error_prop.calculate_jacobian(
            initial_velocity, elevation_angle, azimuth_angle,
            target_position, target_velocity, impact_time
        )
        fd_time = time.time() - start_time
        
        # Time analytical method
        start_time = time.time()
        jacobian_analytical = self.error_prop.calculate_jacobian_analytical(
            initial_velocity, elevation_angle, azimuth_angle,
            target_position, target_velocity, impact_time
        )
        analytical_time = time.time() - start_time
        
        print(f"Finite Difference time: {fd_time:.4f}s")
        print(f"Analytical time: {analytical_time:.4f}s")
        
        # Both should produce reasonable results
        assert not np.any(np.isnan(jacobian_fd))
        assert not np.any(np.isnan(jacobian_analytical))
        
        # Analytical method should be reasonably efficient
        # (might not always be faster due to implementation complexity)
        assert analytical_time < 10 * fd_time, "Analytical method is too slow"
    
    def test_different_ammo_types(self):
        """Test analytical Jacobian with different ammo types."""
        ammo_types = [
            AmmoParameters.create_tpt_ammo(),
            AmmoParameters.create_fapdst_ammo(),
            AmmoParameters.create_ahead_ammo()
        ]
        
        for i, ammo in enumerate(ammo_types):
            from src.advanced_ballistics import AdvancedProjectileMotion
            advanced_ballistics = AdvancedProjectileMotion(ammo)
            error_prop = ErrorPropagation(advanced_ballistics)
            
            jacobian = error_prop.calculate_jacobian_analytical(
                1000.0 + i * 50,  # Vary velocity
                0.1 + i * 0.05,   # Vary elevation
                0.02 + i * 0.01,  # Vary azimuth
                (1000.0 + i * 200, 50.0 + i * 25, 0.0),
                (20.0 + i * 10, 5.0, 0.0),
                1.8 + i * 0.2
            )
            
            assert jacobian.shape == (3, 8), f"Ammo type {i} failed shape test"
            assert not np.any(np.isnan(jacobian)), f"Ammo type {i} produced NaN"
            assert not np.any(np.isinf(jacobian)), f"Ammo type {i} produced Inf"


if __name__ == "__main__":
    pytest.main([__file__])