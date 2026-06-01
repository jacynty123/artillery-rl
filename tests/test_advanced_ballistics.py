"""
Unit tests for advanced ballistics implementation.

Tests the 6-DOF ballistics model, ammunition parameters, atmospheric conditions,
and integration with other system components.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_ballistics import (
    AmmoParameters, 
    AtmosphericConditions,
    AdvancedProjectileMotion
)


class TestAmmoParameters:
    """Test cases for ammunition parameter definitions."""
    
    def test_tpt_ammo_creation(self):
        """Test TPT ammunition parameters."""
        ammo = AmmoParameters.create_tpt_ammo()
        
        assert ammo.mass == 0.55
        assert ammo.caliber == 0.035
        assert ammo.v0 == 1180.0
        assert len(ammo.coeffC0) == 8
        assert len(ammo.coeffCL) == 8
        assert len(ammo.coeffCspin) == 8
        
    def test_fapdst_ammo_creation(self):
        """Test FAPDS-T ammunition parameters."""
        ammo = AmmoParameters.create_fapdst_ammo()
        
        assert ammo.mass == 0.297
        assert ammo.v0 == 1440.0
        assert len(ammo.coeffC0) == 8
        
    def test_ahead_ammo_creation(self):
        """Test AHEAD ammunition parameters."""
        ammo = AmmoParameters.create_ahead_ammo()
        
        assert ammo.mass == 0.745
        assert ammo.v0 == 1050.0
        assert len(ammo.coeffC0) == 8


class TestAtmosphericConditions:
    """Test cases for atmospheric modeling."""
    
    def test_sea_level_conditions(self):
        """Test standard sea level atmospheric conditions."""
        atmosphere = AtmosphericConditions()
        density, sound_speed, temperature = atmosphere.get_conditions(0.0)
        
        # Standard sea level values 
        assert density > 1.0         # kg/m³ should be reasonable
        assert sound_speed > 300     # m/s should be reasonable  
        assert temperature > 200     # K should be reasonable
        
    def test_altitude_effects(self):
        """Test that atmospheric properties change with altitude."""
        atmosphere = AtmosphericConditions()
        
        sea_level = atmosphere.get_conditions(0.0)
        high_alt = atmosphere.get_conditions(5000.0)  # 5km
        
        # Density should decrease with altitude
        assert high_alt[0] < sea_level[0]  # density
        
    def test_high_altitude(self):
        """Test atmospheric conditions at high altitude."""
        atmosphere = AtmosphericConditions()
        density, sound_speed, temperature = atmosphere.get_conditions(10000.0)
        
        # Should be lower than sea level values
        assert density > 0        # Should be positive
        assert sound_speed > 200  # Should be reasonable
        assert temperature > 150  # Should be reasonable


class TestAdvancedProjectileMotion:
    """Test cases for advanced ballistic calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ammo = AmmoParameters.create_tpt_ammo()
        self.atmosphere = AtmosphericConditions()
        self.ballistics = AdvancedProjectileMotion(self.ammo, self.atmosphere)
        
    def test_initialization(self):
        """Test proper initialization of advanced ballistics."""
        assert hasattr(self.ballistics, 'ammo')
        assert hasattr(self.ballistics, 'atmosphere')
        assert self.ballistics.ammo is not None
        assert self.ballistics.atmosphere is not None
        
    def test_air_frame_calculation(self):
        """Test airframe parameter calculations."""
        result = self.ballistics._air_frame(1000.0, 100.0)  # velocity, altitude
        
        # Should return valid airframe parameters
        assert isinstance(result, (list, tuple, np.ndarray))
        assert len(result) > 0
        
    def test_aerodynamic_coefficients(self):
        """Test calculation of aerodynamic coefficients."""
        velocity = 800.0
        altitude = 1000.0
        
        # Test coefficient calculations don't raise errors
        try:
            airframe = self.ballistics._air_frame(velocity, altitude)
            # Should complete without error
            assert True
        except Exception as e:
            pytest.fail(f"Aerodynamic coefficient calculation failed: {e}")
            
    def test_motion_equations(self):
        """Test the motion equations implementation."""
        # Initial state vector [altitude, x, y, velocity, elevation, azimuth, spin, range]
        state = np.array([1000.0, 0.0, 0.0, 800.0, 0.1, 0.0, 100.0, 0.0])
        time = 0.0
        
        derivatives = self.ballistics._motion_equations(time, state)
        
        # Should return 8 derivatives
        assert len(derivatives) == 8
        assert all(np.isfinite(derivatives))
        
    def test_basic_trajectory_calculation(self):
        """Test basic trajectory calculation functionality."""
        times, trajectory = self.ballistics.calculate_trajectory(
            initial_position=(0.0, 0.0, 100.0),
            initial_velocity=800.0,
            elevation_angle=0.1,
            azimuth_angle=0.0,
            max_time=5.0,
            time_step=0.1
        )
        
        # Should produce reasonable results
        assert len(times) > 10
        assert len(trajectory) == len(times)
        assert trajectory.shape[1] == 8  # 8 state variables
        
        # Projectile should move forward
        assert trajectory[-1, 1] > 0  # x-position increases
        
    def test_trajectory_with_different_ammo(self):
        """Test trajectory calculation with different ammunition types."""
        # Test FAPDS-T ammo
        fapdst_ammo = AmmoParameters.create_fapdst_ammo()
        fapdst_ballistics = AdvancedProjectileMotion(fapdst_ammo, self.atmosphere)
        
        times, trajectory = fapdst_ballistics.calculate_trajectory(
            initial_position=(0.0, 0.0, 100.0),
            initial_velocity=1440.0,  # Higher velocity for FAPDS-T
            elevation_angle=0.05,
            azimuth_angle=0.0,
            max_time=3.0,
            time_step=0.1
        )
        
        assert len(times) > 10
        assert trajectory[-1, 1] > 0  # Forward progress
        
    def test_ground_impact_detection(self):
        """Test that trajectory stops when projectile hits ground."""
        times, trajectory = self.ballistics.calculate_trajectory(
            initial_position=(0.0, 0.0, 100.0),  # Start at 100m altitude
            initial_velocity=500.0,
            elevation_angle=-0.1,  # Downward angle
            azimuth_angle=0.0,
            max_time=20.0,
            time_step=0.01
        )
        
        # Should stop before max_time when hitting ground
        assert len(times) < 20.0 / 0.01
        
        # Final altitude should be near ground level
        assert trajectory[-1, 0] <= 1.0  # altitude ≤ 1m
        
    def test_mach_number_effects(self):
        """Test that Mach number calculation affects trajectory."""
        # Test subsonic and supersonic cases
        subsonic_times, subsonic_traj = self.ballistics.calculate_trajectory(
            initial_position=(0.0, 0.0, 100.0),
            initial_velocity=300.0,  # Subsonic
            elevation_angle=0.1,
            azimuth_angle=0.0,
            max_time=10.0,
            time_step=0.1
        )
        
        supersonic_times, supersonic_traj = self.ballistics.calculate_trajectory(
            initial_position=(0.0, 0.0, 100.0),
            initial_velocity=1200.0,  # Supersonic
            elevation_angle=0.1,
            azimuth_angle=0.0,
            max_time=10.0,
            time_step=0.1
        )
        
        # Both should produce valid results
        assert len(subsonic_times) > 5
        assert len(supersonic_times) > 5
        
        # Higher velocity should generally achieve greater range
        subsonic_range = subsonic_traj[-1, 1]
        supersonic_range = supersonic_traj[-1, 1]
        assert supersonic_range > subsonic_range


class TestIntegrationAdvancedBallistics:
    """Integration tests for advanced ballistics behavior."""
    
    def test_aerodynamic_drag_effects(self):
        """Test that aerodynamic effects significantly impact trajectory."""
        ammo = AmmoParameters.create_tpt_ammo()
        atmosphere = AtmosphericConditions()
        advanced = AdvancedProjectileMotion(ammo, atmosphere)
        
        # Calculate trajectory with realistic parameters
        times, trajectory = advanced.calculate_trajectory(
            initial_position=(0.0, 0.0, 0.0),
            initial_velocity=800.0,
            elevation_angle=np.radians(45),  # 45 degree launch
            azimuth_angle=0.0,
            max_time=15.0,
            time_step=0.1
        )
        
        # Advanced ballistics should show realistic ranges
        max_range = np.max(trajectory[:, 1])  # x-position
        
        # For 45° launch, simple ballistics would give v²/g ≈ 65km
        # Advanced should be much less due to drag
        expected_simple_range = 800**2 / 9.81  # ~65,000m
        
        # Advanced should show significant range reduction due to drag
        assert max_range < expected_simple_range * 0.3  # Less than 30% of simple range
        assert max_range > 1000  # But still reasonable (>1km)