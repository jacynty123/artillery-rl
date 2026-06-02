#!/usr/bin/env python3
"""
Integration tests for dynamic covariance integration.

Tests the complete pipeline from trajectory simulation through Kalman filtering
to hit probability calculation with dynamic covariance estimates.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from hit_probability_scenarios import create_scenario, analyze_scenario, compute_dynamic_target_covariance
from src.kalman_filter import TargetKalmanFilter, MeasurementType


class TestDynamicCovarianceIntegration:
    """Test dynamic covariance integration end-to-end."""

    def test_dynamic_covariance_computation(self):
        """Test that dynamic covariance computation produces reasonable results."""
        # Create a test scenario
        scenario = create_scenario(
            range_m=1000.0,
            target_length=12.0,
            target_width=2.3,
            target_height=2.3,
            target_vx=150.0,
            target_vy=15.0,
            target_vz=0.0,
            tracking_duration=10.0
        )

        impact_time = 5.0  # 5 second flight time

        # Compute dynamic covariance
        dynamic_cov = compute_dynamic_target_covariance(scenario, impact_time)

        # Check basic properties
        assert dynamic_cov.shape == (6, 6), "Covariance should be 6x6"
        assert np.allclose(dynamic_cov, dynamic_cov.T), "Covariance should be symmetric"

        # Check positive definiteness
        eigenvals = np.linalg.eigvals(dynamic_cov)
        assert np.all(eigenvals > 0), "Covariance should be positive definite"

        # Position uncertainties should be reasonable (< 100m)
        pos_std = np.sqrt(np.diag(dynamic_cov)[:3])
        assert np.all(pos_std < 100.0), f"Position uncertainties too high: {pos_std}"

        # Velocity uncertainties should be reasonable (< 50 m/s)
        vel_std = np.sqrt(np.diag(dynamic_cov)[3:])
        assert np.all(vel_std < 50.0), f"Velocity uncertainties too high: {vel_std}"


    def test_tracking_duration_effect(self):
        """Test that longer tracking duration reduces uncertainty."""
        scenario = create_scenario(
            range_m=1000.0,
            target_length=12.0,
            target_width=2.3,
            target_height=2.3,
            target_vx=150.0,
            target_vy=15.0,
            target_vz=0.0,
            tracking_duration=10.0
        )
        impact_time = 5.0

        # Test with different tracking durations
        durations = [2.0, 5.0, 10.0]
        covariances = []

        for duration in durations:
            scenario["tracking_duration"] = duration
            cov = compute_dynamic_target_covariance(scenario, impact_time)
            covariances.append(cov)

        # Longer tracking should generally reduce uncertainty
        trace_2s = np.trace(covariances[0])
        trace_5s = np.trace(covariances[1])
        trace_10s = np.trace(covariances[2])

        # 10s tracking should have lower or equal uncertainty compared to 2s
        assert trace_10s <= trace_2s * 1.1, "Longer tracking should reduce uncertainty"

    def test_measurement_noise_effect(self):
        """Test that higher measurement noise increases uncertainty."""
        scenario = create_scenario(
            range_m=1000.0,
            target_length=12.0,
            target_width=2.3,
            target_height=2.3,
            target_vx=150.0,
            target_vy=15.0,
            target_vz=0.0,
            tracking_duration=10.0
        )
        impact_time = 5.0

        # Test with different measurement noise levels
        noise_levels = [
            np.array([2.0, 2.0, 5.0]),   # Low noise
            np.array([5.0, 5.0, 10.0]),  # Medium noise
            np.array([10.0, 10.0, 20.0]) # High noise
        ]

        covariances = []
        for noise in noise_levels:
            scenario["measurement_noise_std"] = noise
            cov = compute_dynamic_target_covariance(scenario, impact_time)
            covariances.append(cov)

        # Higher measurement noise should increase uncertainty
        trace_low = np.trace(covariances[0])
        trace_med = np.trace(covariances[1])
        trace_high = np.trace(covariances[2])

        assert trace_med > trace_low, "Higher measurement noise should increase uncertainty"
        assert trace_high > trace_med, "Higher measurement noise should increase uncertainty"

    def test_kalman_filter_convergence(self):
        """Test that Kalman filter converges and produces stable estimates."""
        scenario = create_scenario(
            range_m=1000.0,
            target_length=12.0,
            target_width=2.3,
            target_height=2.3,
            target_vx=150.0,
            target_vy=15.0,
            target_vz=0.0,
            tracking_duration=10.0
        )
        impact_time = 5.0

        # Run multiple times to check consistency
        covariances = []
        for _ in range(3):
            cov = compute_dynamic_target_covariance(scenario, impact_time)
            covariances.append(cov)

        # Results should be reasonably consistent (within factor of 2)
        traces = [np.trace(cov) for cov in covariances]
        mean_trace = np.mean(traces)
        std_trace = np.std(traces)

        # Coefficient of variation should be reasonable
        cv = std_trace / mean_trace
        assert cv < 0.5, f"Filter results too variable: CV={cv:.3f}"

    def test_create_scenario_with_dynamic_covariance(self):
        """Test that create_scenario function properly sets up dynamic covariance parameters."""
        from hit_probability_scenarios import create_scenario

        # Test creating a scenario (dynamic covariance is always enabled)
        scenario = create_scenario(
            range_m=1000.0,
            target_length=12.0,
            target_width=2.3,
            target_height=2.3,
            target_vx=150.0,
            target_vy=15.0,
            target_vz=0.0,
            tracking_duration=10.0
        )

        assert "use_dynamic_covariance" in scenario, "Scenario missing dynamic covariance flag"
        assert scenario["use_dynamic_covariance"] == True, "Scenario should always use dynamic covariance"
        assert "tracking_duration" in scenario, "Scenario missing tracking duration"
        assert "measurement_noise_std" in scenario, "Scenario missing measurement noise"
        assert "process_noise_std" in scenario, "Scenario missing process noise"
        assert scenario["tracking_duration"] == 10.0, "Tracking duration not set correctly"

    def test_covariance_physical_reasonableness(self):
        """Test that computed covariances are physically reasonable."""
        # Create multiple test scenarios with different parameters
        scenarios = [
            create_scenario(
                range_m=500.0,
                target_length=8.0,
                target_width=2.0,
                target_height=2.0,
                target_vx=100.0,
                target_vy=10.0,
                target_vz=0.0,
                tracking_duration=8.0
            ),
            create_scenario(
                range_m=1500.0,
                target_length=15.0,
                target_width=3.0,
                target_height=3.0,
                target_vx=200.0,
                target_vy=20.0,
                target_vz=5.0,
                tracking_duration=12.0
            )
        ]
        impact_time = 5.0

        for scenario in scenarios:
            cov = compute_dynamic_target_covariance(scenario, impact_time)

            # Position standard deviations should be positive and reasonable
            pos_std = np.sqrt(np.diag(cov)[:3])
            assert np.all(pos_std > 0), "Position uncertainties must be positive"
            assert np.all(pos_std < 1000.0), "Position uncertainties unrealistically high"

            # Velocity standard deviations should be positive and reasonable
            vel_std = np.sqrt(np.diag(cov)[3:])
            assert np.all(vel_std > 0), "Velocity uncertainties must be positive"
            assert np.all(vel_std < 200.0), "Velocity uncertainties unrealistically high"

            # Covariance should have reasonable condition number
            cond_num = np.linalg.cond(cov)
            assert cond_num < 1e10, f"Covariance ill-conditioned: cond={cond_num:.1e}"