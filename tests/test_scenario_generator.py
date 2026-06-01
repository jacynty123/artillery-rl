"""
Tests for Scenario Generator

Tests the scenario generation functionality for RL training.
"""

import pytest
import numpy as np
from rl_training.curriculum.scenario_generator import ScenarioGenerator, ScenarioParameters


class TestScenarioGenerator:
    """Test the scenario generator functionality."""

    def test_initialization(self):
        """Test that the generator initializes correctly."""
        generator = ScenarioGenerator(seed=42)
        assert generator.rng is not None
        assert generator.range_bounds == (800.0, 4200.0)

    def test_generate_scenario(self):
        """Test generating a single scenario."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario()

        assert isinstance(scenario, ScenarioParameters)
        assert 800.0 <= scenario.range_m <= 4200.0
        assert 8.0 <= scenario.target_length <= 12.0
        assert 6.0 <= scenario.target_width <= 10.0
        assert 2.5 <= scenario.target_height <= 5.5
        assert -50.0 <= scenario.target_vx <= 50.0
        assert -50.0 <= scenario.target_vy <= 50.0
        assert -3.0 <= scenario.target_vz <= 3.0
        assert 10.0 <= scenario.tracking_duration <= 15.0
        assert len(scenario.measurement_noise_std) == 3
        assert len(scenario.process_noise_std) == 3

    def test_generate_batch(self):
        """Test generating a batch of scenarios."""
        generator = ScenarioGenerator(seed=42)
        batch = generator.generate_batch(5)

        assert len(batch) == 5
        assert all(isinstance(s, ScenarioParameters) for s in batch)

        # Check that scenarios are diverse (not all identical)
        ranges = [s.range_m for s in batch]
        assert len(set(ranges)) > 1  # At least some different ranges

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        gen1 = ScenarioGenerator(seed=42)
        gen2 = ScenarioGenerator(seed=42)

        s1 = gen1.generate_scenario()
        s2 = gen2.generate_scenario()

        assert s1.range_m == s2.range_m
        assert s1.target_length == s2.target_length
        assert np.array_equal(s1.measurement_noise_std, s2.measurement_noise_std)

    def test_generate_scenario_with_difficulty_easy(self):
        """Test generating easy difficulty scenarios."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario_with_difficulty("easy")

        assert isinstance(scenario, ScenarioParameters)
        assert 200.0 <= scenario.range_m <= 500.0  # Close range
        assert 8.0 <= scenario.target_length <= 15.0  # Large target
        assert 10.0 <= scenario.tracking_duration <= 20.0  # Long tracking
        assert np.all(scenario.measurement_noise_std <= 1.5)  # Good sensors

    def test_generate_scenario_with_difficulty_hard(self):
        """Test generating hard difficulty scenarios."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario_with_difficulty("hard")

        assert isinstance(scenario, ScenarioParameters)
        assert 1500.0 <= scenario.range_m <= 2000.0  # Long range
        assert 2.0 <= scenario.target_length <= 6.0  # Small target
        assert 5.0 <= scenario.tracking_duration <= 10.0  # Short tracking
        assert np.all(scenario.measurement_noise_std >= 3.0)  # Poor sensors

    def test_generate_scenario_with_difficulty_medium(self):
        """Test generating medium difficulty scenarios."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario_with_difficulty("medium")

        assert isinstance(scenario, ScenarioParameters)
        assert 500.0 <= scenario.range_m <= 1500.0  # Medium range
        assert 4.0 <= scenario.target_length <= 12.0  # Medium target
        assert 7.0 <= scenario.tracking_duration <= 15.0  # Medium tracking

    def test_to_dict_conversion(self):
        """Test conversion to dictionary format."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario()
        scenario_dict = scenario.to_dict()

        assert isinstance(scenario_dict, dict)
        assert "range_m" in scenario_dict
        assert "target_length" in scenario_dict
        assert "measurement_noise_std" in scenario_dict
        assert isinstance(
            scenario_dict["measurement_noise_std"], list
        )  # Now returns list for JSON compatibility

    def test_parameter_bounds(self):
        """Test that all parameters stay within defined bounds."""
        generator = ScenarioGenerator(seed=42)

        # Generate many scenarios to test bounds
        scenarios = generator.generate_batch(100)

        for scenario in scenarios:
            assert (
                generator.range_bounds[0]
                <= scenario.range_m
                <= generator.range_bounds[1]
            )
            assert (
                generator.target_size_bounds[0]
                <= scenario.target_length
                <= generator.target_size_bounds[1]
            )
            # width and height are generated with separate multipliers,
            # not directly from target_size_bounds
            assert 6.0 <= scenario.target_width <= 10.0
            assert 2.5 <= scenario.target_height <= 5.5
            assert (
                generator.target_velocity_bounds[0]
                <= scenario.target_vx
                <= generator.target_velocity_bounds[1]
            )
            assert (
                generator.target_velocity_bounds[0]
                <= scenario.target_vy
                <= generator.target_velocity_bounds[1]
            )
            assert (
                generator.target_velocity_bounds[0]
                <= scenario.target_vz
                <= generator.target_velocity_bounds[1]
            )
            assert (
                generator.tracking_duration_bounds[0]
                <= scenario.tracking_duration
                <= generator.tracking_duration_bounds[1]
            )
            assert np.all(
                scenario.measurement_noise_std >= generator.measurement_noise_bounds[0]
            )
            assert np.all(
                scenario.measurement_noise_std <= generator.measurement_noise_bounds[1]
            )
            assert np.all(
                scenario.process_noise_std >= generator.process_noise_bounds[0]
            )
            assert np.all(
                scenario.process_noise_std <= generator.process_noise_bounds[1]
            )

    def test_invalid_difficulty(self):
        """Test that invalid difficulty raises ValueError."""
        generator = ScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="Difficulty must be one of"):
            generator.generate_scenario_with_difficulty("invalid")

    def test_dict_conversion_fidelity(self):
        """Test that dictionary conversion preserves all data."""
        generator = ScenarioGenerator(seed=42)
        scenario = generator.generate_scenario()
        scenario_dict = scenario.to_dict()

        # Verify all fields are preserved
        for field in scenario.__dataclass_fields__:
            original = getattr(scenario, field)
            converted = scenario_dict[field]
            if isinstance(original, np.ndarray):
                assert np.array_equal(original, np.array(converted))
            else:
                assert original == converted

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        generator = ScenarioGenerator(seed=42)
        original = generator.generate_scenario()

        # Serialize to JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)

        # Deserialize from JSON
        restored = ScenarioParameters.from_json(json_str)

        # Verify all fields match
        assert original.range_m == restored.range_m
        assert original.target_length == restored.target_length
        assert original.target_width == restored.target_width
        assert original.target_height == restored.target_height
        assert original.target_vx == restored.target_vx
        assert original.target_vy == restored.target_vy
        assert original.target_vz == restored.target_vz
        assert original.tracking_duration == restored.tracking_duration
        assert np.array_equal(
            original.measurement_noise_std, restored.measurement_noise_std
        )
        assert np.array_equal(original.process_noise_std, restored.process_noise_std)
