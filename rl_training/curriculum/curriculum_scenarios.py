"""
Curriculum Learning Scenarios

Defines a fixed set of scenarios organized into phases for curriculum learning.
Enables caching benefits while covering key variations in range, target size, and velocity.
"""

from typing import Dict, List, Tuple
import numpy as np
from rl_training.curriculum.scenario_generator import ScenarioParameters


class CurriculumScenarios:
    """Manages curriculum learning with phased scenario introduction."""
    
    def __init__(self, phase: int = 1, seed: int = 42):
        """
        Initialize curriculum with specified phase.
        
        Args:
            phase: Learning phase (1, 2, or 3)
            seed: Random seed for scenario generation (not used, scenarios are fixed)
        """
        self.phase = phase
        self.seed = seed
        self.scenarios = self._create_phase_scenarios(phase)
        self.current_idx = 0
        
        # Track which scenarios are "easy" vs "hard" for monitoring
        self.difficulty = self._assign_difficulty()
    
    def _create_phase_scenarios(self, phase: int) -> List[Dict]:
        """Create scenarios for the specified phase."""
        scenarios = []
        
        # Phase 1: Kalman Convergence Basics - learn to wait for filter stabilization
        scenarios.extend([
            # Stationary targets at various ranges (HP improves only with convergence)
            {
                'name': 'Static_Close',
                'range_m': 800.0,
                'target_length': 10.0,
                'target_width': 8.0,
                'target_height': 4.0,
                'velocity': (0.0, 0.0, 0.0),  # Stationary
                'tracking_duration': 10.0,
                'measurement_noise_std': 0.5,
                'description': 'Stationary target at close range - learn Kalman convergence timing'
            },
            {
                'name': 'Static_Medium',
                'range_m': 1500.0,
                'target_length': 10.0,
                'target_width': 8.0,
                'target_height': 4.0,
                'velocity': (0.0, 0.0, 0.0),  # Stationary
                'tracking_duration': 12.0,
                'measurement_noise_std': 0.5,
                'description': 'Stationary target at medium range - longer convergence time'
            },
            # Slow moving targets (minimal range change, focus on convergence)
            {
                'name': 'Slow_Approach',
                'range_m': 1000.0,
                'target_length': 10.0,
                'target_width': 8.0,
                'target_height': 4.0,
                'velocity': (-10.0, 0.0, 0.0),  # Slow approaching
                'tracking_duration': 10.0,
                'measurement_noise_std': 0.5,
                'description': 'Slow approaching target - balance convergence vs range improvement'
            },
            {
                'name': 'Slow_Recede',
                'range_m': 1000.0,
                'target_length': 10.0,
                'target_width': 8.0,
                'target_height': 4.0,
                'velocity': (10.0, 0.0, 0.0),  # Slow receding
                'tracking_duration': 10.0,
                'measurement_noise_std': 0.5,
                'description': 'Slow receding target - HP degrades, fire early'
            },
        ])
        
        # Phase 2: Range Optimization - learn to wait for better engagement geometry
        if phase >= 2:
            scenarios.extend([
                # Approaching targets that enter optimal range
                {
                    'name': 'Fast_Approach_Long',
                    'range_m': 3000.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-50.0, 0.0, 0.0),  # Fast approaching from long range
                    'tracking_duration': 20.0,
                    'measurement_noise_std': 0.2,  # Reduced from 0.5 for better HP estimation
                    'description': 'Fast approaching target from long range - learn range optimization'
                },
                {
                    'name': 'Faster_Approach_Mid',
                    'range_m': 3500.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-65.0, 0.0, 0.0),  # Intermediate fast approaching
                    'tracking_duration': 17.0,
                    'measurement_noise_std': 0.2,
                    'description': 'Intermediate fast approaching target - bridges gap to Very_Fast_Approach'
                },
                {
                    'name': 'Medium_Approach',
                    'range_m': 2000.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-30.0, 0.0, 0.0),  # Medium approaching
                    'tracking_duration': 15.0,
                    'measurement_noise_std': 0.5,
                    'description': 'Medium approaching target - balance range improvement vs time pressure'
                },
                # Targets that pass through optimal range
                {
                    'name': 'Very_Fast_Approach',
                    'range_m': 4000.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-80.0, 0.0, 0.0),  # Very fast approaching
                    'tracking_duration': 15.0,
                    'measurement_noise_std': 0.2,  # Reduced from 0.5 for better HP estimation
                    'process_noise_std': 0.5,
                    'description': 'Very fast approaching target - observe HP peak and decline'
                },
            ])
        
        # Phase 3: Complex Engagement Geometry - learn advanced timing patterns
        if phase >= 3:
            scenarios.extend([
                # Crossing targets
                {
                    'name': 'Fast_Crossing',
                    'range_m': 1500.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (0.0, 40.0, 0.0),  # Fast crossing
                    'tracking_duration': 12.0,
                    'measurement_noise_std': 0.5,
                    'description': 'Fast crossing target - learn lateral motion timing'
                },
                # Diagonal approaches
                {
                    'name': 'Diagonal_Approach',
                    'range_m': 2000.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-30.0, 20.0, 0.0),  # Diagonal approaching
                    'tracking_duration': 15.0,
                    'measurement_noise_std': 0.5,
                    'description': 'Diagonal approaching target - complex engagement geometry'
                },
                # Maneuvering targets (with process noise)
                {
                    'name': 'Maneuvering_Target',
                    'range_m': 1200.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-20.0, 10.0, 0.0),  # Base velocity
                    'tracking_duration': 12.0,
                    'measurement_noise_std': 0.5,
                    'process_noise_std': 0.5,  # High process noise = maneuvering
                    'description': 'Maneuvering target with unpredictable motion'
                },
                # Mixed motion patterns
                {
                    'name': 'Complex_3D_Motion',
                    'range_m': 1800.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (15.0, 25.0, 8.0),  # Complex 3D motion
                    'tracking_duration': 14.0,
                    'measurement_noise_std': 0.5,
                    'description': 'Complex 3D target motion - advanced geometry learning'
                },
            ])
        
        # Phase 4: Add long-range scenarios (2.5-3km)
        if phase >= 4:
            scenarios.extend([
                {
                    'name': 'ExtremeLong_Medium_Slow',
                    'range_m': 2500.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (5.0, 0.0, 0.0),  # Slow receding
                    'tracking_duration': 20.0,
                    'measurement_noise_std': 0.1,
                    'description': 'Extreme long range (2.5km), medium target, slow receding'
                },
                {
                    'name': 'ExtremeLong_Large_Lateral',
                    'range_m': 3000.0,
                    'target_length': 12.0,
                    'target_width': 10.0,
                    'target_height': 5.0,
                    'velocity': (0.0, 15.0, 0.0),  # Moderate lateral
                    'tracking_duration': 20.0,
                    'measurement_noise_std': 0.1,
                    'description': 'Extreme long range (3km), large target, lateral motion'
                },
                {
                    'name': 'ExtremeLong_Medium_Approaching',
                    'range_m': 3000.0,
                    'target_length': 10.0,
                    'target_width': 8.0,
                    'target_height': 4.0,
                    'velocity': (-8.0, 0.0, 0.0),  # Slow approaching
                    'tracking_duration': 30.0,  # Extended to allow closure
                    'measurement_noise_std': 0.1,
                    'description': 'Extreme long range (3km), medium target, approaching - observe HP improvement'
                },
            ])
        
        return scenarios
    
    def _assign_difficulty(self) -> Dict[str, str]:
        """Assign difficulty level to each scenario for monitoring."""
        difficulty = {}
        for scenario in self.scenarios:
            name = scenario['name']
            
            # Approaching targets are "hard" due to safety constraints
            if 'Approaching' in name:
                difficulty[name] = 'hard'
            # Extreme long range (>2000m) is "hard"
            elif scenario['range_m'] >= 2000:
                difficulty[name] = 'hard'
            # Small targets at long range are "hard"
            elif scenario['range_m'] >= 900 and scenario['target_length'] <= 8:
                difficulty[name] = 'hard'
            # Very close range is "hard"
            elif scenario['range_m'] <= 400:
                difficulty[name] = 'hard'
            else:
                difficulty[name] = 'easy'
        
        return difficulty
    
    def get_next_scenario(self):
        """
        Get next scenario in round-robin fashion.
        
        Returns:
            ScenarioParameters object
        """
        scenario_spec = self.scenarios[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.scenarios)
        
        # Create ScenarioParameters directly
        vx, vy, vz = scenario_spec['velocity']
        
        scenario = ScenarioParameters(
            range_m=scenario_spec['range_m'],
            target_length=scenario_spec['target_length'],
            target_width=scenario_spec['target_width'],
            target_height=scenario_spec['target_height'],
            target_vx=vx,
            target_vy=vy,
            target_vz=vz,
            tracking_duration=scenario_spec['tracking_duration'],
            measurement_noise_std=np.full(3, scenario_spec['measurement_noise_std']),
            process_noise_std=np.full(3, scenario_spec.get('process_noise_std', 0.5))  # Default or specified
        )
        
        # Add metadata
        scenario.name = scenario_spec['name']
        scenario.description = scenario_spec['description']
        
        return scenario
    
    def get_scenario_by_index(self, idx: int):
        """Get specific scenario by index."""
        scenario_spec = self.scenarios[idx]
        
        vx, vy, vz = scenario_spec['velocity']
        
        scenario = ScenarioParameters(
            range_m=scenario_spec['range_m'],
            target_length=scenario_spec['target_length'],
            target_width=scenario_spec['target_width'],
            target_height=scenario_spec['target_height'],
            target_vx=vx,
            target_vy=vy,
            target_vz=vz,
            tracking_duration=scenario_spec['tracking_duration'],
            measurement_noise_std=np.full(3, scenario_spec['measurement_noise_std']),
            process_noise_std=np.full(3, scenario_spec.get('process_noise_std', 0.5))
        )
        
        scenario.name = scenario_spec['name']
        scenario.description = scenario_spec['description']
        
        return scenario
    
    def get_all_scenarios(self) -> List:
        """Get all scenarios for evaluation."""
        return [self.get_scenario_by_index(i) for i in range(len(self.scenarios))]
    
    def num_scenarios(self) -> int:
        """Return number of scenarios in current phase."""
        return len(self.scenarios)
    
    def get_phase_requirements(self) -> Dict:
        """Get success criteria for current phase."""
        if self.phase == 1:
            return {
                'min_avg_hp': 0.70,
                'min_hp_per_scenario': 0.60,
                'max_unsafe_scenarios': 0,  # No scenarios can fire at unsafe range
                'description': 'Phase 1: Master basic scenarios'
            }
        elif self.phase == 2:
            return {
                'min_avg_hp': 0.65,
                'min_hp_per_scenario': 0.55,
                'max_unsafe_scenarios': 0,
                'description': 'Phase 2: Handle more variations'
            }
        elif self.phase == 3:
            return {
                'min_avg_hp': 0.60,
                'min_hp_per_scenario': 0.50,
                'max_unsafe_scenarios': 0,
                'description': 'Phase 3: Robust to edge cases'
            }
        else:  # Phase 4
            return {
                'min_avg_hp': 0.55,
                'min_hp_per_scenario': 0.40,  # Achievable at max 3km with proper timing
                'max_unsafe_scenarios': 0,
                'description': 'Phase 4: Extreme long-range mastery (2.5-3km)'
            }
    
    def print_curriculum_info(self):
        """Print curriculum information."""
        print("=" * 80)
        print(f"CURRICULUM PHASE {self.phase}")
        print("=" * 80)
        print(f"Number of scenarios: {len(self.scenarios)}")
        print(f"\nScenarios:")
        
        for i, scenario in enumerate(self.scenarios):
            difficulty = self.difficulty[scenario['name']]
            print(f"  {i+1}. {scenario['name']:30s} [{difficulty:4s}] - {scenario['description']}")
        
        requirements = self.get_phase_requirements()
        print(f"\nSuccess Criteria:")
        print(f"  - Average HP >= {requirements['min_avg_hp']:.2f}")
        print(f"  - Each scenario HP >= {requirements['min_hp_per_scenario']:.2f}")
        print(f"  - All scenarios must fire at safe range (>200m)")
        print("=" * 80)
