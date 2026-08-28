"""
Gun (artillery piece) domain model and gun-aware physics wrappers.

Introduces ``GunParameters`` with an explicit position plus thin wrappers that route the
existing single-shooter ballistics (``find_optimal_firing_angles``) and hit-probability
calculation through a gun. A gun at the origin reproduces the current single-shooter
behaviour exactly, so existing code and tests are unaffected.

Runtime state (cooldown, rounds remaining) is not modelled yet; the reload/magazine
fields here are static configuration only for now.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.find_optimal_firing_angles import find_optimal_firing_angles


@dataclass
class GunParameters:
    """Static parameters of a single artillery gun."""

    gun_id: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    ammo_type: str = "tpt"
    muzzle_velocity: float = 1000.0
    traverse_limits: Tuple[float, float] = (-np.pi, np.pi)  # azimuth bounds (rad)
    elevation_limits: Tuple[float, float] = (np.radians(-5.0), np.radians(85.0))
    min_range_m: float = 200.0
    max_range_m: float = 5000.0
    reload_time_s: float = 0.0
    magazine: int = 1000

    def __post_init__(self):
        # Always store the position as a length-3 float array.
        self.position = np.asarray(self.position, dtype=float).reshape(3)

    @classmethod
    def default_at_origin(cls) -> "GunParameters":
        """The implicit gun assumed by the current single-shooter code: origin, TPT."""
        return cls()


@dataclass
class FiringSolution:
    """Result of solving a gun's firing solution against a target."""

    elevation: float
    azimuth: float
    time_to_impact: Optional[float]
    hit_point: Optional[np.ndarray]
    miss_distance: float
    feasible: bool
    infeasible_reason: str = ""


def _check_feasibility(gun: GunParameters, target_init, elevation: float, azimuth: float):
    """Return (feasible, reason) for a candidate solution given the gun's limits."""
    rel = np.asarray(target_init, dtype=float).reshape(3) - gun.position
    range_m = float(np.linalg.norm(rel))
    if range_m < gun.min_range_m:
        return False, f"range {range_m:.0f}m below min {gun.min_range_m:.0f}m"
    if range_m > gun.max_range_m:
        return False, f"range {range_m:.0f}m above max {gun.max_range_m:.0f}m"
    az_min, az_max = gun.traverse_limits
    if not (az_min <= azimuth <= az_max):
        return False, f"azimuth {np.degrees(azimuth):.1f}deg outside traverse"
    el_min, el_max = gun.elevation_limits
    if not (el_min <= elevation <= el_max):
        return False, f"elevation {np.degrees(elevation):.1f}deg outside limits"
    return True, ""


def solve_firing_solution(
    gun: GunParameters,
    target_init,
    target_vel,
    max_time: float = 30.0,
    adaptive_mode: str = "two_phase",
) -> FiringSolution:
    """Solve ``gun``'s firing solution against a target, adding feasibility checks.

    Wraps ``find_optimal_firing_angles`` using the gun's position, muzzle velocity and
    ammo type. For a gun at the origin this reproduces ``find_optimal_firing_angles``
    exactly (the wrapper only adds the feasibility verdict).
    """
    elev, azim, t_impact, hit_point, min_dist = find_optimal_firing_angles(
        gun.position,
        np.asarray(target_init, dtype=float).reshape(3),
        np.asarray(target_vel, dtype=float).reshape(3),
        gun.muzzle_velocity,
        gun.ammo_type,
        max_time=max_time,
        adaptive_mode=adaptive_mode,
    )
    feasible, reason = _check_feasibility(gun, target_init, elev, azim)
    return FiringSolution(
        elevation=elev,
        azimuth=azim,
        time_to_impact=t_impact,
        hit_point=hit_point,
        miss_distance=min_dist,
        feasible=feasible,
        infeasible_reason=reason,
    )


def hit_probability_for_gun(calculator, gun: GunParameters, target_position,
                            target_velocity, **kwargs) -> float:
    """Hit probability of ``gun`` against a target.

    Transforms the target into the gun's frame (subtract the gun position) and delegates
    to the existing origin-based ``calculator.calculate_hit_probability``. For a gun at
    the origin the relative position equals the absolute position, so the result is
    identical to calling the calculator directly.
    """
    rel_pos = np.asarray(target_position, dtype=float).reshape(3) - gun.position
    return calculator.calculate_hit_probability(tuple(rel_pos), target_velocity, **kwargs)


@dataclass
class Battery:
    """A collection of guns — the ``M`` in the M-guns extension.

    Lets you parameterise the number of guns, e.g. ``Battery.line(n_guns=4)``. This is a
    pure domain container; wiring a battery into the RL environment / training loop is
    future work.
    """

    guns: List[GunParameters]

    @property
    def n_guns(self) -> int:
        return len(self.guns)

    def __len__(self) -> int:
        return len(self.guns)

    def __iter__(self):
        return iter(self.guns)

    def __getitem__(self, index) -> GunParameters:
        return self.guns[index]

    @classmethod
    def line(
        cls,
        n_guns: int,
        spacing: float = 50.0,
        ammo_type: str = "tpt",
        muzzle_velocity: float = 1000.0,
        origin=(0.0, 0.0, 0.0),
        axis: str = "y",
    ) -> "Battery":
        """Build ``n_guns`` evenly spaced along an axis, centred on ``origin``."""
        if n_guns < 1:
            raise ValueError("n_guns must be >= 1")
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y' or 'z'")
        base = np.asarray(origin, dtype=float).reshape(3)
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        offsets = (np.arange(n_guns) - (n_guns - 1) / 2.0) * spacing
        guns = []
        for i, off in enumerate(offsets):
            pos = base.copy()
            pos[axis_idx] += off
            guns.append(
                GunParameters(
                    gun_id=i, position=pos, ammo_type=ammo_type,
                    muzzle_velocity=muzzle_velocity,
                )
            )
        return cls(guns=guns)

    @classmethod
    def from_positions(
        cls, positions, ammo_type: str = "tpt", muzzle_velocity: float = 1000.0
    ) -> "Battery":
        """Build a battery from an explicit list of gun positions."""
        return cls(
            guns=[
                GunParameters(
                    gun_id=i, position=p, ammo_type=ammo_type,
                    muzzle_velocity=muzzle_velocity,
                )
                for i, p in enumerate(positions)
            ]
        )
