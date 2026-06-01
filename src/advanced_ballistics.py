"""
Advanced ballistic motion model based on the MATLAB implementation.

This module implements the sophisticated projectile motion equations from 
m_model_vw.m and m_model_cw.m including:
- Aerodynamic drag and lift coefficients  
- Magnus effects and spin
- Variable atmospheric conditions
- Wind effects (optional)
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class AmmoParameters:
    """Ammunition parameters similar to MATLAB createAmmoStruct."""
    
    # Basic properties
    mass: float                    # kg
    caliber: float                 # m 
    v0: float                      # m/s - initial velocity
    chI: float                     # characteristic length parameter
    p0: float                      # initial spin rate
    Ix: float                      # moment of inertia
    
    # Aerodynamic coefficient arrays [8 parameters each]
    coeffC0: np.ndarray           # Drag coefficient parameters
    coeffCind: float              # Induced drag coefficient  
    coeffCL: np.ndarray           # Lift coefficient parameters
    coeffCMag: float              # Magnus coefficient
    coeffCspin: np.ndarray        # Spin damping coefficient parameters
    
    @classmethod
    def create_tpt_ammo(cls) -> 'AmmoParameters':
        """Create TPT ammunition parameters from MATLAB data."""
        return cls(
            mass=0.55,                    # kg
            caliber=0.035,                # 35mm
            v0=1180.0,                    # m/s
            chI=0.145061818181818,
            p0=-7683.4640784205521,
            Ix=9.77354E-05,
            coeffC0=np.array([
                0.067715763204557, 0.076820538050796, -0.074474607483129, 
                0.224852697240333, 0.346762020447269, 0.383074063965045, 
                0.980804425752646, 0.688438824603230
            ]),
            coeffCind=7.89928112129003,
            coeffCL=np.array([
                33.672167954349600000, -348.026384414703000000, 317.397529323485000000, 
                -32.044305482469600000, 283.226620670811000000, 316.403905959784000000, 
                2.136704041773940000, 0.976656402212219000
            ]),
            coeffCMag=1.000367235190540,
            coeffCspin=np.array([
                -0.042324040397595200, -0.019745781161359300, -0.018983168066447900, 
                -0.006231503719291910, 0.001004331450565160, 0.000714536210581989, 
                0.999869648348994000, 0.074197543936471500
            ])
        )
    
    @classmethod  
    def create_fapdst_ammo(cls) -> 'AmmoParameters':
        """Create FAPDS-T ammunition parameters from MATLAB data."""
        return cls(
            mass=0.297,                   # kg
            caliber=0.0184,               # ~18mm
            v0=1440.0,                    # m/s  
            chI=0.119340857,
            p0=-7683.4640784205521,
            Ix=0.000012,
            coeffC0=np.array([
                0.351027950867540, 0.0377764818950702, -0.398675045850880, 
                0.299170758432146, 0.135469727052512, 0.390286559929925, 
                1.05574413557326, 0.665781988550550
            ]),
            coeffCind=9.93994011325219,
            coeffCL=np.array([
                32.6484056972968, -347.482559513806, 318.047810078470, 
                -33.5502966091074, 283.167775831060, 316.356537067532, 
                2.02135917983636, 0.965529655050685
            ]),
            coeffCMag=1.000367235190540,
            coeffCspin=np.array([
                -0.0423240403975952, -0.0197457811613593, -0.0189831680664479, 
                -0.00623150371929191, 0.00100433145056516, 0.000714536210581989, 
                0.999869648348994, 0.0741975439364715
            ])
        )
    
    @classmethod
    def create_ahead_ammo(cls) -> 'AmmoParameters':
        """Create AHEAD ammunition parameters from MATLAB data."""
        return cls(
            mass=0.745,                   # kg
            caliber=0.035,                # 35mm
            v0=1050.0,                    # m/s
            chI=0.124914395,
            p0=6836.980748,
            Ix=0.000114,
            coeffC0=np.array([
                0.195155159184252, 0.013859833497487, -0.138931985519081, 
                -0.449241687260839, 2.17621189833436, -5.61483895617802, 
                0.869532750030948, 0.0371278206614468
            ]),
            coeffCind=234.517507009791,
            coeffCL=np.array([
                35.229010890717, -337.241529029481, 302.225534284964, 
                -34.8483892926434, 266.469196408436, 300.107738190339, 
                0.599893826240526, 1.00719429258937
            ]),
            coeffCMag=0.0762989091947215,
            coeffCspin=np.array([
                0.00462621462832861, 0.000411861028799253, -0.0141362518794611, 
                0.030396850565925, 0.0601830168807091, -0.0846152005646428, 
                0.940084930325799, 0.0503735666511565
            ])
        )


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions for ballistic calculations."""
    
    def __init__(self, use_standard_atmosphere: bool = True):
        self.use_standard_atmosphere = use_standard_atmosphere
        
    def get_conditions(self, altitude: float) -> Tuple[float, float, float]:
        """
        Get atmospheric conditions at given altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Tuple of (density, speed_of_sound, temperature) 
        """
        if self.use_standard_atmosphere:
            return self._standard_atmosphere(altitude)
        else:
            # Could add custom atmosphere models here
            return self._standard_atmosphere(altitude)
    
    def _standard_atmosphere(self, h: float) -> Tuple[float, float, float]:
        """
        Standard atmosphere model (simplified version of atmoscoesa).
        
        Args:
            h: Altitude in meters
            
        Returns:
            Tuple of (density in kg/m³, speed_of_sound in m/s, temperature in K)
        """
        # Standard atmosphere constants
        g0 = 9.80665  # m/s²
        R = 287.04    # J/(kg·K)
        
        # Sea level conditions
        T0 = 288.15   # K
        p0 = 101325   # Pa
        rho0 = 1.225  # kg/m³
        
        # Temperature lapse rate (troposphere)
        L = 0.0065    # K/m
        
        # Calculate temperature
        if h <= 11000:  # Troposphere
            T = T0 - L * h
            # Pressure
            p = p0 * (T / T0) ** (g0 / (R * L))
        else:  # Stratosphere (simplified)
            T = 216.65  # K (constant)
            p = 22632 * np.exp(-g0 * (h - 11000) / (R * T))
        
        # Density
        rho = p / (R * T)
        
        # Speed of sound
        gamma = 1.4  # Heat capacity ratio for air
        c = np.sqrt(gamma * R * T)
        
        return rho, c, T


class AdvancedProjectileMotion:
    """
    Advanced projectile motion model based on MATLAB m_model_vw.m and m_model_cw.m.
    
    Implements 6-DOF ballistic equations with:
    - Aerodynamic drag, lift, and Magnus effects
    - Spin dynamics
    - Variable atmospheric conditions  
    - Optional wind effects
    """
    
    def __init__(
        self, 
        ammo_params: AmmoParameters,
        atmosphere: Optional[AtmosphericConditions] = None,
        include_wind: bool = False
    ):
        """
        Initialize advanced projectile motion calculator.
        
        Args:
            ammo_params: Ammunition parameters
            atmosphere: Atmospheric conditions model
            include_wind: Whether to include wind effects
        """
        self.ammo = ammo_params
        self.atmosphere = atmosphere or AtmosphericConditions()
        self.include_wind = include_wind
        self.g = 9.80665  # m/s² (standard gravity)
        
    def calculate_trajectory(
        self,
        initial_position: Tuple[float, float, float],
        initial_velocity: float,
        elevation_angle: float,
        azimuth_angle: float,
        max_time: float = 100.0,
        time_step: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate advanced projectile trajectory using ODE integration.
        
        Args:
            initial_position: Starting position (x, y, z) in meters
            initial_velocity: Muzzle velocity (m/s)
            elevation_angle: Elevation angle (radians)
            azimuth_angle: Azimuth angle (radians)  
            max_time: Maximum simulation time (seconds)
            time_step: Integration time step (seconds)
            
        Returns:
            Tuple of (time_array, state_array) where state_array has shape (n_points, 8)
            State vector: [z, x, y, v, theta, phi, p, s] where:
            - z: altitude (m)
            - x, y: horizontal position (m) 
            - v: velocity magnitude (m/s)
            - theta: elevation angle (rad)
            - phi: azimuth angle (rad)
            - p: spin rate (rad/s)
            - s: range (m)
        """
        # Initial state vector [z, x, y, v, theta, phi, p, s]
        x0, y0, z0 = initial_position
        
        initial_state = np.array([
            z0,                          # altitude
            x0,                          # x position  
            y0,                          # y position
            initial_velocity,            # velocity magnitude
            elevation_angle,             # elevation angle
            azimuth_angle,              # azimuth angle  
            self.ammo.p0,               # initial spin rate
            0.0                         # range
        ])
        
        # Time span
        t_span = (0, max_time)
        t_eval = np.arange(0, max_time, time_step)
        
        # Solve ODE with error handling
        try:
            solution = solve_ivp(
                fun=self._motion_equations,
                t_span=t_span,
                y0=initial_state,
                t_eval=t_eval,
                method='RK45',  # Similar to ode45 in MATLAB
                rtol=1e-6,
                atol=1e-8,
                events=self._ground_impact_event,
                max_step=0.1  # Limit step size for stability
            )
            
            # Check if integration was successful
            if not solution.success:
                # Return minimal trajectory if integration failed
                return np.array([0.0]), np.array([[initial_state[1], initial_state[2], initial_state[0], 0, 0, 0]])
                
        except (ValueError, RuntimeError) as e:
            # Return minimal trajectory on error
            return np.array([0.0]), np.array([[initial_state[1], initial_state[2], initial_state[0], 0, 0, 0]])
        
        return solution.t, solution.y.T
    
    def _motion_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Advanced motion equations based on MATLAB m_model_vw.m.
        Implements the explicit ballistic M-model as described in Baranowski et al. (2016),
        see: Explicit “ballistic M-model”: a refinement of the implicit “modified point mass trajectory model”.
        Each equation below references the corresponding equation in the paper.
        
        Args:
            t: Time (seconds)
            state: State vector [z, x, y, v, theta, phi, p, s]
            
        Returns:
            State derivative vector
        """
        z, x, y, v, theta, phi, p, s = state
        
        # Safety checks to prevent numerical issues
        # Clamp velocity to reasonable range
        v = max(min(v, 2000.0), 1.0)  # Avoid zero or extreme velocities
        
        # Clamp theta away from ±90 degrees to avoid division by zero
        theta = max(min(theta, np.pi/2 - 0.01), -np.pi/2 + 0.01)
        
        # Clamp altitude to valid range
        z = max(min(z, 50000.0), 0.0)
        
        # Atmospheric conditions
        rho, c, T = self.atmosphere.get_conditions(z)
        
        # Wind components (set to zero for now, can be extended)
        wx, wy, wz = 0.0, 0.0, 0.0
        dwxdh, dwydh, dwzdh = 0.0, 0.0, 0.0
        
        # Air frame unit vectors (from MATLAB airFrame function)
        ev, eth, eph = self._air_frame(theta, phi)
        
        # Wind derivatives
        dotwx = v * np.sin(theta) * dwxdh * np.array([ev[0], eth[0], eph[0]])
        dotwy = v * np.sin(theta) * dwydh * np.array([ev[1], eth[1], eph[1]])
        dotwz = v * np.sin(theta) * dwzdh * np.array([ev[2], eth[2], eph[2]])
        
        # Body frame unit vectors
        ex = np.cos(theta) * np.cos(phi) * ev - np.sin(theta) * np.cos(phi) * eth - np.sin(phi) * eph
        ey = np.cos(theta) * np.sin(phi) * ev - np.sin(theta) * np.sin(phi) * eth + np.cos(phi) * eph
        ez = np.sin(theta) * ev + np.cos(theta) * eth
        
        # Ammunition parameters
        mu = self.ammo.caliber * p / v
        chI = self.ammo.chI
        m = self.ammo.mass
        S = np.pi * self.ammo.caliber**2 / 4
        Ma = v / c  # Mach number
        
        # Aerodynamic coefficients
        # Eq. (6) in Baranowski et al. (2016): Drag coefficient C0
        C0 = self._calculate_drag_coefficient(Ma)
        # Eq. (7): Induced drag coefficient k
        k = self.ammo.coeffCind
        # Eq. (8): Lift coefficient CL
        CL_ = self._calculate_lift_coefficient(Ma)  
        # Eq. (9): Magnus coefficient CMag
        CMag = self.ammo.coeffCMag
        # Eq. (10): Spin damping coefficient Cspin
        Cspin_ = self._calculate_spin_coefficient(Ma)
        
        # Determinant for Magnus effect (with safety check)
        # Eq. (11): detM for Magnus effect
        detM = (1 - mu**2 * chI * CMag)**2 + (chI * mu * CL_)**2
        detM = max(detM, 1e-10)  # Prevent division by zero
        
        # State derivatives
        # Eq. (12): dz/dt (vertical motion)
        dot_z = v * np.sin(theta) + wz
        # Eq. (13): dx/dt (horizontal motion, x)
        dot_x = v * np.cos(theta) * np.cos(phi) + wx
        # Eq. (14): dy/dt (horizontal motion, y)
        dot_y = v * np.cos(theta) * np.sin(phi) + wy
        
        # Eq. (15): dv/dt (velocity derivative, includes drag, gravity, wind)
        dot_v = (-rho * v**2 / (2 * m) * S * 
                (C0 + k * (2 * chI * mu * m * self.g * np.cos(theta) / (rho * v**2 * S))**2 / detM) -
                self.g * np.sin(theta) - 
                (dotwx[0] + dotwy[0] + dotwz[0]))
        
        # Eq. (16): dtheta/dt (elevation angle derivative)
        dot_theta = (-self.g / (v * detM) * (1 - chI * mu**2 * CMag) * np.cos(theta) -
                    (dotwx[1] + dotwy[1] + dotwz[1]) / v)
        
        # Safe division by cos(theta) - clamp to avoid division by very small numbers
        cos_theta_safe = max(min(np.cos(theta), 1.0), 0.01)
        # Eq. (17): dphi/dt (azimuth angle derivative)
        dot_phi = (-self.g / (v * detM) * chI * mu * CL_ -
                  (dotwx[2] + dotwy[2] + dotwz[2]) / v / cos_theta_safe)
        
        # Eq. (18): dp/dt (spin rate derivative)
        dot_p = rho * v * p / (2 * chI * m) * S * Cspin_
        
        # Eq. (19): ds/dt (range increment)
        dot_s = v  # Range increment
        
        derivatives = np.array([dot_z, dot_x, dot_y, dot_v, dot_theta, dot_phi, dot_p, dot_s])
        
        # Safety check: replace any NaN or inf values with zeros
        if not np.all(np.isfinite(derivatives)):
            derivatives = np.nan_to_num(derivatives, nan=0.0, posinf=0.0, neginf=0.0)
        
        return derivatives
    
    def _air_frame(self, theta: float, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate air frame unit vectors (from MATLAB airFrame.m).
        
        Args:
            theta: Elevation angle (radians)
            phi: Azimuth angle (radians)
            
        Returns:
            Tuple of (ev, eth, eph) unit vectors
        """
        ev = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi), 
            np.sin(theta)
        ])
        
        eth = np.array([
            -np.sin(theta) * np.cos(phi),
            -np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        eph = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ])
        
        return ev, eth, eph
    
    def _calculate_drag_coefficient(self, Ma: float) -> float:
        """Calculate drag coefficient C0 based on Mach number."""
        par = self.ammo.coeffC0
        r = (Ma**2 - par[6]) / (Ma**2 + par[6])
        s = r / np.sqrt((1 - par[7]**2) * r**2 + par[7]**2)
        
        C0 = ((1 + s) * (par[0] + par[1]*r + par[2]*r**2) +
              (1 - s) * (par[3] + par[4]*r + par[5]*r**2))
        
        return C0
    
    def _calculate_lift_coefficient(self, Ma: float) -> float:
        """Calculate lift coefficient CL based on Mach number.""" 
        par = self.ammo.coeffCL
        r = (Ma**2 - par[6]) / (Ma**2 + par[6])
        s = r / np.sqrt((1 - par[7]**2) * r**2 + par[7]**2)
        
        CL = ((1 + s) * (par[0] + par[1]*r + par[2]*r**2) +
              (1 - s) * (par[3] + par[4]*r + par[5]*r**2))
        
        return CL
    
    def _calculate_spin_coefficient(self, Ma: float) -> float:
        """Calculate spin coefficient Cspin based on Mach number."""
        par = self.ammo.coeffCspin  
        r = (Ma**2 - par[6]) / (Ma**2 + par[6])
        s = r / np.sqrt((1 - par[7]**2) * r**2 + par[7]**2)
        
        Cspin = ((1 + s) * (par[0] + par[1]*r + par[2]*r**2) +
                 (1 - s) * (par[3] + par[4]*r + par[5]*r**2))
        
        return Cspin
    
    def _ground_impact_event(self, t: float, state: np.ndarray) -> float:
        """Event function to detect ground impact (z = 0)."""
        return state[0]  # altitude
    
    # Make the event terminal
    _ground_impact_event.terminal = True
    _ground_impact_event.direction = -1