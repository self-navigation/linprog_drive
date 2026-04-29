"""
robot.py — Unicycle robot state and dynamics.
"""

from __future__ import annotations

import math
import numpy as np


class Robot:
    """
    Unicycle robot with Euler-integrated kinematics.

    State : [x (m),  y (m),  θ (rad)]
    Input : v (m/s forward speed),  ω (rad/s yaw rate)

    Kinematics
    ----------
        x_new = x + v·cos(θ)·dt
        y_new = y + v·sin(θ)·dt
        θ_new = θ + ω·dt            (wrapped to [-π, π])
    """

    # Physical footprint radius — also used by simulation.py for rendering.
    RADIUS_M: float = 0.3  # metres

    def __init__(self, x: float, y: float, theta: float = 0.0) -> None:
        self.state = np.array([x, y, theta], dtype=float)

        # Last applied commands — available as linearisation point for the LP.
        self.v_last: float = 0.0
        self.omega_last: float = 0.0

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def theta(self) -> float:
        return float(self.state[2])

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self, v: float, omega: float, dt: float) -> None:
        """Advance state one Euler step of length dt seconds."""
        x, y, th = self.state
        self.state[0] += v * math.cos(th) * dt
        self.state[1] += v * math.sin(th) * dt
        self.state[2] += omega * dt

        # Wrap heading to [-π, π]
        self.state[2] = (self.state[2] + math.pi) % (2 * math.pi) - math.pi

        self.v_last = v
        self.omega_last = omega
