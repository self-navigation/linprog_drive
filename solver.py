"""
solver.py — Solver interface and concrete implementations.

Solvers
-------
    KeyboardSolver   WASD manual driving
    GradientSolver   follows the vector field directly — pure proportional
                     heading controller that aligns the robot with V(x)

The LP OCP solver will be added here as a further subclass of BaseSolver.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from map import GridMap


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseSolver(ABC):
    """
    Common interface every solver must implement.

    solve() is called once per simulation step and must return
    the commanded (v, ω) for the unicycle model.

    on_key_press / on_key_release are no-ops by default so solvers that
    do not use keyboard input do not need to override them.
    """

    # Human-readable label shown in the HUD solver line.
    NAME: str = "base"

    @abstractmethod
    def solve(
        self,
        state: np.ndarray,  # [x, y, θ]
        vector_field,  # VectorField instance
        grid_map: "GridMap",
    ) -> Tuple[float, float]:  # (v m/s, ω rad/s)
        ...

    def on_key_press(self, key: int) -> None:
        pass

    def on_key_release(self, key: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Keyboard solver
# ---------------------------------------------------------------------------


class KeyboardSolver(BaseSolver):
    """
    Translates live WASD keyboard state into (v, ω) commands.

    Key mapping
    -----------
        W   full forward speed
        S   full reverse speed
        A   turn left  (positive ω, CCW)
        D   turn right (negative ω, CW)

    Opposite keys cancel: W+S → v=0, A+D → ω=0.
    """

    NAME: str = "keyboard"
    V_MAX: float = 1.0  # m/s
    OMEGA_MAX: float = 1.5  # rad/s

    def __init__(self) -> None:
        self._pressed: set[int] = set()

    def on_key_press(self, key: int) -> None:
        self._pressed.add(key)

    def on_key_release(self, key: int) -> None:
        self._pressed.discard(key)

    def solve(
        self,
        state: np.ndarray,
        vector_field,
        grid_map: "GridMap",
    ) -> Tuple[float, float]:
        import arcade  # deferred — no top-level arcade dependency

        v = omega = 0.0
        if arcade.key.W in self._pressed:
            v += self.V_MAX
        if arcade.key.S in self._pressed:
            v -= self.V_MAX
        if arcade.key.A in self._pressed:
            omega += self.OMEGA_MAX
        if arcade.key.D in self._pressed:
            omega -= self.OMEGA_MAX
        return v, omega


# ---------------------------------------------------------------------------
# Gradient-following solver
# ---------------------------------------------------------------------------


class GradientSolver(BaseSolver):
    """
    Follows the navigation vector field directly.

    At each step:
      1. Query the vector field for the desired direction at the robot's
         current position.
      2. Compute the heading error  Δθ = desired_θ − robot_θ  (wrapped to [-π, π]).
      3. Command full forward speed V_MAX and a proportional yaw correction
         ω = K_ω · Δθ, clamped to ±OMEGA_MAX.

    This is the simplest possible feedback controller — a pure proportional
    heading regulator.  It has no look-ahead and no obstacle constraints of
    its own; it relies on the vector field to route around obstacles and on
    Robot.step()'s collision resolver as a backstop.

    When the vector field returns a zero vector (robot is at the goal, or
    within VectorField.ARRIVAL_RADIUS_M) both v and ω are set to zero.
    """

    NAME: str = "gradient"
    V_MAX: float = 1.0  # m/s   — forward speed (constant when moving)
    OMEGA_MAX: float = 2.0  # rad/s — yaw rate cap
    K_OMEGA: float = 3.0  # proportional gain on heading error (rad/s per rad)

    def solve(
        self,
        state: np.ndarray,
        vector_field,
        grid_map: "GridMap",
    ) -> Tuple[float, float]:
        x, y, theta = float(state[0]), float(state[1]), float(state[2])

        direction = vector_field.query(x, y)

        # Zero vector means "arrived" — stop
        if direction[0] == 0.0 and direction[1] == 0.0:
            return 0.0, 0.0

        desired_theta = math.atan2(float(direction[1]), float(direction[0]))

        # Heading error wrapped to [-π, π]
        err = desired_theta - theta
        err = (err + math.pi) % (2 * math.pi) - math.pi

        omega = max(-self.OMEGA_MAX, min(self.OMEGA_MAX, self.K_OMEGA * err))

        return self.V_MAX, omega
