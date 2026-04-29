"""
solver.py — Solver interface and concrete implementations.

The real LP OCP solver will be added here as a subclass of BaseSolver.
For now, KeyboardSolver lets a human drive the robot with WASD so
the full simulation loop can be exercised before the LP is written.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    # Avoid circular imports at runtime; used only for type hints.
    from map import GridMap


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseSolver(ABC):
    """
    Common interface every solver must implement.

    solve() is called once per simulation step and must return
    the commanded (v, ω) for the unicycle model.
    """

    @abstractmethod
    def solve(
        self,
        state: np.ndarray,  # [x, y, θ]
        vector_field,  # VectorField instance (may be dummy)
        grid_map: "GridMap",
    ) -> Tuple[float, float]:  # (v m/s, ω rad/s)
        ...


# ---------------------------------------------------------------------------
# Keyboard mock solver
# ---------------------------------------------------------------------------


class KeyboardSolver(BaseSolver):
    """
    Translates live WASD keyboard state into (v, ω) commands.

    The simulation calls on_key_press / on_key_release to keep this
    solver's internal key-set in sync with the window's events.

    Key mapping
    -----------
        W   full forward speed
        S   full reverse speed
        A   turn left  (positive ω, CCW in our coordinate system)
        D   turn right (negative ω, CW)

    Holding W+A simultaneously drives forward while turning left, etc.
    Opposite keys cancel: W+S → v=0, A+D → ω=0.
    """

    V_MAX: float = 1.0  # m/s   — max linear speed
    OMEGA_MAX: float = 1.5  # rad/s — max yaw rate

    def __init__(self) -> None:
        self._pressed: set[int] = set()

    # ------------------------------------------------------------------
    # Key state management  (called by simulation.py)
    # ------------------------------------------------------------------

    def on_key_press(self, key: int) -> None:
        self._pressed.add(key)

    def on_key_release(self, key: int) -> None:
        self._pressed.discard(key)

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    def solve(
        self,
        state: np.ndarray,
        vector_field,
        grid_map: "GridMap",
    ) -> Tuple[float, float]:
        """Return (v, ω) from the current keyboard state."""
        import arcade  # imported here so solver.py has no top-level arcade dep

        v = 0.0
        omega = 0.0

        if arcade.key.W in self._pressed:
            v += self.V_MAX
        if arcade.key.S in self._pressed:
            v -= self.V_MAX
        if arcade.key.A in self._pressed:
            omega += self.OMEGA_MAX
        if arcade.key.D in self._pressed:
            omega -= self.OMEGA_MAX

        return v, omega
