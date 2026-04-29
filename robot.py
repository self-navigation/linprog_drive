"""
robot.py — Unicycle robot state, dynamics, and collision resolution.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from map import GridMap


class Robot:
    """
    Unicycle robot with Euler-integrated kinematics and grid collision.

    State : [x (m),  y (m),  θ (rad)]
    Input : v (m/s forward speed),  ω (rad/s yaw rate)

    Kinematics
    ----------
        x_new = x + v·cos(θ)·dt
        y_new = y + v·sin(θ)·dt
        θ_new = θ + ω·dt            (wrapped to [-π, π])

    Collision
    ---------
    The robot's footprint is a circle of radius RADIUS_M.
    After each Euler step the position is corrected by
    _resolve_collision(), which pushes the centre out of any
    obstacle cells it overlaps.

    Collision is cell-based: we iterate over every grid cell whose
    axis-aligned bounding box overlaps the robot circle and, for each
    obstacle cell, compute the closest point on that cell's square to
    the robot centre. If the distance is less than RADIUS_M the robot
    is pushed back along the contact normal by the penetration depth.

    This approach is deliberately map-format-agnostic: it only uses the
    GridMap.grid boolean array and GridMap.cell_size, so it works
    identically whether the map was loaded from a .map text file, a PNG
    image, or anything else.
    """

    # Physical footprint radius — also used by simulation.py for rendering.
    RADIUS_M: float = 0.3  # metres

    # Maximum collision-resolution passes per step.
    # More passes = more accurate corner handling, rarely needs > 3.
    _MAX_RESOLVE_ITERS: int = 8

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

    def step(
        self,
        v: float,
        omega: float,
        dt: float,
        grid_map: Optional["GridMap"] = None,
    ) -> None:
        """
        Advance state one Euler step of length dt seconds.

        If grid_map is provided, collision with obstacle cells is
        resolved after integration.
        """
        x, y, th = self.state

        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += omega * dt
        th = (th + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]

        if grid_map is not None:
            x, y = self._resolve_collision(x, y, grid_map)

        self.state[0] = x
        self.state[1] = y
        self.state[2] = th
        self.v_last = v
        self.omega_last = omega

    # ------------------------------------------------------------------
    # Collision resolution
    # ------------------------------------------------------------------

    def _resolve_collision(
        self, x: float, y: float, grid_map: "GridMap"
    ) -> tuple[float, float]:
        """
        Push (x, y) out of any obstacle cells it overlaps.

        Algorithm — circle vs axis-aligned cell (AABB):
          For each obstacle cell whose bounding box overlaps the robot
          circle, find the closest point on that cell's square to the
          robot centre.  If the distance is less than RADIUS_M, the
          robot is penetrating; push it back along the contact normal
          (robot_centre → closest_point, reversed) by the penetration
          depth.

          We repeat up to _MAX_RESOLVE_ITERS times because resolving
          one contact can create a new one (e.g. a corner squeeze).
        """
        r = self.RADIUS_M
        cs = grid_map.cell_size

        for _ in range(self._MAX_RESOLVE_ITERS):
            # Candidate cells: every cell whose square could touch the circle.
            gx_lo = max(0, math.floor((x - r) / cs))
            gx_hi = min(grid_map.cols - 1, math.floor((x + r) / cs) + 1)
            gy_lo = max(0, math.floor((y - r) / cs))
            gy_hi = min(grid_map.rows - 1, math.floor((y + r) / cs) + 1)

            any_contact = False

            for gy in range(gy_lo, gy_hi + 1):
                for gx in range(gx_lo, gx_hi + 1):
                    if not grid_map.is_obstacle(gx, gy):
                        continue

                    # Cell world-space bounds [x0, x1) × [y0, y1)
                    x0, x1 = gx * cs, (gx + 1) * cs
                    y0, y1 = gy * cs, (gy + 1) * cs

                    # Closest point on the cell square to the robot centre
                    cx = max(x0, min(x, x1))
                    cy = max(y0, min(y, y1))

                    dx = x - cx
                    dy = y - cy
                    dist_sq = dx * dx + dy * dy

                    if dist_sq >= r * r:
                        continue  # no penetration

                    any_contact = True
                    dist = math.sqrt(dist_sq)

                    if dist < 1e-9:
                        # Robot centre is exactly on the cell boundary or
                        # inside the cell — push straight up as a fallback.
                        dx, dy, dist = 0.0, 1.0, 1.0

                    # Push robot out by the penetration depth
                    penetration = r - dist
                    x += (dx / dist) * penetration
                    y += (dy / dist) * penetration

            if not any_contact:
                break  # clean pass — no further resolution needed

        return x, y
