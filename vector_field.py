"""
vector_field.py — Navigation potential field via geodesic distance transform.

Algorithm
---------
Run Dijkstra's shortest-path algorithm from the goal cell outward through
all free cells, using 8-connected grid edges with weights:

    axial     (N/S/E/W)  :  cell_size
    diagonal  (NE/…/SW)  :  cell_size × √2

The result is a geodesic distance field  d[gy, gx]  — the actual shortest
path length in metres from every free cell to the goal, routing around
obstacles.  d = 0 at the goal; obstacle cells get d = ∞.

Compared with the harmonic (Laplace) approach:

    Laplace        — smooth C∞ field, no local minima, but concentrates
                     nearly all variation within a small radius of the goal.
                     90 % of a large map can be numerically indistinguishable
                     from the boundary value (φ ≈ 1), making the gradient
                     too small to steer from.

    Dijkstra / FMM — distance increases linearly with actual path length, so
                     every cell has a value proportional to how far it really
                     is from the goal.  Gradient is ≈ 1 everywhere in free
                     space, and the colourmap shows the full topology of the
                     environment at a glance.

Normalisation
-------------
After computing raw distances the field is normalised to φ ∈ [0, 1]:

    φ = d / d_max       where d_max = max finite distance in the map

This keeps the interface identical to the previous version so the rest of
the pipeline (simulation.py, LP cost weights) is unchanged.

Vector field
------------
    V(x) = −∇φ(x) / ‖∇φ(x)‖

Because Dijkstra geodesic distances satisfy ‖∇d‖ ≈ 1 everywhere in free
space (the eikonal property), the gradient is well-defined and approximately
unit-magnitude at every reachable cell.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from map import GridMap


class VectorField:
    """
    Navigation potential field computed from geodesic (Dijkstra) distances.

    Usage
    -----
    vf = VectorField()
    vf.compute(grid_map, goal_world)   # once — O((rows×cols) log(rows×cols))
    direction = vf.query(wx, wy)       # unit vector toward goal
    cost      = vf.potential(wx, wy)   # scalar φ ∈ [0, 1]
    """

    # Return zero from query() when already this close to the goal (metres).
    ARRIVAL_RADIUS_M: float = 0.15

    # Below this gradient magnitude fall back to straight-to-goal.
    # Should only trigger at the goal cell itself.
    _GRAD_THRESHOLD: float = 1e-6

    def __init__(self) -> None:
        self._phi: np.ndarray | None = None
        self._vx: np.ndarray | None = None
        self._vy: np.ndarray | None = None
        self._cs: float = 1.0
        self._goal_world: tuple[float, float] | None = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def phi(self) -> np.ndarray | None:
        """Normalised potential array φ ∈ [0, 1], shape (rows, cols)."""
        return self._phi

    def compute(
        self,
        grid_map: "GridMap",
        goal_world: Tuple[float, float],
    ) -> None:
        """
        Run Dijkstra from the goal cell and build the navigation field.

        Parameters
        ----------
        grid_map   : GridMap with boolean obstacle grid and cell_size.
        goal_world : (wx, wy) goal position in metres.
        """
        self._cs = grid_map.cell_size
        self._goal_world = goal_world
        rows, cols = grid_map.rows, grid_map.cols
        gx_g, gy_g = grid_map.goal

        # ── 1. Geodesic distances via Dijkstra ────────────────────────
        dist = self._dijkstra(grid_map.grid, gy_g, gx_g)

        # ── 2. Normalise to φ ∈ [0, 1] ───────────────────────────────
        finite = np.isfinite(dist)
        n_reachable = int(finite.sum())

        if n_reachable == 0:
            self._phi = np.ones((rows, cols))
            self._phi[gy_g, gx_g] = 0.0
            self._vx = np.zeros((rows, cols))
            self._vy = np.zeros((rows, cols))
            self._ready = True
            return

        d_max = float(dist[finite].max())
        if d_max < 1e-12:
            d_max = 1.0

        # Obstacle cells get d_max so np.gradient sees a plateau
        # at the boundary rather than a ∞ discontinuity.
        phi = dist.copy()
        phi[~finite] = d_max
        phi /= d_max
        self._phi = phi

        # ── 3. Vector field = −∇φ ─────────────────────────────────────
        # np.gradient: axis-0 = rows (y direction), axis-1 = cols (x).
        grad = np.gradient(phi)
        self._vy = -grad[0]  # V_y = −dφ/dy
        self._vx = -grad[1]  # V_x = −dφ/dx

        # Zero the field inside actual obstacles — undefined there.
        self._vx[grid_map.grid] = 0.0
        self._vy[grid_map.grid] = 0.0

        self._ready = True
        print(
            f"[VectorField] Dijkstra: {n_reachable} reachable cells, "
            f"d_max = {d_max:.2f} m"
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query(self, wx: float, wy: float) -> np.ndarray:
        """
        Return the unit navigation vector at world position (wx, wy).

        Returns the zero vector within ARRIVAL_RADIUS_M of the goal.
        Falls back to a straight-to-goal direction if the interpolated
        gradient is below _GRAD_THRESHOLD (only at the goal cell itself).
        """
        if not self._ready:
            return np.zeros(2)

        # Zero at goal
        if self._goal_world is not None:
            gx_w, gy_w = self._goal_world
            if (wx - gx_w) ** 2 + (wy - gy_w) ** 2 < self.ARRIVAL_RADIUS_M**2:
                return np.zeros(2)

        vx = self._bilinear(self._vx, wx, wy)
        vy = self._bilinear(self._vy, wx, wy)
        mag = math.sqrt(vx * vx + vy * vy)

        if mag >= self._GRAD_THRESHOLD:
            return np.array([vx / mag, vy / mag])

        # Fallback: straight to goal
        if self._goal_world is None:
            return np.zeros(2)
        gx_w, gy_w = self._goal_world
        dx, dy = gx_w - wx, gy_w - wy
        d = math.sqrt(dx * dx + dy * dy)
        return np.zeros(2) if d < 1e-9 else np.array([dx / d, dy / d])

    def potential(self, wx: float, wy: float) -> float:
        """
        Return φ ∈ [0, 1] at world position (wx, wy).
        0 at the goal, 1 at the farthest reachable cell.
        """
        if not self._ready:
            return 0.0
        return float(self._bilinear(self._phi, wx, wy))

    # ------------------------------------------------------------------
    # Dijkstra on 8-connected grid
    # ------------------------------------------------------------------

    def _dijkstra(
        self,
        grid: np.ndarray,
        start_gy: int,
        start_gx: int,
    ) -> np.ndarray:
        """
        Geodesic distances from (start_gy, start_gx) to all reachable cells.

        Axial edges have weight cell_size; diagonal edges cell_size × √2.
        Obstacle cells are impassable; unreachable free cells keep np.inf.
        """
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)

        if grid[start_gy, start_gx]:
            return dist

        cs = self._cs
        diag_cs = cs * math.sqrt(2)

        _MOVES = (
            (-1, 0, cs),
            (1, 0, cs),
            (0, -1, cs),
            (0, 1, cs),
            (-1, -1, diag_cs),
            (-1, 1, diag_cs),
            (1, -1, diag_cs),
            (1, 1, diag_cs),
        )

        dist[start_gy, start_gx] = 0.0
        heap = [(0.0, start_gy, start_gx)]

        while heap:
            d, gy, gx = heapq.heappop(heap)
            if d > dist[gy, gx]:
                continue  # stale entry

            for dy, dx, w in _MOVES:
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < rows and 0 <= nx < cols and not grid[ny, nx]:
                    nd = d + w
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, ny, nx))

        return dist

    # ------------------------------------------------------------------
    # Bilinear interpolation (cell-centred)
    # ------------------------------------------------------------------

    def _bilinear(self, arr: np.ndarray, wx: float, wy: float) -> float:
        """
        Bilinear interpolation at world point (wx, wy) for a cell-centred array.
        Cell (gx, gy) has its centre at ((gx + 0.5)·cs, (gy + 0.5)·cs).
        """
        cs = self._cs
        rows, cols = arr.shape

        gxf = wx / cs - 0.5
        gyf = wy / cs - 0.5
        gx0 = int(math.floor(gxf))
        tx = gxf - gx0
        gy0 = int(math.floor(gyf))
        ty = gyf - gy0
        gx0 = max(0, min(gx0, cols - 1))
        gx1 = min(gx0 + 1, cols - 1)
        gy0 = max(0, min(gy0, rows - 1))
        gy1 = min(gy0 + 1, rows - 1)

        return (
            arr[gy0, gx0] * (1 - tx) * (1 - ty)
            + arr[gy0, gx1] * tx * (1 - ty)
            + arr[gy1, gx0] * (1 - tx) * ty
            + arr[gy1, gx1] * tx * ty
        )
