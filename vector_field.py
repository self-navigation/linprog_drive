"""
vector_field.py — Navigation potential via cost-weighted geodesic distances.

Pipeline
--------
1. EDT pass   — multi-source Dijkstra from every obstacle cell outward.
                Produces d_obs[gy, gx]: distance (metres) to the nearest
                obstacle for every free cell.

2. Cost field — per-cell traversal cost derived from d_obs:

        c(x) = 1  +  alpha * max(0,  (r - d_obs(x)) / r) ** 2

   c = 1 everywhere beyond radius r  (unaffected open space).
   c rises smoothly to  1 + alpha  at the obstacle boundary.
   The quadratic ramp keeps the gradient continuous.

3. Navigation pass — Dijkstra from the goal with edge weights:

        w(u, v) = raw_dist(u, v) * 0.5 * (c(u) + c(v))

   where raw_dist is cell_size or cell_size*sqrt(2) for diagonal edges.
   Cells near obstacles accumulate distance faster, so optimal paths
   naturally route through open space without sealing any corridor.

4. Normalisation — T / T_max  →  phi in [0, 1].

5. Vector field — V = -grad(phi) / |grad(phi)|.

Parameters
----------
repulsion_radius : float   metres — repulsion zone around obstacles.
repulsion_alpha  : float   dimensionless — extra cost at obstacle boundary.
                           0 = pure Dijkstra (current behaviour).
                           3 = moderate repulsion.
                           10 = strong repulsion.
"""

from __future__ import annotations

import heapq
import math
from typing import Tuple

import numpy as np

from map import GridMap


class VectorField:
    """
    Cost-weighted navigation potential field.

    Usage
    -----
    vf = VectorField()
    vf.compute(grid_map, goal_world)        # default parameters
    vf.compute(grid_map, goal_world,
               repulsion_radius=0.5,
               repulsion_alpha=5.0)         # stronger wall avoidance

    direction = vf.query(wx, wy)            # unit vector toward goal
    cost      = vf.potential(wx, wy)        # scalar phi in [0, 1]
    """

    # Stop commanding motion when this close to goal (metres).
    ARRIVAL_RADIUS_M: float = 0.15

    # Gradient magnitude threshold below which fall back to straight-to-goal.
    _GRAD_THRESHOLD: float = 1e-6

    def __init__(self) -> None:
        self._phi: np.ndarray | None = None
        self._vx: np.ndarray | None = None
        self._vy: np.ndarray | None = None
        self._cs: float = 1.0
        self._goal_world: tuple[float, float] | None = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def phi(self) -> np.ndarray | None:
        """Normalised potential phi in [0, 1], shape (rows, cols)."""
        return self._phi

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(
        self,
        grid_map: GridMap,
        goal_world: Tuple[float, float],
        repulsion_radius: float = 0.6,
        repulsion_alpha: float = 5.0,
    ) -> None:
        """
        Run the two-pass EDT + cost-weighted Dijkstra pipeline.

        Parameters
        ----------
        grid_map         : GridMap with boolean obstacle grid and cell_size.
        goal_world       : (wx, wy) goal position in metres.
        repulsion_radius : Obstacle influence radius in metres.
                           Cells closer than this to any obstacle pay
                           higher traversal cost.  Default 0.6 m.
        repulsion_alpha  : Extra cost factor at the obstacle boundary.
                           0 = pure shortest-path (no repulsion).
                           Default 5.0.
        """
        self._cs = grid_map.cell_size
        self._goal_world = goal_world
        rows, cols = grid_map.rows, grid_map.cols
        gx_g, gy_g = grid_map.goal

        # ── 1. EDT: distance to nearest obstacle ─────────────────────
        d_obs = self._edt(grid_map.grid)

        # ── 2. Per-cell traversal cost ────────────────────────────────
        cost = self._cost_field(d_obs, repulsion_radius, repulsion_alpha)

        # ── 3. Navigation Dijkstra from goal with weighted edges ──────
        nav = self._nav_dijkstra(grid_map.grid, cost, gy_g, gx_g)

        # ── 4. Normalise to phi in [0, 1] ─────────────────────────────
        finite = np.isfinite(nav)
        n_reachable = int(finite.sum())

        if n_reachable == 0:
            self._phi = np.ones((rows, cols))
            self._phi[gy_g, gx_g] = 0.0
            self._vx = np.zeros((rows, cols))
            self._vy = np.zeros((rows, cols))
            self._ready = True
            return

        t_max = float(nav[finite].max())
        if t_max < 1e-12:
            t_max = 1.0

        phi = nav.copy()
        phi[~finite] = t_max  # obstacle cells → plateau
        phi /= t_max
        self._phi = phi

        # ── 5. Vector field = -grad(phi) ─────────────────────────────
        grad = np.gradient(phi)
        self._vy = -grad[0]  # axis-0 is rows = y
        self._vx = -grad[1]  # axis-1 is cols = x
        self._vx[grid_map.grid] = 0.0
        self._vy[grid_map.grid] = 0.0

        self._ready = True

        # Report stats
        pct_repulsed = float((d_obs[~grid_map.grid] < repulsion_radius).mean()) * 100
        print(
            f"[VectorField] EDT + nav Dijkstra: {n_reachable} reachable cells, "
            f"T_max = {t_max:.2f},  "
            f"{pct_repulsed:.1f}% free cells inside repulsion zone "
            f"(r={repulsion_radius} m, alpha={repulsion_alpha})"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, wx: float, wy: float) -> np.ndarray:
        """Unit navigation vector at world position (wx, wy)."""
        if not self._ready:
            return np.zeros(2)

        if self._goal_world is not None:
            gx_w, gy_w = self._goal_world
            if (wx - gx_w) ** 2 + (wy - gy_w) ** 2 < self.ARRIVAL_RADIUS_M**2:
                return np.zeros(2)

        vx = self._bilinear(self._vx, wx, wy)
        vy = self._bilinear(self._vy, wx, wy)
        mag = math.sqrt(vx * vx + vy * vy)

        if mag >= self._GRAD_THRESHOLD:
            return np.array([vx / mag, vy / mag])

        # Fallback: straight toward goal
        if self._goal_world is None:
            return np.zeros(2)
        gx_w, gy_w = self._goal_world
        dx, dy = gx_w - wx, gy_w - wy
        d = math.sqrt(dx * dx + dy * dy)
        return np.zeros(2) if d < 1e-9 else np.array([dx / d, dy / d])

    def potential(self, wx: float, wy: float) -> float:
        """Normalised potential phi in [0, 1] at world position (wx, wy)."""
        if not self._ready:
            return 0.0
        return float(self._bilinear(self._phi, wx, wy))

    # ------------------------------------------------------------------
    # EDT — multi-source Dijkstra from all obstacle cells
    # ------------------------------------------------------------------

    def _edt(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute distance to nearest obstacle for every free cell.

        Seeds the priority queue with every obstacle cell at distance 0
        and expands outward using 8-connected Dijkstra.

        Returns an array of the same shape as grid:
            d_obs[gy, gx] = 0          for obstacle cells
            d_obs[gy, gx] = distance   for free cells (metres)
        """
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)
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

        heap = []
        # Seed: every obstacle cell at distance 0
        gy_obs, gx_obs = np.where(grid)
        for gy, gx in zip(gy_obs.tolist(), gx_obs.tolist()):
            dist[gy, gx] = 0.0
            heap.append((0.0, gy, gx))
        heapq.heapify(heap)

        while heap:
            d, gy, gx = heapq.heappop(heap)
            if d > dist[gy, gx]:
                continue
            for dy, dx, w in _MOVES:
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    nd = d + w
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, ny, nx))

        return dist

    # ------------------------------------------------------------------
    # Cost field
    # ------------------------------------------------------------------

    def _cost_field(
        self,
        d_obs: np.ndarray,
        radius: float,
        alpha: float,
    ) -> np.ndarray:
        """
        Per-cell traversal cost  c(x) = 1 + alpha * ramp(d_obs, radius)^2.

        ramp = max(0, (radius - d_obs) / radius)
             = 1 at d_obs = 0  (obstacle boundary)
             = 0 at d_obs >= radius

        The quadratic shape keeps the gradient of the cost field smooth,
        avoiding a kink at d_obs = radius.
        """
        if alpha == 0.0 or radius <= 0.0:
            return np.ones_like(d_obs)

        ramp = np.maximum(0.0, (radius - d_obs) / radius)
        return 1.0 + alpha * ramp**2

    # ------------------------------------------------------------------
    # Navigation Dijkstra with per-cell costs
    # ------------------------------------------------------------------

    def _nav_dijkstra(
        self,
        grid: np.ndarray,
        cost: np.ndarray,
        start_gy: int,
        start_gx: int,
    ) -> np.ndarray:
        """
        Weighted Dijkstra from goal cell.

        Edge weight between cells u and v:
            w(u, v) = raw_dist * 0.5 * (cost[u] + cost[v])

        Averaging the two endpoint costs gives a trapezoidal integration
        of the cost field along the edge — more accurate than using only
        the destination cell's cost.
        """
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)
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

        if grid[start_gy, start_gx]:
            return dist

        dist[start_gy, start_gx] = 0.0
        heap = [(0.0, start_gy, start_gx)]

        while heap:
            d, gy, gx = heapq.heappop(heap)
            if d > dist[gy, gx]:
                continue
            c_here = cost[gy, gx]
            for dy, dx, raw in _MOVES:
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < rows and 0 <= nx < cols and not grid[ny, nx]:
                    w = raw * 0.5 * (c_here + cost[ny, nx])
                    nd = d + w
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, ny, nx))

        return dist

    # ------------------------------------------------------------------
    # Bilinear interpolation (cell-centred)
    # ------------------------------------------------------------------

    def _bilinear(self, arr: np.ndarray, wx: float, wy: float) -> float:
        cs = self._cs
        rows, cols = arr.shape
        gxf = wx / cs - 0.5
        gx0 = int(math.floor(gxf))
        tx = gxf - gx0
        gyf = wy / cs - 0.5
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
