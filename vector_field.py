"""
vector_field.py — Navigation potential via cost-weighted geodesic distances.

Pipeline (VectorField — Dijkstra-based)
----------------------------------------
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

FMMVectorField — Fast Marching Method (skfmm-based)
-----------------------------------------------------
Replaces the Dijkstra navigation pass with skfmm.travel_time(), which
solves the Eikonal equation exactly on the continuous domain.

Speed function: free cells travel at full speed (1.0); cells near
obstacles travel slower via an exponential decay:
    speed(d) = exp(-decay_rate * (1 - d/repulsion_radius))   d < radius
    speed(d) = 1.0                                             d >= radius

Obstacle cells are masked out so FMM skips them; they are then filled
with an EDT-based depth penalty so the gradient inside walls always
points toward the nearest free cell (recovery direction).

An optional wall-repulsion potential and Gaussian direction smoothing
(ported from the ROS2 FMMVectorFieldNode) are applied before the final
unit-vector normalisation.
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


# ---------------------------------------------------------------------------
# FMM-based vector field (ported from ROS2 FMMVectorFieldNode)
# ---------------------------------------------------------------------------


class FMMVectorField:
    """
    Navigation potential field computed via the Fast Marching Method.

    Implements the same interface as VectorField so the simulation can
    substitute it with a single line change.

    Usage
    -----
    vf = FMMVectorField()
    vf.compute(grid_map, goal_world)
    direction = vf.query(wx, wy)   # unit vector toward goal
    cost      = vf.potential(wx, wy)
    """

    ARRIVAL_RADIUS_M: float = 0.15
    _GRAD_THRESHOLD: float = 1e-6

    def __init__(self) -> None:
        self._phi: np.ndarray | None = None
        self._vx: np.ndarray | None = None
        self._vy: np.ndarray | None = None
        self._cs: float = 1.0
        self._goal_world: tuple[float, float] | None = None
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def phi(self) -> np.ndarray | None:
        return self._phi

    # ------------------------------------------------------------------
    # Public compute
    # ------------------------------------------------------------------

    def compute(
        self,
        grid_map: GridMap,
        goal_world: Tuple[float, float],
        repulsion_radius: float = 0.5,
        repulsion_alpha: float = 5.0,
        obstacle_slope_factor: float = 400.0,
        field_smooth_sigma: float = 2.5,
    ) -> None:
        """
        Run FMM on the occupancy grid and build the vector field.

        Parameters
        ----------
        grid_map              : GridMap with boolean obstacle grid and cell_size.
        goal_world            : (wx, wy) goal position in metres.
        repulsion_radius      : Radius around obstacles where speed is reduced (m).
        repulsion_alpha       : Exponential decay rate inside repulsion zone.
                                Higher values = steeper speed drop near walls.
        obstacle_slope_factor : Slope of the EDT fill inside obstacle cells
                                (penalty = factor * max_T * depth_m).
        field_smooth_sigma    : Gaussian blur sigma in cells applied to (vx, vy)
                                before re-normalisation. 0 disables smoothing.
        """
        try:
            import skfmm
        except ImportError as exc:
            raise ImportError(
                "FMMVectorField requires skfmm: pip install scikit-fmm"
            ) from exc
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        self._cs = grid_map.cell_size
        self._goal_world = goal_world
        cs = grid_map.cell_size
        obstacle_mask = grid_map.grid  # True = obstacle

        gx_g, gy_g = grid_map.goal

        if obstacle_mask[gy_g, gx_g]:
            print("[FMMVectorField] Goal cell is inside an obstacle — aborting.")
            return

        # ── Speed field from occupancy grid ───────────────────────────
        # Free cells near obstacles travel slower; obstacle cells are masked.
        free_mask = ~obstacle_mask
        if obstacle_mask.any() and free_mask.any():
            edt_free = distance_transform_edt(free_mask) * cs
        else:
            edt_free = np.full(obstacle_mask.shape, np.inf)

        speed = np.ones(obstacle_mask.shape, dtype=np.float64)
        if repulsion_radius > 0.0 and repulsion_alpha > 0.0:
            near = free_mask & (edt_free < repulsion_radius)
            norm_d = edt_free[near] / repulsion_radius  # 0 at wall, 1 at boundary
            speed[near] = np.clip(np.exp(-repulsion_alpha * (1.0 - norm_d)), 0.01, 1.0)
        speed[obstacle_mask] = 0.0  # will be masked out in FMM

        speed_ma = np.ma.MaskedArray(speed, mask=obstacle_mask)

        # ── FMM: solve Eikonal from goal ───────────────────────────────
        phi_init = np.ones(obstacle_mask.shape, dtype=np.float64)
        phi_init[gy_g, gx_g] = -1.0
        phi_ma = np.ma.MaskedArray(phi_init, mask=obstacle_mask)

        try:
            travel_time = skfmm.travel_time(phi_ma, speed_ma, dx=cs)
        except Exception as exc:
            print(f"[FMMVectorField] FMM failed: {exc}")
            return

        tt = np.array(travel_time, dtype=np.float64)
        if np.ma.is_masked(travel_time):
            tt[travel_time.mask] = np.nan

        # ── Fill obstacle cells with EDT-depth penalty ─────────────────
        obs_nan = np.isnan(tt)
        max_T = float(np.nanmax(tt)) if not np.all(np.isnan(tt)) else 1.0

        if obs_nan.any() and (~obs_nan).any():
            edt_obs = distance_transform_edt(obs_nan) * cs
            slope = obstacle_slope_factor * max_T
            tt[obs_nan] = max_T + slope * edt_obs[obs_nan]
        else:
            tt[obs_nan] = max_T * (1.0 + obstacle_slope_factor * cs)

        # ── Wall-repulsion potential ───────────────────────────────────
        # Adds a quadratic penalty near walls so the gradient escapes the
        # plateau created by the inflation zone.
        if repulsion_radius > 0.0 and obstacle_mask.any() and free_mask.any():
            edt_to_wall = distance_transform_edt(free_mask) * cs
            peak_penalty = 3.0 * max_T  # wall_repulsion_strength = 3.0
            near_wall = free_mask & (edt_to_wall < repulsion_radius)
            if near_wall.any():
                ratio = (repulsion_radius - edt_to_wall[near_wall]) / repulsion_radius
                tt[near_wall] += peak_penalty * ratio * ratio

        # ── Gradient → direction field ────────────────────────────────
        d_row, d_col = np.gradient(tt, cs)
        vx = -d_col  # col axis = x
        vy = -d_row  # row axis = y

        bad = np.isnan(vx) | np.isnan(vy) | np.isinf(vx) | np.isinf(vy)
        vx[bad] = 0.0
        vy[bad] = 0.0

        # ── Gaussian smoothing before normalisation ────────────────────
        if field_smooth_sigma > 0.0:
            vx = gaussian_filter(vx, sigma=field_smooth_sigma)
            vy = gaussian_filter(vy, sigma=field_smooth_sigma)

        mag = np.sqrt(vx ** 2 + vy ** 2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        vx /= safe_mag
        vy /= safe_mag

        # Zero out directions inside obstacle cells
        vx[obstacle_mask] = 0.0
        vy[obstacle_mask] = 0.0

        # ── Normalise travel time → phi in [0, 1] ─────────────────────
        free_tt = tt.copy()
        free_tt[obstacle_mask] = np.nan
        finite_max = float(np.nanmax(free_tt)) if not np.all(np.isnan(free_tt)) else 1.0
        if finite_max < 1e-12:
            finite_max = 1.0
        phi = np.clip(free_tt / finite_max, 0.0, 1.0)
        phi[obstacle_mask] = 1.0  # obstacles at max potential

        self._phi = phi
        self._vx = vx
        self._vy = vy
        self._ready = True

        n_reachable = int(np.isfinite(free_tt).sum())
        print(
            f"[FMMVectorField] FMM solve: {n_reachable} reachable cells, "
            f"max_T={max_T:.2f}, repulsion r={repulsion_radius}m α={repulsion_alpha}"
        )

    # ------------------------------------------------------------------
    # Query — same interface as VectorField
    # ------------------------------------------------------------------

    def query(self, wx: float, wy: float) -> np.ndarray:
        if not self._ready:
            return np.zeros(2)
        if self._goal_world is not None:
            gx_w, gy_w = self._goal_world
            if (wx - gx_w) ** 2 + (wy - gy_w) ** 2 < self.ARRIVAL_RADIUS_M ** 2:
                return np.zeros(2)

        vx = self._bilinear(self._vx, wx, wy)
        vy = self._bilinear(self._vy, wx, wy)
        mag = math.sqrt(vx * vx + vy * vy)

        if mag >= self._GRAD_THRESHOLD:
            return np.array([vx / mag, vy / mag])

        if self._goal_world is None:
            return np.zeros(2)
        gx_w, gy_w = self._goal_world
        dx, dy = gx_w - wx, gy_w - wy
        d = math.sqrt(dx * dx + dy * dy)
        return np.zeros(2) if d < 1e-9 else np.array([dx / d, dy / d])

    def potential(self, wx: float, wy: float) -> float:
        if not self._ready:
            return 0.0
        return float(self._bilinear(self._phi, wx, wy))

    # ------------------------------------------------------------------
    # Bilinear interpolation (cell-centred) — same as VectorField
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
