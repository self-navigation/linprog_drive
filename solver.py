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
from typing import Tuple

import numpy as np

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
        grid_map: GridMap,
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
        grid_map: GridMap,
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
        grid_map: GridMap,
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


# ---------------------------------------------------------------------------
# LP OCP solver  (Kühne, Lages, Gomes da Silva Jr., MechRob 2004 — §II–III)
# ---------------------------------------------------------------------------


class LPSolver(BaseSolver):
    """
    Receding-horizon LP-based OCP for a unicycle robot.

    Formulation
    -----------
    Follows Kühne et al. (2004) with two adaptations:

    1. Reference is the vector field direction, not a pre-planned trajectory.
       At each step the "reference car" drives at V_REF in the field's current
       direction from the robot's current position.

    2. L1 cost (absolute value penalties) rather than quadratic, keeping the
       problem a pure LP solvable with scipy.optimize.linprog.

    Error model
    -----------
    Let x̃ = x − x_r (state error w.r.t. reference car).
    Linearising the unicycle around (x_r, u_r) gives (Kühne eq. 7):

        x̃(k+1) = A · x̃(k) + B · ũ(k)

    with:
        A = [[1,  0,  −v_r · sin(θ_r) · T],     B = [[cos(θ_r) · T,  0],
             [0,  1,   v_r · cos(θ_r) · T],           [sin(θ_r) · T,  0],
             [0,  0,   1               ]]               [0,             T]]

    where T = DT (planning timestep), v_r = V_REF, θ_r = field direction.

    A and B are evaluated once at the current robot position and held
    constant for all N horizon steps — valid for short horizons where
    the heading change is small.

    Initial error
    -------------
    x̃_0 = [0, 0, wrap(θ − θ_r)]
    Position error is zero because we linearise at the current position.
    Heading error is the only non-zero initial component.

    Decision variables  (10·N total)
    ---------------------------------
    z[0   : 2N]   stacked ũ_k = [δv_k, δω_k]     k = 0..N−1
    z[2N  : 5N]   stacked x̃_{k+1}                k = 0..N−1
    z[5N  : 8N]   L1 auxiliaries t_k ≥ |x̃_{k+1}| k = 0..N−1
    z[8N  : 10N]  L1 auxiliaries s_k ≥ |ũ_k|      k = 0..N−1

    LP structure
    ------------
    Objective:   min  Σ_k [ Q·t_k + R·s_k ]         (linear in z)
    Equality:    x̃_{k+1} − A·x̃_k − B·ũ_k = A·x̃_0  (k=0), 0  (k>0)
    Inequality:  ±x̃_{k+1} − t_k ≤ 0               (L1 epigraph, state)
                 ±ũ_k − s_k ≤ 0                    (L1 epigraph, input)
    Bounds:      δv_k ∈ [v_min−v_r, v_max−v_r]
                 δω_k ∈ [−ω_max, ω_max]
                 t_k, s_k ≥ 0
                 x̃_{k+1} unbounded

    Solver: scipy.optimize.linprog with HiGHS backend.
    Falls back to GradientSolver on infeasibility or solver failure.
    """

    NAME = "lp_mpc"

    # Horizon and timing
    N: int = 5  # prediction horizon (steps)
    DT: float = 0.10  # planning timestep (s) — independent of simulation fps

    # Reference speed (the "reference car" forward speed)
    V_REF: float = 0.8  # m/s — slightly below V_MAX for headroom

    # Input bounds
    V_MAX: float = 1.0  # m/s
    V_MIN: float = -0.3  # m/s  (allow limited reverse)
    OMEGA_MAX: float = 1.5  # rad/s

    # L1 cost weights
    # Q weights state error [x, y, θ] — θ is most important for a field follower
    Q: np.ndarray = np.array([0.5, 0.5, 5.0])
    # R weights input perturbation [δv, δω] — regularise away from large inputs
    R: np.ndarray = np.array([0.2, 0.1])

    def __init__(self) -> None:
        self._fallback = GradientSolver()
        self._last_status: str = "—"

    @property
    def last_status(self) -> str:
        """Solver status string for HUD display."""
        return self._last_status

    def solve(
        self,
        state: np.ndarray,
        vector_field,
        grid_map: GridMap,
    ) -> Tuple[float, float]:

        x0, y0, theta0 = float(state[0]), float(state[1]), float(state[2])

        # ── Reference from vector field ───────────────────────────────
        direction = vector_field.query(x0, y0)
        if direction[0] == 0.0 and direction[1] == 0.0:
            self._last_status = "arrived"
            return 0.0, 0.0

        theta_r = math.atan2(float(direction[1]), float(direction[0]))
        v_r = self.V_REF

        # ── Initial heading error  (wrapped to [−π, π]) ───────────────
        dtheta = theta0 - theta_r
        dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
        # Position error is 0: we linearise at the current robot position.
        x_tilde_0 = np.array([0.0, 0.0, dtheta])

        # ── Linearisation matrices (constant over horizon) ────────────
        T = self.DT
        ct = math.cos(theta_r)
        st = math.sin(theta_r)

        A = np.array(
            [
                [1.0, 0.0, -v_r * st * T],
                [0.0, 1.0, v_r * ct * T],
                [0.0, 0.0, 1.0],
            ]
        )
        B = np.array(
            [
                [ct * T, 0.0],
                [st * T, 0.0],
                [0.0, T],
            ]
        )

        # ── Build the LP ──────────────────────────────────────────────
        N = self.N
        nz = 10 * N  # total decision variables

        # Index helpers
        def ui(k, i):
            return 2 * k + i  # ũ_k[i]     k∈[0,N), i∈{0,1}

        def xi(k, i):
            return 2 * N + 3 * k + i  # x̃_{k+1}[i] k∈[0,N), i∈{0,1,2}

        def ti(k, i):
            return 5 * N + 3 * k + i  # t_k[i]     k∈[0,N), i∈{0,1,2}

        def si(k, i):
            return 8 * N + 2 * k + i  # s_k[i]     k∈[0,N), i∈{0,1}

        # ── Cost vector ───────────────────────────────────────────────
        c = np.zeros(nz)
        for k in range(N):
            for i in range(3):
                c[ti(k, i)] = self.Q[i]
            for i in range(2):
                c[si(k, i)] = self.R[i]

        # ── Equality constraints: dynamics ────────────────────────────
        # 3·N equations: x̃_{k+1} − A·x̃_k − B·ũ_k = rhs_k
        # rhs_0 = A·x̃_0  (x̃_0 known); rhs_{k>0} = 0
        n_eq = 3 * N
        Aeq = np.zeros((n_eq, nz))
        beq = np.zeros(n_eq)

        for k in range(N):
            row = 3 * k

            # +I on x̃_{k+1}
            for i in range(3):
                Aeq[row + i, xi(k, i)] = 1.0

            # −B on ũ_k
            for i in range(3):
                for j in range(2):
                    Aeq[row + i, ui(k, j)] -= B[i, j]

            # −A on x̃_k  (or move A·x̃_0 to RHS when k=0)
            if k == 0:
                beq[row : row + 3] = A @ x_tilde_0
            else:
                for i in range(3):
                    for j in range(3):
                        Aeq[row + i, xi(k - 1, j)] -= A[i, j]

        # ── Inequality constraints: L1 epigraph ───────────────────────
        # 10·N rows:
        #   state  (6 per step): ±x̃_{k+1} − t_k ≤ 0
        #   input  (4 per step): ±ũ_k     − s_k ≤ 0
        n_ub = 10 * N
        Aub = np.zeros((n_ub, nz))
        bub = np.zeros(n_ub)

        row = 0
        for k in range(N):
            for i in range(3):  # x̃_{k+1} − t_k ≤ 0
                Aub[row, xi(k, i)] = 1.0
                Aub[row, ti(k, i)] = -1.0
                row += 1
            for i in range(3):  # −x̃_{k+1} − t_k ≤ 0
                Aub[row, xi(k, i)] = -1.0
                Aub[row, ti(k, i)] = -1.0
                row += 1
            for i in range(2):  # ũ_k − s_k ≤ 0
                Aub[row, ui(k, i)] = 1.0
                Aub[row, si(k, i)] = -1.0
                row += 1
            for i in range(2):  # −ũ_k − s_k ≤ 0
                Aub[row, ui(k, i)] = -1.0
                Aub[row, si(k, i)] = -1.0
                row += 1

        # ── Variable bounds ───────────────────────────────────────────
        bounds = []
        for k in range(N):
            bounds.append((self.V_MIN - v_r, self.V_MAX - v_r))  # δv_k
            bounds.append((-self.OMEGA_MAX, self.OMEGA_MAX))  # δω_k
        for k in range(N):
            for _ in range(3):
                bounds.append((None, None))  # x̃_{k+1}  free
        for k in range(N):
            for _ in range(3):
                bounds.append((0.0, None))  # t_k ≥ 0
        for k in range(N):
            for _ in range(2):
                bounds.append((0.0, None))  # s_k ≥ 0

        # ── Solve ─────────────────────────────────────────────────────
        from simplex import linprog

        result = linprog(
            c,
            A_ub=Aub,
            b_ub=bub,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="highs",
        )

        if result.status != 0:
            # 0=optimal, anything else = infeasible/unbounded/numerical issue
            self._last_status = f"fail({result.status})"
            return self._fallback.solve(state, vector_field, grid_map)

        self._last_status = f"ok  obj={result.fun:.3f}"

        # ── Extract first control action ──────────────────────────────
        dv = float(result.x[ui(0, 0)])
        domeg = float(result.x[ui(0, 1)])

        v = v_r + dv
        omega = domeg  # ω_r = 0

        return v, omega


# ---------------------------------------------------------------------------
# Chebyshev NLP solver
# ---------------------------------------------------------------------------


class ChebSolver(BaseSolver):
    """
    Receding-horizon NLP solver using Chebyshev control parameterisation.

    Instead of N piecewise-constant control actions (as in LPSolver), the
    control inputs v(t) and omega(t) over [0, T_HORIZON] are each represented
    as a degree-M Chebyshev polynomial:

        v(t)     = sum_{k=0}^{M}  c_v[k]  * T_k(tau(t))
        omega(t) = sum_{k=0}^{M}  c_om[k] * T_k(tau(t))

    where tau(t) = 1 - 2t/T_HORIZON maps [0, T_HORIZON] -> [+1, -1] so
    that tau=+1 is the current instant and tau=-1 is the horizon end.

    Decision variables: z in R^{2*(M+1)} = [c_v | c_om]  (10 floats for M=4)

    The OCP is solved by Nelder-Mead each call (single phase, no Newton-CG).
    Newton-CG was removed because its internal Hessian-vector finite differences
    (approx_fhess_p) called _cost ~3800 times per solve vs ~600 for NM alone,
    and the precision gain is irrelevant for a receding-horizon controller whose
    result is immediately discarded after one timestep anyway.

    Performance key: the vector field is queried exactly M+1 times per solve call
    (once along the warm-start trajectory before NM runs) and the cached directions
    are passed to _cost.  _cost itself is pure numpy with no further queries.

    The Chebyshev basis addresses the Runge effect that would arise from fitting a
    high-degree polynomial on a uniform grid: Chebyshev-Gauss-Lobatto (CGL) nodes
    are cosine-spaced, denser near the endpoints, and the evaluation matrix is a
    DCT-I -- well-conditioned and fast even without FFT at small M.

    Input constraints (v, omega bounds) are enforced as a soft quadratic penalty
    because Nelder-Mead does not support explicit bound constraints.  LAM_BOUND is
    set high enough that violations are negligible at convergence.

    Falls back to GradientSolver on any optimisation failure.
    """

    NAME = "cheb_nlp"

    M: int = 4  # Chebyshev degree; M+1 nodes, 2*(M+1) decision vars
    T_HORIZON: float = 1.0  # planning window length (s)

    V_REF: float = 0.8  # nominal forward speed for warm-start and cost
    V_MAX: float = 1.0
    V_MIN: float = -0.3
    OMEGA_MAX: float = 1.5

    # Cost weights
    Q_THETA: float = 5.0  # heading-error penalty (dominant term)
    Q_POS: float = 0.3  # position-drift penalty (keeps robot on field path)
    R_V: float = 0.5  # deviation from V_REF
    R_OMEGA: float = 0.2  # omega magnitude

    # Soft bound penalty -- large enough that bound violations are negligible
    LAM_BOUND: float = 20.0

    # Optimiser budget -- warm starts mean we rarely need the full count
    NM_MAXITER: int = 150
    NM_XATOL: float = 1e-3

    def __init__(self) -> None:
        self._fallback = GradientSolver()
        self._z_warm: np.ndarray | None = None
        self._last_theta_r: float = 0.0
        self._last_status: str = "-"

        # Build the Chebyshev basis once -- it never changes
        self._nodes, self._times, self._dt, self._C = self._build_basis()

    # ------------------------------------------------------------------
    # Basis construction (called once in __init__)
    # ------------------------------------------------------------------

    def _build_basis(self):
        """
        Precompute CGL nodes, physical times, time deltas, and the
        Chebyshev evaluation matrix C where C[j, k] = T_k(tau_j).

        CGL nodes: tau_j = cos(pi*j/M)  for j = 0, ..., M
          j=0 -> tau=+1 -> t=0   (current instant)
          j=M -> tau=-1 -> t=T   (horizon end)

        The matrix C is the DCT-I matrix (unnormalised).  Multiplying C by
        a coefficient vector c gives the polynomial values at all nodes
        simultaneously -- this is the "Chebyshev matrix" the teacher mentioned.
        """
        M = self.M
        j = np.arange(M + 1, dtype=float)
        nodes = np.cos(np.pi * j / M)  # tau_j in [-1,+1]
        times = self.T_HORIZON / 2.0 * (1.0 - nodes)  # t_j in [0, T]

        dt = np.diff(times)  # M uneven intervals

        # C[j, k] = cos(k * pi * j / M) = T_k(tau_j)
        k = np.arange(M + 1, dtype=float)
        C = np.cos(np.outer(j, k) * np.pi / M)  # (M+1) x (M+1)

        return nodes, times, dt, C

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode(self, z: np.ndarray):
        """
        Split z into Chebyshev coefficients and evaluate at all nodes.
        Returns v_vals and omega_vals, shape (M+1,), NOT yet clipped.
        Clipping is done inside _cost so the penalty term sees the violation.
        """
        M = self.M
        c_v = z[: M + 1]
        c_om = z[M + 1 :]
        v_vals = self._C @ c_v
        om_vals = self._C @ c_om
        return v_vals, om_vals

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _simulate(
        self,
        state: np.ndarray,
        v_vals: np.ndarray,
        om_vals: np.ndarray,
    ) -> np.ndarray:
        """
        Euler-integrate the unicycle forward along the M uneven time intervals.
        Returns trajectory of shape (M+1, 3): one row per node including x0.
        """
        traj = np.empty((self.M + 1, 3))
        traj[0] = state
        for j in range(self.M):
            x, y, th = traj[j]
            v = float(np.clip(v_vals[j], self.V_MIN, self.V_MAX))
            om = float(np.clip(om_vals[j], -self.OMEGA_MAX, self.OMEGA_MAX))
            dt = self._dt[j]
            traj[j + 1, 0] = x + v * math.cos(th) * dt
            traj[j + 1, 1] = y + v * math.sin(th) * dt
            traj[j + 1, 2] = th + om * dt
        return traj

    def _bound_penalty(self, v_vals: np.ndarray, om_vals: np.ndarray) -> float:
        """
        Quadratic soft penalty for bound violations -- Nelder-Mead has no
        native bound support so we fold constraints into the objective.
        """
        rv = np.maximum(0.0, v_vals - self.V_MAX) ** 2
        rv += np.maximum(0.0, self.V_MIN - v_vals) ** 2
        rom = np.maximum(0.0, np.abs(om_vals) - self.OMEGA_MAX) ** 2
        return float(self.LAM_BOUND * (rv.sum() + rom.sum()))

    def _cost(
        self,
        z: np.ndarray,
        state: np.ndarray,
        theta_refs: np.ndarray,  # pre-queried field directions, shape (M+1,)
    ) -> float:
        """
        theta_refs is queried ONCE per solve call along the warm-start trajectory
        and passed in here so that no further field lookups happen during NM.
        """
        v_vals, om_vals = self._decode(z)
        traj = self._simulate(state, v_vals, om_vals)

        x0, y0 = state[0], state[1]

        # Vectorised heading error over nodes 1..M
        th = traj[1:, 2]
        th_refs = theta_refs[1:]
        err = th - th_refs
        err = (err + math.pi) % (2.0 * math.pi) - math.pi
        heading_cost = float(np.dot(self.Q_THETA * err**2, self._dt))

        # Position drift cost
        dx = traj[1:, 0] - x0
        dy = traj[1:, 1] - y0
        pos_cost = float(np.dot(self.Q_POS * (dx**2 + dy**2), self._dt))

        # Input regularisation over all nodes
        v_dev = v_vals - self.V_REF
        reg_cost = self.R_V * float(np.dot(v_dev, v_dev)) + self.R_OMEGA * float(
            np.dot(om_vals, om_vals)
        )

        return heading_cost + pos_cost + reg_cost + self._bound_penalty(v_vals, om_vals)

    # ------------------------------------------------------------------
    # Warm-start management
    # ------------------------------------------------------------------

    def _default_z(self) -> np.ndarray:
        """
        Constant-speed, zero-omega trajectory as the default warm start.
        The constant polynomial has only a c0 coefficient; all higher terms
        are zero, giving v(t) = V_REF and omega(t) = 0 everywhere.
        """
        z = np.zeros(2 * (self.M + 1))
        # c0 term: T_0(tau) = 1, so f(t) = c0 for all t
        # But C[j,0] = 1 for all j, so to get v_vals = V_REF we need c0 = V_REF
        # only when the other coefficients are 0 (which they are here).
        z[0] = self.V_REF
        return z

    def _should_reset_warm_start(self, theta_r: float) -> bool:
        """Reset if the field direction has rotated sharply since last call."""
        delta = abs(self._wrap(theta_r - self._last_theta_r))
        return delta > math.radians(40)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        state: np.ndarray,
        vector_field,
        grid_map: GridMap,
    ) -> Tuple[float, float]:
        from scipy.optimize import minimize

        x0, y0, theta0 = float(state[0]), float(state[1]), float(state[2])

        direction = vector_field.query(x0, y0)
        if direction[0] == 0.0 and direction[1] == 0.0:
            self._last_status = "arrived"
            return 0.0, 0.0

        theta_r = math.atan2(float(direction[1]), float(direction[0]))

        # Choose warm start
        if self._z_warm is None or self._should_reset_warm_start(theta_r):
            z0 = self._default_z()
        else:
            z0 = self._z_warm.copy()
        self._last_theta_r = theta_r

        # -- Pre-cache field directions (M+1 queries total, done ONCE) ---------
        # Simulate the warm-start trajectory and query the field at each node.
        # These cached directions are reused for every _cost call inside NM,
        # reducing field queries from ~O(NM_MAXITER * M) down to M+1 per solve.
        #
        # Small approximation: during NM the actual trajectory shifts slightly
        # from the warm-start path, so the cached directions are not perfectly
        # aligned with the candidate positions.  For a smooth field and a
        # receding-horizon controller with a good warm start, the error is small
        # and inconsequential for the first control action.
        v0_vals, om0_vals = self._decode(z0)
        ref_traj = self._simulate(state, v0_vals, om0_vals)
        theta_refs = np.empty(self.M + 1)
        theta_refs[0] = theta_r  # already queried above
        for j in range(1, self.M + 1):
            d = vector_field.query(float(ref_traj[j, 0]), float(ref_traj[j, 1]))
            if d[0] != 0.0 or d[1] != 0.0:
                theta_refs[j] = math.atan2(float(d[1]), float(d[0]))
            else:
                theta_refs[j] = theta_refs[j - 1]  # hold last direction at goal

        cost_fn = lambda z: self._cost(z, state, theta_refs)  # noqa: E731

        res = minimize(
            cost_fn,
            z0,
            method="Nelder-Mead",
            options={
                "maxiter": self.NM_MAXITER,
                "xatol": self.NM_XATOL,
                "fatol": self.NM_XATOL,
            },
        )

        if not res.success and res.fun > 1e6:
            self._last_status = f"nm_fail"
            return self._fallback.solve(state, vector_field, grid_map)

        self._z_warm = res.x
        self._last_status = f"ok  J={res.fun:.3f}"

        # Extract action at t=0 (tau=+1, node j=0): just read off decoded node 0
        v_vals, om_vals = self._decode(res.x)
        v = float(np.clip(v_vals[0], self.V_MIN, self.V_MAX))
        omega = float(np.clip(om_vals[0], -self.OMEGA_MAX, self.OMEGA_MAX))

        return v, omega
