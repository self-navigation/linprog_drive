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
    N: int = 10  # prediction horizon (steps)
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
        from scipy.optimize import linprog

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
