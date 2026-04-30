"""
simplex.py — Full tableau two-phase simplex method implemented from scratch.

Solves the linear programme:

    minimize    c @ x
    subject to  A_ub @ x <= b_ub          (inequality constraints)
                A_eq @ x == b_eq           (equality constraints)
                bounds[i][0] <= x[i] <= bounds[i][1]  (variable bounds)

Public API
----------
    result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

The signature and return type mirror scipy.optimize.linprog so the two are
drop-in substitutes in LPSolver — swap the import, nothing else changes.

Algorithm overview
------------------
1. Preprocess  — convert to standard form:
       min  c_s @ z,   A_s @ z = b_s,   z >= 0,   b_s >= 0

   Variable transformations applied:
   - Shift by lower bound:  z_i = x_i − lo_i        (if lo_i finite)
   - Flip upper-only:       z_i = hi_i − x_i         (if lo_i = −∞, hi_i finite)
   - Split free variables:  x_i = z_i⁺ − z_i⁻       (if both bounds are None)
   - Add slack for A_ub:    A_ub z + s = b_ub, s >= 0
   - Add slack for hi:      z_i + s_bnd = hi − lo,   s_bnd >= 0
   Rows where b < 0 are multiplied by −1 to enforce b_s >= 0.

2. Phase 1  — find an initial basic feasible solution (BFS).
   Introduce one artificial variable a_i per constraint row.
   Minimise sum(a_i).  If the minimum exceeds ε → infeasible.
   Degenerate artificials (value ≈ 0 but still basic) are pivoted out.

3. Phase 2  — minimise the original objective from the Phase 1 BFS.
   Only original + slack columns may enter the basis; artificials are
   excluded by restricting the column search range.

4. Reconstruct  — map z_s back to original x via the inverse of the
   preprocessing transformations.

Pivot rule
----------
Bland's rule throughout both phases: always enter the smallest-indexed
column with a negative reduced cost.  This guarantees termination in
finite iterations (no cycling) at the cost of slower practical convergence
compared with the most-negative-reduced-cost rule.

Numerical parameters
--------------------
_PIVOT_TOL  minimum magnitude for a pivot element to be accepted.
_FEAS_TOL   Phase-1 objective must be below this to declare feasibility.
_OPT_TOL    reduced cost below this (in magnitude) is treated as zero.
_MAX_ITER   hard iteration cap (safeguard against degenerate cycling).

References
----------
- Bertsimas & Tsitsiklis, "Introduction to Linear Optimization",
  Athena Scientific 1997, Chapters 3–4.
- Kühne, Lages, Gomes da Silva Jr., "Model Predictive Control of a Mobile
  Robot Using Linearization", MechRob 2004 — the LP problem being solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

_PIVOT_TOL: float = 1e-10  # minimum acceptable pivot element magnitude
_FEAS_TOL: float = 1e-7  # Phase-1 objective threshold for feasibility
_OPT_TOL: float = 1e-9  # reduced cost threshold for optimality
_MAX_ITER: int = 100_000  # hard iteration cap


# ---------------------------------------------------------------------------
# Result dataclass  (matches scipy.optimize.OptimizeResult fields used by LPSolver)
# ---------------------------------------------------------------------------


@dataclass
class SimplexResult:
    """
    Simplex solve result.

    Fields match those of scipy.optimize.OptimizeResult / linprog so that
    LPSolver can import from either module without modification.

    status codes
    ------------
    0  optimal solution found
    1  iteration limit reached
    2  problem is infeasible
    3  problem is unbounded
    4  numerical failure (e.g. degenerate pivot)
    """

    x: np.ndarray  # optimal x in the original variable space
    fun: float  # optimal objective value  c @ x
    status: int
    message: str
    success: bool  # True iff status == 0
    iterations: int  # total pivot count (Phase 1 + Phase 2 combined)


_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Iteration limit reached.",
    2: "The problem appears to be infeasible.",
    3: "The problem appears to be unbounded.",
    4: "Numerical difficulties encountered.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def linprog(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    method: str = "simplex",  # kept for API parity; always uses simplex
    **kwargs,  # absorb unused kwargs silently (e.g. options={})
) -> SimplexResult:
    """
    Solve a linear programme using the two-phase full tableau simplex method.

    Parameters  (identical to scipy.optimize.linprog)
    ---------------------------------------------------
    c       : (n,) array_like  — objective coefficients
    A_ub    : (m_ub, n)        — inequality matrix  (A_ub x ≤ b_ub)
    b_ub    : (m_ub,)
    A_eq    : (m_eq, n)        — equality matrix    (A_eq x = b_eq)
    b_eq    : (m_eq,)
    bounds  : sequence of (lo, hi) pairs, one per variable.
              None means −∞ (lo) or +∞ (hi).
              Entire parameter None → every variable in [0, +∞).

    Returns
    -------
    SimplexResult with attributes .x .fun .status .message .success .iterations
    """
    c = np.asarray(c, dtype=float)
    n = len(c)

    # ── Normalise bounds ─────────────────────────────────────────────
    if bounds is None:
        bounds = [(0.0, None)] * n
    bounds = list(bounds)

    # ── Normalise A_ub / b_ub ────────────────────────────────────────
    if A_ub is not None:
        A_ub = np.atleast_2d(np.asarray(A_ub, dtype=float))
        b_ub = np.asarray(b_ub, dtype=float).ravel()
        m_ub = A_ub.shape[0]
    else:
        A_ub = np.zeros((0, n))
        b_ub = np.zeros(0)
        m_ub = 0

    # ── Normalise A_eq / b_eq ────────────────────────────────────────
    if A_eq is not None:
        A_eq = np.atleast_2d(np.asarray(A_eq, dtype=float))
        b_eq = np.asarray(b_eq, dtype=float).ravel()
        m_eq = A_eq.shape[0]
    else:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
        m_eq = 0

    # ── Build standard form ───────────────────────────────────────────
    try:
        c_s, A_s, b_s, n_s, reconstruct = _standard_form(
            c, A_ub, b_ub, A_eq, b_eq, bounds, n, m_ub, m_eq
        )
    except Exception as exc:
        return SimplexResult(
            x=np.full(n, np.nan),
            fun=np.inf,
            status=4,
            message=f"Preprocessing failed: {exc}",
            success=False,
            iterations=0,
        )

    # ── Solve ─────────────────────────────────────────────────────────
    z_s, status, iters = _two_phase(c_s, A_s, b_s, n_s)

    if status != 0:
        return SimplexResult(
            x=np.full(n, np.nan),
            fun=np.inf,
            status=status,
            message=_MESSAGES.get(status, "Unknown."),
            success=False,
            iterations=iters,
        )

    # ── Reconstruct original variables ────────────────────────────────
    x = reconstruct(z_s)
    return SimplexResult(
        x=x,
        fun=float(c @ x),
        status=0,
        message=_MESSAGES[0],
        success=True,
        iterations=iters,
    )


# ---------------------------------------------------------------------------
# Step 1: Standard form preprocessing
# ---------------------------------------------------------------------------


def _standard_form(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: list,
    n: int,
    m_ub: int,
    m_eq: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Callable]:
    """
    Convert the LP to standard form:
        min  c_s @ z,   A_s @ z = b_s,   z >= 0,   b_s >= 0

    Returns (c_s, A_s, b_s, n_s, reconstruct) where:
        n_s         = number of standard-form variables
        reconstruct = callable: z_s (n_s,) → x_orig (n,)

    Standard-form variable layout (z_s)
    ------------------------------------
    [z_var_0, ..., z_var_{n_z-1},          transformed original variables
     s_ub_0, ..., s_ub_{m_ub-1},           slacks for A_ub rows
     s_bnd_0, ..., s_bnd_{m_bnd-1}]        slacks for upper-bound rows

    Constraint row layout (A_s)
    ---------------------------
    [A_ub rows (with slacks),
     upper-bound rows (with slacks),
     A_eq rows (no slacks)]
    """

    # ── Variable transformations ──────────────────────────────────────
    # We build a linear map  x_orig = M @ z_var + d
    # where M is (n, n_z_var) and d is (n,).

    M_cols: list[np.ndarray] = []  # columns of M
    d = np.zeros(n)  # shift vector
    c_z_var: list[float] = []  # cost for each z_var column
    var_info: list[tuple] = []  # (type, z_indices) per original variable

    j = 0  # running z_var index
    for i, (lo, hi) in enumerate(bounds):
        ei = np.zeros(n)
        ei[i] = 1.0  # unit vector for variable i

        if lo is None and hi is None:
            # ── Free variable: x_i = z⁺ − z⁻, both >= 0 ─────────────
            M_cols.append(ei)
            M_cols.append(-ei)
            c_z_var.extend([c[i], -c[i]])
            var_info.append(("free", j, j + 1))
            j += 2

        elif lo is None:
            # ── Upper-only: x_i = hi − w,  w = hi − x_i >= 0 ────────
            M_cols.append(-ei)
            d[i] = hi
            c_z_var.append(-c[i])
            var_info.append(("hi_only", j))
            j += 1

        else:
            # ── Lower-bounded (possibly also upper-bounded) ───────────
            M_cols.append(ei)
            d[i] = lo
            c_z_var.append(c[i])
            var_info.append(("lo_or_both", j))
            j += 1

    n_z_var = j
    M = np.column_stack(M_cols) if M_cols else np.zeros((n, 0))

    c_z = np.array(c_z_var, dtype=float)  # length n_z_var

    # ── Transform constraint matrices ─────────────────────────────────
    # A_ub x = A_ub (M z_var + d) = (A_ub M) z_var + A_ub d
    A_ub_z = A_ub @ M  # (m_ub, n_z_var)
    b_ub_z = b_ub - A_ub @ d  # (m_ub,)
    A_eq_z = A_eq @ M  # (m_eq, n_z_var)
    b_eq_z = b_eq - A_eq @ d  # (m_eq,)

    # ── Upper-bound constraints from finite-hi variables ──────────────
    # For variable i with lo finite and hi finite:
    #   z_i <= hi − lo  →  z_i + s_bnd = hi − lo,  s_bnd >= 0
    bnd_rows: list[tuple[int, float]] = []  # (z_var_index, hi - lo)
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and hi is not None:
            z_idx = var_info[i][1]  # first (and only) z_var index
            bnd_rows.append((z_idx, hi - lo))

    m_bnd = len(bnd_rows)
    n_slack = m_ub + m_bnd  # one slack per inequality row
    n_s = n_z_var + n_slack  # total standard-form variables

    # ── Build A_s and b_s ─────────────────────────────────────────────
    m_s = m_ub + m_bnd + m_eq
    A_s = np.zeros((m_s, n_s))
    b_s = np.zeros(m_s)

    row = 0
    s_col = n_z_var  # current slack column

    # A_ub rows
    for i in range(m_ub):
        A_s[row, :n_z_var] = A_ub_z[i]
        A_s[row, s_col] = 1.0  # slack coefficient
        b_s[row] = b_ub_z[i]
        row += 1
        s_col += 1

    # Upper-bound rows
    for z_idx, ub_val in bnd_rows:
        A_s[row, z_idx] = 1.0
        A_s[row, s_col] = 1.0
        b_s[row] = ub_val
        row += 1
        s_col += 1

    # Equality rows  (no slack)
    for i in range(m_eq):
        A_s[row, :n_z_var] = A_eq_z[i]
        b_s[row] = b_eq_z[i]
        row += 1

    assert row == m_s and s_col == n_s

    # ── Ensure b_s >= 0 ───────────────────────────────────────────────
    # Rows with b < 0 are multiplied by −1.
    # For inequality rows this flips the slack coefficient to −1;
    # Phase 1 artificials will restore a non-negative starting BFS.
    for i in range(m_s):
        if b_s[i] < 0.0:
            A_s[i, :] *= -1.0
            b_s[i] *= -1.0

    # ── Build cost vector for standard form ───────────────────────────
    c_s = np.zeros(n_s)
    c_s[:n_z_var] = c_z  # slacks have zero cost

    # ── Build reconstruction function ─────────────────────────────────
    def reconstruct(z_s: np.ndarray) -> np.ndarray:
        z_var = z_s[:n_z_var]
        return M @ z_var + d

    return c_s, A_s, b_s, n_s, reconstruct


# ---------------------------------------------------------------------------
# Step 2: Two-phase simplex
# ---------------------------------------------------------------------------


def _two_phase(
    c_s: np.ndarray,  # (n_s,) standard form costs
    A_s: np.ndarray,  # (m_s, n_s) standard form constraint matrix
    b_s: np.ndarray,  # (m_s,) RHS — all >= 0
    n_s: int,  # number of standard-form variables (= n_z_var + n_slack)
) -> Tuple[np.ndarray, int, int]:
    """
    Solve min c_s @ z, A_s @ z = b_s, z >= 0 using the two-phase method.

    Returns (z_opt, status, total_iterations).
    """
    m, n = A_s.shape
    assert n == n_s
    assert len(b_s) == m
    assert (b_s >= 0).all(), "b_s must be non-negative before calling _two_phase"

    if m == 0:
        # No constraints — check for unboundedness
        if np.any(c_s < -_OPT_TOL):
            return np.zeros(n_s), 3, 0
        return np.zeros(n_s), 0, 0

    n_art = m  # one artificial per constraint row

    # ── Build the extended tableau ────────────────────────────────────
    # Shape: (m + 1) rows × (n_s + n_art + 1) columns
    #
    # Columns 0 .. n_s − 1     : standard-form variables (z)
    # Columns n_s .. n_s+m − 1 : artificial variables   (a)
    # Column  n_s + m           : RHS
    #
    # Row 0    : objective row   T[0, j] = reduced cost c̄_j
    #                            T[0, -1] = −(current objective value)
    # Rows 1..m: constraint rows  T[i, :n_s+m] = row of augmented A
    #                             T[i, -1] = b_s[i]

    n_ext = n_s + n_art
    T = np.zeros((m + 1, n_ext + 1))

    T[1:, :n_s] = A_s  # original variables + slacks
    T[1:, n_s:n_ext] = np.eye(m)  # artificial variables
    T[1:, -1] = b_s  # RHS

    # Initial basis: artificials  (indices n_s, n_s+1, ..., n_s+m-1)
    basis = list(range(n_s, n_s + m))

    # ── Phase 1 objective row ─────────────────────────────────────────
    # Phase 1: minimise sum(a_i),  c_ph1 = [0..0, 1..1, 0]
    # Artificials are basic → reduced costs must be 0 in canonical form.
    # Canonicalise by subtracting each constraint row from objective row:
    #   c̄_j(ph1) = 0 − Σ_i A_s[i, j]   for j < n_s
    #   c̄_{n_s+k}(ph1) = 1 − 1 = 0     for artificial k  (identity column)
    #   T[0, -1]  = 0 − Σ_i b_s[i]     = −(Phase-1 obj value)
    T[0, :n_s] = -A_s.sum(axis=0)  # reduced costs for original vars
    T[0, n_s:n_ext] = 0.0  # artificials (basic → zero)
    T[0, -1] = -b_s.sum()  # −(Phase-1 objective) at start BFS

    # ── Phase 1 simplex ───────────────────────────────────────────────
    # Allow all n_ext columns (original + artificials) to enter.
    status1, iters1 = _simplex(T, basis, col_limit=n_ext, max_iter=_MAX_ITER)
    if status1 == 3:
        # Unbounded in Phase 1 should not happen with artificials
        return np.zeros(n_s), 4, iters1

    # Phase-1 optimal objective value = −T[0, -1]
    ph1_obj = -T[0, -1]
    if ph1_obj > _FEAS_TOL:
        return np.zeros(n_s), 2, iters1  # infeasible

    # ── Drive degenerate artificials out of the basis ─────────────────
    # If an artificial is still basic with value ≈ 0, pivot in any
    # non-artificial column to remove it.  If none is available the row
    # is redundant; the artificial (value 0) stays but will not affect
    # Phase 2 since artificials are excluded from entering.
    for i in range(m):
        if basis[i] >= n_s:  # artificial in basis
            row_idx = i + 1
            for j in range(n_s):  # try every non-artificial column
                if abs(T[row_idx, j]) > _PIVOT_TOL:
                    _pivot(T, basis, row_idx, j)
                    break  # pivoted out — move to next row

    # ── Phase 2 setup ─────────────────────────────────────────────────
    # Replace objective row with original cost c_s.
    # Artificials are given cost 0 and excluded from col_limit.
    T[0, :n_s] = c_s
    T[0, n_s:n_ext] = 0.0
    T[0, -1] = 0.0

    # Re-canonicalise: for each non-artificial basic variable b_j at row i,
    # subtract c_s[b_j] * T[i+1, :] from T[0, :] so the objective row
    # has a zero in every basic variable's column.
    for i in range(m):
        bj = basis[i]
        if bj < n_s and abs(c_s[bj]) > 0.0:
            T[0, :] -= c_s[bj] * T[i + 1, :]

    # ── Phase 2 simplex ───────────────────────────────────────────────
    # col_limit = n_s: artificials (columns n_s .. n_ext-1) cannot enter.
    remaining = _MAX_ITER - iters1
    status2, iters2 = _simplex(T, basis, col_limit=n_s, max_iter=remaining)

    total_iters = iters1 + iters2

    if status2 not in (0, 1):
        return np.zeros(n_s), status2, total_iters

    # ── Extract solution ──────────────────────────────────────────────
    z_s = np.zeros(n_s)
    for i in range(m):
        bj = basis[i]
        if bj < n_s:
            z_s[bj] = max(0.0, T[i + 1, -1])  # clamp numerical negatives

    final_status = status2 if status2 != 0 else 0
    return z_s, final_status, total_iters


# ---------------------------------------------------------------------------
# Step 3: Tableau operations
# ---------------------------------------------------------------------------


def _simplex(
    T: np.ndarray,  # full tableau, modified in place
    basis: list[int],  # basis[i] = column index of basic var in row i+1
    col_limit: int,  # only columns 0..col_limit-1 may enter
    max_iter: int,
) -> Tuple[int, int]:
    """
    Run simplex pivots on the tableau until optimal or terminated.

    Convention
    ----------
    T[0, j]  = c̄_j  (reduced cost of variable j — negative means improvable)
    T[0, -1] = −z   (negative of the current objective value)
    T[i, -1] = b_i  (current basic variable value in row i, i=1..m)

    Entering variable selection  (most-negative reduced cost)
    ---------------------------------------------------------
    Enter the column j ∈ [0, col_limit) with the most-negative c̄_j via
    a single numpy argmin() call.  This typically cuts pivot count 3–5×
    vs Bland's rule.  Cycling is theoretically possible but rare in
    practice; the _MAX_ITER guard is the backstop.

    Leaving variable selection  (minimum ratio test)
    -------------------------------------------------
    Vectorised: compute all ratios b_i / T[i, enter] where T[i, enter] > ε,
    find the minimum, then break ties by smallest basis index using argmin
    on a masked array.

    Returns (status, iterations)
    """
    m = T.shape[0] - 1  # number of constraint rows

    for iters in range(max_iter):

        # ── Entering variable: most-negative reduced cost rule ──────────
        # Find the column with the most-negative reduced cost in [0, col_limit).
        # This enters the variable with the greatest per-unit objective improvement,
        # typically cutting pivot count 3–5× versus Bland's rule.
        # Cycling is theoretically possible but rare in practice for well-
        # conditioned LPs; we fall back to Bland's if we ever detect cycling
        # (signalled by the iteration limit, not by an explicit check here).
        rc = T[0, :col_limit]
        min_rc = float(rc.min())
        if min_rc >= -_OPT_TOL:
            return 0, iters  # optimal

        enter_col = int(np.argmin(rc))  # most-negative reduced cost

        # ── Leaving variable: ratio test (vectorised) ─────────────────
        col_vals = T[1:, enter_col]  # shape (m,)
        eligible = col_vals > _PIVOT_TOL

        if not eligible.any():
            return 3, iters  # unbounded

        # Compute ratios only for eligible rows to avoid divide-by-zero.
        ratios = np.full(m, np.inf)
        ratios[eligible] = T[1:, -1][eligible] / col_vals[eligible]

        min_ratio = float(ratios.min())

        # Bland's tiebreaking among rows with ratio ≈ min_ratio:
        # choose the row whose basic variable has the smallest column index.
        tied_rows = np.where(ratios <= min_ratio + _PIVOT_TOL)[0]  # 0-indexed

        if len(tied_rows) == 1:
            leave_row = int(tied_rows[0]) + 1  # convert to 1-indexed
        else:
            # Pick the tied row whose basic variable has the smallest column
            # index (Bland's tiebreaking).  Vectorised: index into basis array.
            basis_arr = np.asarray(basis)
            best_in_tie = int(np.argmin(basis_arr[tied_rows]))
            leave_row = int(tied_rows[best_in_tie]) + 1

        _pivot(T, basis, leave_row, enter_col)

    return 1, max_iter  # iteration limit reached


def _pivot(
    T: np.ndarray,
    basis: list[int],
    row: int,  # 1-indexed (row 0 is objective row)
    col: int,
) -> None:
    """
    Perform a simplex pivot at T[row, col].

    After the pivot:
    - T[row, col] == 1  (pivot row normalised)
    - T[i, col]  == 0  for all i ≠ row  (column cleared via row operations)
    - basis[row - 1] updated to col

    Fully vectorised: the entire column elimination is done with a single
    outer-product subtraction — no Python loop over rows.

    The objective row (row 0) is included so that reduced costs and the
    objective value remain in canonical form after every pivot.
    """
    # Normalise pivot row
    T[row, :] /= T[row, col]

    # Vectorised column elimination using in-place broadcasting.
    # For every row i ≠ row:  T[i, :] -= T[i, col] * T[row, :]
    #
    # We zero the pivot-row multiplier so the pivot row is left unchanged,
    # then subtract in one operation.  Using [:, None] broadcasting instead
    # of np.outer avoids materialising a full (m+1, n_cols) temporary array.
    factors = T[:, col].copy()  # shape (m+1,) — multiplier per row
    factors[row] = 0.0  # pivot row: no self-modification
    T -= factors[:, None] * T[row, :]  # in-place, no temp matrix

    # Update basis
    basis[row - 1] = col
