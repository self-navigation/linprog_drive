"""
grid_views.py — Grid coloring functions for the simulation renderer.

Each view is a callable with signature:
    (grid_map: GridMap, vector_field, cell_px: int) -> np.ndarray

The returned array has shape (rows * cell_px, cols * cell_px, 4) with
dtype uint8 and RGBA channels. Arcade's y-axis points up, so row 0 of
the grid (gy=0) maps to the bottom of the screen; the pixel array is
built in that orientation (row 0 of the array = gy=0 = screen bottom).

To add a new view, define a function with the signature above, optionally
define a Colormap for it, and add a GridView entry to VIEWS at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from map import GridMap
from vector_field import VectorField

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# vector_field is typed loosely so this module doesn't import from vector_field
# (avoiding a circular dependency if solvers ever import grid_views).
GridViewFn = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Colormap — describes how to draw the sidebar colour bar
# ---------------------------------------------------------------------------


@dataclass
class ColormapTick:
    """A labelled tick on the colorbar at a normalised position in [0, 1]."""

    pos: float  # 0 = bottom of bar, 1 = top
    label: str  # shown to the left of the bar


@dataclass
class Colormap:
    color_fn: Callable[[float], tuple[int, int, int]]
    ticks: list[ColormapTick]
    title: str
    _lut: np.ndarray | None = field(default=None, init=False, repr=False)
    _lut_size: int = field(default=2048, init=False, repr=False)

    def _build_lut(self) -> None:
        """Build lookup table once (lazy initialization)."""
        if self._lut is not None:
            return

        lut = np.zeros((self._lut_size, 4), dtype=np.uint8)
        lut[:, 3] = 255  # Alpha channel

        # Pre-compute color for each index
        for i in range(self._lut_size):
            t = i / (self._lut_size - 1)
            r, g, b = self.color_fn(t)
            lut[i, :3] = [r, g, b]

        self._lut = lut

    def rgba_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorised: values shape (...), returns (..., 4) uint8 RGBA."""
        self._build_lut()

        # Clip to [0, 1] and convert to lookup indices
        clipped = np.clip(values, 0.0, 1.0)
        indices = (clipped * (self._lut_size - 1)).astype(np.uint32)

        # Vectorized lookup using fancy indexing
        original_shape = values.shape
        rgba = self._lut[indices.ravel()]

        return rgba.reshape(*original_shape, 4)


# ---------------------------------------------------------------------------
# Colormap definitions
# ---------------------------------------------------------------------------


def _cm_potential(t: float) -> tuple[int, int, int]:
    """Green (goal) → blue (mid) → red (far/walls)."""
    colormap = plt.cm.get_cmap("nipy_spectral")

    compress = lambda t: t**1.25
    r, g, b, a = colormap(compress(t))

    # desaturate
    sat_fac = 0.5
    r = sat_fac * r + (1 - sat_fac) * 0.5
    g = sat_fac * g + (1 - sat_fac) * 0.5
    b = sat_fac * b + (1 - sat_fac) * 0.5

    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return r, g, b


def _cm_diverging(t: float) -> tuple[int, int, int]:
    colormap = plt.cm.get_cmap("bwr")

    r, g, b, a = colormap(t)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return r, g, b


def _cm_magnitude(t: float) -> tuple[int, int, int]:
    """Black (flat) -> purple -> orange -> white (steep)."""
    r, g, b, a = plt.cm.get_cmap("magma")(t)
    return int(r * 255), int(g * 255), int(b * 255)


COLORMAP_POTENTIAL = Colormap(
    color_fn=_cm_potential,
    ticks=[
        ColormapTick(0.0, "0.0  goal"),
        ColormapTick(0.5, "0.5"),
        ColormapTick(1.0, "1.0  far"),
    ],
    title="potential φ",
)

COLORMAP_MAGNITUDE = Colormap(
    color_fn=_cm_magnitude,
    ticks=[
        ColormapTick(0.0, "0   flat"),
        ColormapTick(0.5, "mid"),
        ColormapTick(1.0, "1   steep"),
    ],
    title="|\u2207\u03c6|",
)

COLORMAP_CURL = Colormap(
    color_fn=_cm_diverging,
    ticks=[
        ColormapTick(0.0, "CW  -"),
        ColormapTick(0.5, "0"),
        ColormapTick(1.0, "CCW +"),
    ],
    title="curl",
)


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

_COL_OBSTACLE = np.array([55, 55, 65, 255], dtype=np.uint8)
_COL_FREE = np.array([185, 188, 195, 255], dtype=np.uint8)


def _apply_colormap_vectorised(values: np.ndarray, colormap: Colormap) -> np.ndarray:
    return colormap.rgba_array(values)


def _upscale(cell_rgba: np.ndarray, cell_px: int) -> np.ndarray:
    """Nearest-neighbour upscale from (rows, cols, 4) to (rows*px, cols*px, 4)."""
    return np.repeat(np.repeat(cell_rgba, cell_px, axis=0), cell_px, axis=1)


# ---------------------------------------------------------------------------
# View functions
# ---------------------------------------------------------------------------


def view_normal(grid_map: GridMap, vector_field, cell_px: int) -> np.ndarray:
    """Standard occupancy view: dark obstacles, muted-grey free cells."""
    cell_rgba = np.where(
        grid_map.grid[..., np.newaxis], _COL_OBSTACLE, _COL_FREE
    ).astype(np.uint8)
    return _upscale(cell_rgba, cell_px)


def view_potential(
    grid_map: GridMap, vector_field: VectorField, cell_px: int
) -> np.ndarray:
    """Potential φ: green (goal) → blue (mid) → red (far/walls)."""
    phi = vector_field.phi
    if phi is None:
        return view_normal(grid_map, vector_field, cell_px)

    cell_rgba = _apply_colormap_vectorised(phi, COLORMAP_POTENTIAL)
    cell_rgba[grid_map.grid] = _COL_OBSTACLE
    return _upscale(cell_rgba, cell_px)


def view_field_angle(grid_map: GridMap, vector_field, cell_px: int) -> np.ndarray:
    """
    Navigation direction as a hue wheel.

    Each free cell is coloured by the angle of -grad(phi):
        East  (right)  -> red
        North (up)     -> yellow-green
        West  (left)   -> cyan
        South (down)   -> magenta

    Saturation encodes signal strength: cells where the gradient is weak
    (flat potential region) fade to grey so ambiguous areas stand out.
    Obstacle cells are drawn in the standard dark colour.

    No sidebar colormap is shown because the hue encoding is circular
    (east maps to both ends of any linear bar).
    """
    from matplotlib.colors import hsv_to_rgb

    phi = vector_field.phi
    if phi is None:
        return view_normal(grid_map, vector_field, cell_px)

    grad = np.gradient(phi)
    # Field direction is -grad(phi); atan2 gives angle in [-pi, pi]
    vx = -grad[1]
    vy = -grad[0]
    angle = np.arctan2(vy, vx)
    hue = (angle + np.pi) / (2.0 * np.pi)  # map [-pi, pi] -> [0, 1]

    # Saturation proportional to gradient magnitude, clipped at 99th percentile
    # so one steep obstacle boundary doesn't wash out the rest of the map.
    mag = np.sqrt(vx**2 + vy**2)
    free = ~grid_map.grid
    p99 = float(np.percentile(mag[free], 99)) if free.any() else 1.0
    sat = np.clip(mag / max(p99, 1e-12), 0.0, 1.0)

    val = np.full_like(hue, 0.88)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)

    cell_rgba = np.empty((*grid_map.grid.shape, 4), dtype=np.uint8)
    cell_rgba[..., :3] = rgb
    cell_rgba[..., 3] = 255
    cell_rgba[grid_map.grid] = _COL_OBSTACLE
    return _upscale(cell_rgba, cell_px)


def view_gradient_magnitude(
    grid_map: GridMap, vector_field, cell_px: int
) -> np.ndarray:
    """
    Gradient magnitude |grad(phi)|.

    Bright (white/orange) = steep potential = strong, unambiguous guidance.
    Dark (black/purple)   = flat region     = weak signal, robot may wander.

    Useful for spotting plateau regions near concave obstacle corners or
    anywhere the potential field loses navigational authority.
    Normalised to the 99th percentile of free-cell values so a single steep
    wall boundary does not compress the rest of the scale to nothing.
    """
    phi = vector_field.phi
    if phi is None:
        return view_normal(grid_map, vector_field, cell_px)

    grad = np.gradient(phi)
    mag = np.sqrt(grad[0] ** 2 + grad[1] ** 2)

    free = ~grid_map.grid
    p99 = float(np.percentile(mag[free], 99)) if free.any() else 1.0
    if p99 < 1e-12:
        p99 = 1.0
    mag_norm = np.clip(mag / p99, 0.0, 1.0)

    cell_rgba = _apply_colormap_vectorised(mag_norm, COLORMAP_MAGNITUDE)
    cell_rgba[grid_map.grid] = _COL_OBSTACLE
    return _upscale(cell_rgba, cell_px)


def view_curl(grid_map: GridMap, vector_field, cell_px: int) -> np.ndarray:
    """
    Curl of the unit navigation field: d(vy)/dx - d(vx)/dy.

    A pure gradient field is irrotational (curl = 0 everywhere).  Non-zero
    values indicate numerical artefacts — typically from the Gaussian
    smoothing boundary or a strong repulsion term — that can cause the robot
    to orbit rather than converge.

    Blue  = CCW (positive curl)
    White = zero curl (ideal)
    Red   = CW  (negative curl)

    The scale is normalised to the 98th percentile of free-cell magnitudes
    so that boundary spikes do not dominate.  Cells adjacent to obstacles
    may show elevated curl due to the zero-padding of the obstacle mask
    in the gradient stencil; this is a display artefact, not a real field
    defect.
    """
    # _vx / _vy are private but both VectorField and FMMVectorField expose
    # them under the same name; this view module is intentionally tightly
    # coupled to the vector field implementations.
    vx = getattr(vector_field, "_vx", None)
    vy = getattr(vector_field, "_vy", None)
    if vx is None or vy is None:
        return view_normal(grid_map, vector_field, cell_px)

    # Normalise to unit vectors so the curl reflects direction changes, not
    # magnitude changes (FMMVectorField stores units already; VectorField
    # stores the smoothed-but-unscaled gradient).
    mag = np.sqrt(vx**2 + vy**2)
    safe_mag = np.where(mag > 1e-8, mag, 1.0)
    ux = vx / safe_mag
    uy = vy / safe_mag
    ux[grid_map.grid] = 0.0
    uy[grid_map.grid] = 0.0

    # curl = d(uy)/d(col) - d(ux)/d(row)  [axis 1 = col = x, axis 0 = row = y]
    d_uy_dx = np.gradient(uy, axis=1)
    d_ux_dy = np.gradient(ux, axis=0)
    curl = d_uy_dx - d_ux_dy

    free = ~grid_map.grid
    p98 = float(np.percentile(np.abs(curl[free]), 98)) if free.any() else 1.0
    if p98 < 1e-12:
        p98 = 1.0

    # Map to [0, 1]: 0 = max CW, 0.5 = zero, 1 = max CCW
    curl_norm = np.clip(0.5 + curl / (2.0 * p98), 0.0, 1.0)

    cell_rgba = _apply_colormap_vectorised(curl_norm, COLORMAP_CURL)
    cell_rgba[grid_map.grid] = _COL_OBSTACLE
    return _upscale(cell_rgba, cell_px)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class GridView:
    name: str
    fn: GridViewFn
    colormap: Colormap | None = field(default=None)


VIEWS: list[GridView] = [
    GridView("normal", view_normal, colormap=None),
    GridView("potential phi", view_potential, colormap=COLORMAP_POTENTIAL),
    GridView("field direction", view_field_angle, colormap=None),
    GridView("grad magnitude", view_gradient_magnitude, colormap=COLORMAP_MAGNITUDE),
    GridView("curl", view_curl, colormap=COLORMAP_CURL),
]
