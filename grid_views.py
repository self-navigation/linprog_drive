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
    _lut_size: int = field(default=256, init=False, repr=False)

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
    colormap = plt.cm.get_cmap("viridis")

    r, g, b, a = colormap(t)
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


COLORMAP_POTENTIAL = Colormap(
    color_fn=_cm_potential,
    ticks=[
        ColormapTick(0.0, "0.0  goal"),
        ColormapTick(0.5, "0.5"),
        ColormapTick(1.0, "1.0  far"),
    ],
    title="potential φ",
)

COLORMAP_DIVERGING = Colormap(
    color_fn=_cm_diverging,
    ticks=[
        ColormapTick(0.0, "0.0  low"),
        ColormapTick(0.5, "0.5  flat"),
        ColormapTick(1.0, "1.0  steep"),
    ],
    title="∇φ variation",
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


def view_neighbor_diff(
    grid_map: GridMap, vector_field: VectorField, cell_px: int
) -> np.ndarray:
    """
    Compare each cell to average of 4-connected neighbors.

    0 = much lower than neighbors
    0.5 = equal to average of neighbors
    1 = much higher than neighbors
    """
    phi = vector_field.phi
    if phi is None:
        return view_normal(grid_map, vector_field, cell_px)

    # Calculate average of 4-connected neighbors
    neighbor_avg = (
        np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        + np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
    ) / 4.0

    # Difference from average
    diff = phi - neighbor_avg

    # Zero out wrap-around edges (important!)
    diff[:, 0] = diff[:, -1] = diff[0, :] = diff[-1, :] = 0.0

    # Find max absolute difference to scale appropriately
    d_max = float(np.abs(diff[~grid_map.grid]).max()) if (~grid_map.grid).any() else 1.0
    if d_max < 1e-12:
        d_max = 1.0

    # Map to [0, 1] with 0.5 as neutral (equal to average)
    norm = 0.5 + (diff / (2 * d_max))
    norm = np.clip(norm, 0.0, 1.0)

    cell_rgba = _apply_colormap_vectorised(norm, COLORMAP_DIVERGING)
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
    GridView("potential φ", view_potential, colormap=COLORMAP_POTENTIAL),
    GridView("neighbor diff", view_neighbor_diff, colormap=COLORMAP_DIVERGING),
]
