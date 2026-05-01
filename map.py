"""
map.py — Grid map representation and text-file parser.

All values in the map file are in metres.  The parser converts them to
integer cell indices internally by snapping to the nearest cell.

Coordinate conventions
----------------------
  World space : (wx, wy) in metres.
                wx increases left→right, wy increases bottom→top.

  Grid space  : (gx, gy) integer cell indices derived from world space.
                gx = round(wx / cell_size),  gy = round(wy / cell_size).
                Cell (gx, gy) covers the world region
                [gx*cs, (gx+1)*cs) × [gy*cs, (gy+1)*cs).
                The cell *centre* is at ((gx+0.5)*cs, (gy+0.5)*cs).

  Internal grid: a 1-cell border of obstacles is added automatically,
                 so the numpy array is (rows+2) × (cols+2) and all
                 user-space cell indices are offset by +1 internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import PIL.Image
import numpy as np

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class GridMap:
    """
    Holds the occupancy grid and all derived geometry.

    grid[gy, gx] == True  →  obstacle cell
    grid[gy, gx] == False →  free cell

    ``start`` and ``goal`` are stored in *internal* grid coordinates
    (already offset for the 1-cell border).
    """

    grid: np.ndarray  # shape (rows, cols), dtype bool
    cell_size: float  # metres per cell
    start: Tuple[int, int]  # (gx, gy) internal coordinates
    goal: Tuple[int, int]  # (gx, gy) internal coordinates

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def width_m(self) -> float:
        return self.cols * self.cell_size

    @property
    def height_m(self) -> float:
        return self.rows * self.cell_size

    # ------------------------------------------------------------------
    # Cell queries
    # ------------------------------------------------------------------

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.cols and 0 <= gy < self.rows:
            return not self.grid[gy, gx]
        return False

    def is_obstacle(self, gx: int, gy: int) -> bool:
        return not self.is_free(gx, gy)

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Return the cell that contains the world-space point (wx, wy)."""
        return int(wx / self.cell_size), int(wy / self.cell_size)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the world-space *centre* of cell (gx, gy)."""
        return (gx + 0.5) * self.cell_size, (gy + 0.5) * self.cell_size

    # ------------------------------------------------------------------
    # Numpy view
    # ------------------------------------------------------------------

    def obstacle_mask(self) -> np.ndarray:
        return self.grid.copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snap(metres: float, cell_size: float) -> int:
    """Snap a metre value to the nearest cell index (0-based)."""
    return int(round(metres / cell_size))


def _parse_point(token: str) -> Tuple[float, float]:
    """Parse 'x,y' token as two floats (metres)."""
    parts = token.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y' in metres, got {token!r}")
    return float(parts[0]), float(parts[1])


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """All grid cells along the line from (x0,y0) to (x1,y1)."""
    cells: List[Tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_map_from_bitmap(path: str) -> GridMap:
    texture = PIL.Image.open(path)

    # downsample
    factor = 4
    texture = texture.resize((texture.size[0] // factor, texture.size[1] // factor))

    grid = np.array(texture.convert("1").getdata(), dtype=bool).reshape(
        texture.size[::-1]
    )

    # invert
    grid = ~grid

    return GridMap(grid, 0.05, (10, 10), (grid.shape[1] - 1, grid.shape[0] - 1))


def load_map(path: str) -> GridMap:
    """
    Parse a text map file and return a GridMap.

    All spatial values are in **metres**.  The parser snaps them to the
    nearest cell using ``round(value / cell_size)``.

    File format
    -----------
    Lines beginning with '#' or empty lines are ignored.

    ``size <width_m> <height_m>``
        Physical size of the usable area in metres.  Must appear before
        any spatial directive.  The number of cells is derived as
        ``cols = round(width_m / cell_size)``.

    ``cell <metres>``
        Cell size in metres.  Default: 0.05.  Should appear before
        ``size`` so the column/row count is computed correctly; if it
        appears after, the size directive is re-evaluated.

    ``obstacle <x1,y1> <x2,y2>``
        Line obstacle from world point (x1,y1) to (x2,y2) in metres.
        Both endpoints are snapped to the nearest cell, then every cell
        along the Bresenham line is marked as an obstacle.

    ``rect <x1,y1> <x2,y2>``
        Filled axis-aligned rectangle.  The two metre-space corners are
        snapped to cells and every cell inside is marked as an obstacle.

    ``start <x_m> <y_m>``
        Robot start position in metres (snapped to nearest cell).

    ``goal <x_m> <y_m>``
        Goal position in metres (snapped to nearest cell).

    Border walls
    ------------
    A 1-cell-thick layer of obstacles is added around the full perimeter
    automatically, so the internal numpy grid is (rows+2) × (cols+2).

    Example
    -------
    ::

        size  10.0  7.5
        cell  0.05

        # Horizontal wall at y=3.75 m with 0.9 m doorway
        rect  0.0,3.75   3.7,3.95
        rect  4.6,3.75  10.0,3.95

        start  0.5  0.5
        goal   9.25 6.90
    """

    # Raw parsed values — all in metres
    width_m: float | None = None
    height_m: float | None = None
    cell_size: float = 0.05  # sensible ROS2-compatible default

    # Deferred: (p1_m, p2_m) pairs for obstacle lines
    raw_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    # Deferred: (corner1_m, corner2_m) pairs for filled rects
    raw_rects: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    start_m: Tuple[float, float] | None = None
    goal_m: Tuple[float, float] | None = None

    with open(path, "r") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            keyword = tokens[0].lower()

            try:
                if keyword == "size":
                    if len(tokens) != 3:
                        raise ValueError(
                            "'size' requires exactly 2 arguments: width_m height_m"
                        )
                    width_m = float(tokens[1])
                    height_m = float(tokens[2])
                    if width_m <= 0 or height_m <= 0:
                        raise ValueError("'size' dimensions must be positive")

                elif keyword == "cell":
                    if len(tokens) != 2:
                        raise ValueError("'cell' requires exactly 1 argument: metres")
                    cell_size = float(tokens[1])
                    if cell_size <= 0:
                        raise ValueError("'cell' size must be positive")

                elif keyword == "obstacle":
                    if len(tokens) != 3:
                        raise ValueError(
                            "'obstacle' requires 2 arguments: x1,y1 x2,y2 (metres)"
                        )
                    raw_lines.append((_parse_point(tokens[1]), _parse_point(tokens[2])))

                elif keyword == "rect":
                    if len(tokens) != 3:
                        raise ValueError(
                            "'rect' requires 2 arguments: x1,y1 x2,y2 (metres)"
                        )
                    raw_rects.append((_parse_point(tokens[1]), _parse_point(tokens[2])))

                elif keyword == "start":
                    if len(tokens) != 3:
                        raise ValueError("'start' requires 2 arguments: x_m y_m")
                    start_m = (float(tokens[1]), float(tokens[2]))

                elif keyword == "goal":
                    if len(tokens) != 3:
                        raise ValueError("'goal' requires 2 arguments: x_m y_m")
                    goal_m = (float(tokens[1]), float(tokens[2]))

                else:
                    raise ValueError(f"Unknown keyword '{keyword}'")

            except ValueError as exc:
                raise ValueError(f"Line {line_no}: {exc}") from exc

    # ------------------------------------------------------------------
    # Validate mandatory fields
    # ------------------------------------------------------------------
    if width_m is None or height_m is None:
        raise ValueError("Map file must contain a 'size' directive")
    if start_m is None:
        raise ValueError("Map file must contain a 'start' directive")
    if goal_m is None:
        raise ValueError("Map file must contain a 'goal' directive")

    # ------------------------------------------------------------------
    # Derive grid dimensions from physical size
    # ------------------------------------------------------------------
    cols = round(width_m / cell_size)
    rows = round(height_m / cell_size)
    if cols < 1 or rows < 1:
        raise ValueError(
            f"Grid too small: {width_m}m / {cell_size}m/cell = {cols} cols, "
            f"{height_m}m / {cell_size}m/cell = {rows} rows"
        )

    # Internal grid includes 1-cell border on every side
    total_cols = cols + 2
    total_rows = rows + 2

    grid = np.zeros((total_rows, total_cols), dtype=bool)

    # Perimeter border
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True

    # ------------------------------------------------------------------
    # Helper: snap metre coord to internal (border-offset) cell index
    # ------------------------------------------------------------------
    def to_internal(mx: float, my: float) -> Tuple[int, int]:
        """Snap world metres to internal grid cell (includes +1 border offset)."""
        return _snap(mx, cell_size) + 1, _snap(my, cell_size) + 1

    def mark(gx: int, gy: int, label: str) -> None:
        if not (0 <= gx < total_cols and 0 <= gy < total_rows):
            raise ValueError(
                f"{label}: cell ({gx-1}, {gy-1}) is outside the map "
                f"({cols}×{rows} cells / {width_m}×{height_m} m)"
            )
        grid[gy, gx] = True

    # ------------------------------------------------------------------
    # Rasterise obstacle lines
    # ------------------------------------------------------------------
    for (mx1, my1), (mx2, my2) in raw_lines:
        ix1, iy1 = to_internal(mx1, my1)
        ix2, iy2 = to_internal(mx2, my2)
        for gx, gy in _bresenham(ix1, iy1, ix2, iy2):
            mark(gx, gy, f"obstacle ({mx1},{my1})-({mx2},{my2})")

    # ------------------------------------------------------------------
    # Rasterise filled rectangles
    # ------------------------------------------------------------------
    for (mx1, my1), (mx2, my2) in raw_rects:
        ix1, iy1 = to_internal(mx1, my1)
        ix2, iy2 = to_internal(mx2, my2)
        x_lo, x_hi = sorted([ix1, ix2])
        y_lo, y_hi = sorted([iy1, iy2])
        for gy in range(y_lo, y_hi + 1):
            for gx in range(x_lo, x_hi + 1):
                mark(gx, gy, f"rect ({mx1},{my1})-({mx2},{my2})")

    # ------------------------------------------------------------------
    # Snap start and goal
    # ------------------------------------------------------------------
    sx, sy = to_internal(*start_m)
    gx_g, gy_g = to_internal(*goal_m)

    if not (1 <= sx <= cols and 1 <= sy <= rows):
        raise ValueError(f"Start {start_m} m is outside the usable area")
    if not (1 <= gx_g <= cols and 1 <= gy_g <= rows):
        raise ValueError(f"Goal {goal_m} m is outside the usable area")
    if grid[sy, sx]:
        raise ValueError(f"Start {start_m} m lands on an obstacle")
    if grid[gy_g, gx_g]:
        raise ValueError(f"Goal {goal_m} m lands on an obstacle")

    return GridMap(
        grid=grid,
        cell_size=cell_size,
        start=(sx, sy),
        goal=(gx_g, gy_g),
    )
