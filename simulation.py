"""
simulation.py — Arcade window, game loop, and rendering.

Dummy stubs for VectorField, Robot, and LPOCP are defined here
and will be replaced module by module in later implementation steps.

Screen coordinate convention
-----------------------------
Arcade uses (0,0) at the bottom-left with y increasing upward —
the same orientation as our world/grid coordinate system.
Conversion from world metres to screen pixels is therefore uniform:

    screen_x = world_x / cell_size * cell_px
    screen_y = world_y / cell_size * cell_px
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import arcade

from map import GridMap
from robot import Robot
from solver import BaseSolver, KeyboardSolver

# ---------------------------------------------------------------------------
# Dummy stub — will be replaced by vector_field.py in a later step
# ---------------------------------------------------------------------------


class VectorField:
    """
    Dummy vector field.
    Real implementation: harmonic potential (Laplace solve).
    Returns a zero vector everywhere until implemented.
    """

    def compute(self, grid_map: GridMap, goal_world: Tuple[float, float]) -> None:
        pass

    def query(self, wx: float, wy: float) -> np.ndarray:
        return np.zeros(2)

    def potential(self, wx: float, wy: float) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Simulation window
# ---------------------------------------------------------------------------

# Robot radius comes from Robot.RADIUS_M so rendering and physics stay in sync.

# Colour palette — all as RGBA 4-tuples (arcade 3.x requires alpha channel)
_COL_OBSTACLE = (55, 55, 65, 255)
_COL_FREE = (185, 188, 195, 255)  # muted mid-grey, easy on the eyes
_COL_GOAL_FILL = (70, 210, 90, 110)  # semi-transparent green
_COL_GOAL_RING = (30, 170, 55, 255)  # solid outline
_COL_GOAL_X = (15, 110, 35, 255)  # darker inscribed X
_COL_ROBOT_FILL = (220, 60, 60, 255)
_COL_ROBOT_EDGE = (140, 20, 20, 255)
_COL_HUD_TEXT = (0, 0, 0, 255)
_COL_BACKGROUND = (140, 143, 150, 255)  # matches free cell tone so gaps blend


class Simulation(arcade.Window):
    """
    Main arcade window.

    Responsibilities:
      - Render the grid, robot, and goal every frame.
      - Call LPOCP.solve() → Robot.step() every update.
      - Provide keyboard shortcuts (ESC to quit, R to reset — later).
    """

    def __init__(self, grid_map: GridMap, cell_px: int = 24):
        self.grid_map = grid_map
        self.cell_px = cell_px

        screen_w = grid_map.cols * cell_px
        screen_h = grid_map.rows * cell_px

        super().__init__(screen_w, screen_h, "LP OCP Navigation Simulator")
        self.background_color = _COL_BACKGROUND

        # ------------------------------------------------------------------
        # Instantiate modules
        # ------------------------------------------------------------------
        self.vector_field = VectorField()  # dummy until vector_field.py
        self.solver: BaseSolver = KeyboardSolver()  # swap for LP solver later

        # Place robot at the map start position
        gx_s, gy_s = grid_map.start
        wx_s, wy_s = grid_map.grid_to_world(gx_s, gy_s)
        self.robot = Robot(wx_s, wy_s, theta=0.0)

        # Pre-compute vector field for the goal (no-op with dummy)
        gx_g, gy_g = grid_map.goal
        goal_world = grid_map.grid_to_world(gx_g, gy_g)
        self.vector_field.compute(grid_map, goal_world)

        # Track last commands for HUD display
        self._last_v: float = 0.0
        self._last_omega: float = 0.0

        # Build the static grid shape list once (drawn every frame cheaply)
        self._grid_sprites = self._build_grid_sprites()

        # Persistent Text objects — avoids draw_text PerformanceWarning
        self._hud_texts = self._build_hud_texts()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert world metres → screen pixels."""
        scale = self.cell_px / self.grid_map.cell_size
        return wx * scale, wy * scale

    def cell_center_screen(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the screen pixel centre of a grid cell."""
        px = (gx + 0.5) * self.cell_px
        py = (gy + 0.5) * self.cell_px
        return px, py

    # ------------------------------------------------------------------
    # Grid sprite list (static — rebuilt only on map/goal change)
    # ------------------------------------------------------------------

    def _build_grid_sprites(self) -> arcade.SpriteList:
        """
        Create one solid-colour sprite per cell.
        Using a SpriteList gives batched OpenGL draws — fast even for
        large grids.
        """
        sprite_list = arcade.SpriteList(use_spatial_hash=False)
        inner = self.cell_px - 1  # 1-pixel gap acts as grid line

        for gy in range(self.grid_map.rows):
            for gx in range(self.grid_map.cols):
                color = _COL_OBSTACLE if self.grid_map.grid[gy, gx] else _COL_FREE
                cx, cy = self.cell_center_screen(gx, gy)
                sprite = arcade.SpriteSolidColor(
                    width=inner,
                    height=inner,
                    center_x=cx,
                    center_y=cy,
                    color=color,
                )
                sprite_list.append(sprite)

        return sprite_list

    # ------------------------------------------------------------------
    # Arcade callbacks
    # ------------------------------------------------------------------

    def on_update(self, delta_time: float) -> None:
        """Called every frame. Query solver then advance robot dynamics."""
        v, omega = self.solver.solve(
            self.robot.state,
            self.vector_field,
            self.grid_map,
        )
        self._last_v = v
        self._last_omega = omega
        self.robot.step(v, omega, delta_time)

    def on_draw(self) -> None:
        """Called every frame to render the scene."""
        self.clear()

        # Draw layers back-to-front
        self._draw_grid()
        self._draw_goal()
        self._draw_robot()
        self._draw_hud()

    def on_key_press(self, key: int, modifiers: int) -> None:
        if key == arcade.key.ESCAPE:
            self.close()
        self.solver.on_key_press(key)

    def on_key_release(self, key: int, modifiers: int) -> None:
        self.solver.on_key_release(key)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        """Render the pre-built cell sprite list."""
        self._grid_sprites.draw()

    def _draw_goal(self) -> None:
        """
        Render the goal as a semi-transparent green circle with an
        inscribed darker X so it stays visible over any cell colour.

        Size is based on the physical robot radius so the goal marker
        is visually comparable to the robot footprint.
        """
        gx, gy = self.grid_map.goal
        cx, cy = self.cell_center_screen(gx, gy)

        # Match the robot footprint size so the goal is clearly visible
        m_to_px = self.cell_px / self.grid_map.cell_size
        r = Robot.RADIUS_M * m_to_px

        # Semi-transparent filled circle
        arcade.draw_circle_filled(cx, cy, r, _COL_GOAL_FILL)
        # Solid outline ring
        arcade.draw_circle_outline(cx, cy, r, _COL_GOAL_RING, border_width=2)

        # Inscribed X — two diagonal lines at ±45°
        # Arm endpoints sit at 80 % of the radius so they clear the outline.
        arm = r * 0.80
        diag = arm * math.sqrt(0.5)  # arm * cos(45°)

        lw = max(2, int(r * 0.12))  # line width scales with circle size
        # \ diagonal
        arcade.draw_line(cx - diag, cy - diag, cx + diag, cy + diag, _COL_GOAL_X, lw)
        # // diagonal
        arcade.draw_line(cx - diag, cy + diag, cx + diag, cy - diag, _COL_GOAL_X, lw)

    def _draw_robot(self) -> None:
        """
        Render the robot as a triangle pointing in the heading direction.

        The nose points forward (in the direction of θ); two rear corners
        are spread at ±135° from the nose.
        """
        sx, sy = self.world_to_screen(self.robot.x, self.robot.y)
        th = self.robot.theta

        # Convert physical radius to screen pixels so the robot stays the
        # same physical size regardless of cell_px or cell_size.
        m_to_px = self.cell_px / self.grid_map.cell_size
        r_body = Robot.RADIUS_M * m_to_px  # circle hull radius
        r_nose = r_body * 1.0
        r_rear = r_body * 0.8

        # Draw body circle first so the triangle arrow sits on top
        arcade.draw_circle_filled(sx, sy, r_body, _COL_ROBOT_FILL)
        arcade.draw_circle_outline(sx, sy, r_body, _COL_ROBOT_EDGE, border_width=2)

        # Directional arrow: nose tip + two rear corners
        nose = (sx + r_nose * math.cos(th), sy + r_nose * math.sin(th))
        rear_l = (
            sx + r_rear * math.cos(th + math.radians(140)),
            sy + r_rear * math.sin(th + math.radians(140)),
        )
        rear_r = (
            sx + r_rear * math.cos(th - math.radians(140)),
            sy + r_rear * math.sin(th - math.radians(140)),
        )

        arcade.draw_polygon_filled([nose, rear_l, rear_r], _COL_ROBOT_EDGE)

        # Dot at nose tip
        # dot_r = max(2.0, r_body * 0.12)
        # arcade.draw_circle_filled(nose[0], nose[1], dot_r, (255, 255, 255, 220))

    def _build_hud_texts(self) -> list:
        """
        Create one arcade.Text object per HUD line.
        Created once in __init__, .text updated each frame.
        """
        font_size = 12
        line_h = 18
        pad = 8
        n_lines = 4
        start_y = self.height - pad - font_size
        return [
            arcade.Text(
                text="",
                x=pad,
                y=start_y - i * line_h,
                color=(240, 240, 240, 255),
                font_size=font_size,
                font_name="monospace",
            )
            for i in range(n_lines)
        ]

    def _draw_hud(self) -> None:
        """Overlay current state information in the top-left corner."""
        x_m = self.robot.x
        y_m = self.robot.y
        th_d = math.degrees(self.robot.theta)
        gx, gy = self.grid_map.world_to_grid(x_m, y_m)

        strings = [
            f"pos : ({x_m:.2f} m, {y_m:.2f} m)  cell: ({gx}, {gy})",
            f"\u03b8   : {th_d:+.1f}\u00b0",
            f"v   : {self._last_v:.3f} m/s    \u03c9: {self._last_omega:.3f} rad/s",
            "[ESC] quit",
        ]

        for t, s in zip(self._hud_texts, strings):
            t.text = s

        # Semi-transparent dark panel — readable over any cell colour
        # while still letting the map show through at the edges.
        # draw_lbwh_rectangle_filled(left, bottom, width, height, color)
        line_h = 18
        pad = 8
        panel_h = len(strings) * line_h + pad * 2
        panel_w = 360
        arcade.draw_lbwh_rectangle_filled(
            0,
            self.height - panel_h,
            panel_w,
            panel_h,
            (20, 20, 30, 200),
        )

        for t in self._hud_texts:
            t.draw()
