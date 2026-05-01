"""
simulation.py — Arcade window, game loop, and rendering.

VectorField, Robot, and solver live in their own modules.
This file owns the arcade window, rendering, and key routing.

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
from solver import BaseSolver, KeyboardSolver, GradientSolver, LPSolver
from vector_field import FMMVectorField

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


def _phi_to_color(phi: float) -> tuple:
    """
    Map a potential value φ ∈ [0, 1] to an RGBA colour.

    Three-stop gradient:
        0.0  →  green  (40, 180, 100)  — goal
        0.5  →  blue   (50, 120, 210)  — mid-range
        1.0  →  red    (200,  60,  50) — near obstacles
    """
    phi = max(0.0, min(1.0, phi))
    if phi <= 0.5:
        t = phi * 2.0  # 0 → 1 over first half
        r = int(40 + t * (50 - 40))
        g = int(180 + t * (120 - 180))
        b = int(100 + t * (210 - 100))
    else:
        t = (phi - 0.5) * 2.0  # 0 → 1 over second half
        r = int(50 + t * (200 - 50))
        g = int(120 + t * (60 - 120))
        b = int(210 + t * (50 - 210))
    return (r, g, b, 255)


def _collides_circle(
    query_point: tuple[float, float],
    circle_center: tuple[float, float],
    radius: float,
) -> bool:
    """
    Does the query_point fall within a circle with circle_center and radius?
    """
    return (query_point[0] - circle_center[0]) ** 2 + (
        query_point[1] - circle_center[1]
    ) ** 2 <= radius**2


class Simulation(arcade.Window):
    """
    Main arcade window.

    Responsibilities:
      - Render the grid, robot, and goal every frame.
      - Call LPOCP.solve() → Robot.step() every update.
      - Provide keyboard shortcuts (ESC to quit, R to reset — later).
    """

    def __init__(
        self,
        grid_map: GridMap,
        cell_px: int = 24,
        repulsion_radius: float = 0.6,
        repulsion_alpha: float = 5.0,
    ):
        self.grid_map = grid_map
        self.cell_px = cell_px

        screen_w = grid_map.cols * cell_px
        screen_h = grid_map.rows * cell_px

        super().__init__(screen_w, screen_h, "LP OCP Navigation Simulator")
        self.background_color = _COL_BACKGROUND

        # ------------------------------------------------------------------
        # Instantiate modules
        # ------------------------------------------------------------------
        self.vector_field = FMMVectorField()

        # Ordered list of available solvers — Tab cycles through them
        self._solvers: list[BaseSolver] = [
            KeyboardSolver(),
            GradientSolver(),
            LPSolver(),
        ]
        self._solver_idx: int = 0
        self.solver: BaseSolver = self._solvers[0]

        # Pause — Space freezes robot movement but keeps rendering
        self._paused: bool = True

        # Place robot at the map start position
        gx_s, gy_s = grid_map.start
        wx_s, wy_s = grid_map.grid_to_world(gx_s, gy_s)
        self.robot = Robot(wx_s, wy_s, theta=0.0)

        # Compute vector field (Dijkstra geodesic distances — runs once at startup)
        gx_g, gy_g = grid_map.goal
        goal_world = grid_map.grid_to_world(gx_g, gy_g)

        self.repulsion_radius = repulsion_radius
        self.repulsion_alpha = repulsion_alpha
        self.vector_field.compute(
            grid_map,
            goal_world,
            repulsion_radius=self.repulsion_radius,
            repulsion_alpha=self.repulsion_alpha,
        )
        self.vector_field_invalidated = False

        # Track last commands for HUD display
        self._last_v: float = 0.0
        self._last_omega: float = 0.0

        # View state: False = normal grid, True = potential gradient colourmap
        self._view_gradient: bool = False
        # UI visibility toggle — hides HUD and colorbar when False
        self._show_ui: bool = True

        # Build both sprite lists (gradient list needs the solved φ array)
        self._sprites_normal = self._build_sprites_normal()
        self._sprites_gradient = self._build_sprites_gradient()

        # Colorbar shown in gradient view
        self._colorbar_sprites, self._colorbar_texts = self._build_colorbar()

        # Persistent Text objects — avoids draw_text PerformanceWarning
        self._hud_texts = self._build_hud_texts()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def w2s_scale(self) -> float:
        return self.cell_px / self.grid_map.cell_size

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert world metres → screen pixels."""
        return wx * self.w2s_scale, wy * self.w2s_scale

    def cell_center_screen(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the screen pixel centre of a grid cell."""
        px = (gx + 0.5) * self.cell_px
        py = (gy + 0.5) * self.cell_px
        return px, py

    def screen_to_cell(self, sx: int, sy: int) -> Tuple[int, int]:
        """Convert screen pixels → grid cell."""
        gx = sx // self.cell_px
        gy = sy // self.cell_px
        return int(gx), int(gy)

    # ------------------------------------------------------------------
    # Grid sprite lists (built once after vector field is computed)
    # ------------------------------------------------------------------

    def _make_sprite(self, gx: int, gy: int, color: tuple) -> arcade.SpriteSolidColor:
        inner = self.cell_px - 1
        cx, cy = self.cell_center_screen(gx, gy)
        return arcade.SpriteSolidColor(
            width=inner, height=inner, center_x=cx, center_y=cy, color=color
        )

    def _build_sprites_normal(self) -> arcade.SpriteList:
        """Standard view: dark obstacles, muted-grey free cells."""
        sl = arcade.SpriteList(use_spatial_hash=False)
        for gy in range(self.grid_map.rows):
            for gx in range(self.grid_map.cols):
                color = _COL_OBSTACLE if self.grid_map.grid[gy, gx] else _COL_FREE
                sl.append(self._make_sprite(gx, gy, color))
        return sl

    def _build_sprites_gradient(self) -> arcade.SpriteList:
        """
        Gradient view: free cells coloured by harmonic potential φ.

        Colour stops (φ ∈ [0, 1]):
            0.0  →  green   (goal)
            0.5  →  blue    (mid-range)
            1.0  →  red     (near obstacles)

        Obstacle cells keep their dark colour so walls remain visible.
        """
        sl = arcade.SpriteList(use_spatial_hash=False)
        phi = self.vector_field.phi  # may be None if compute() not done yet

        for gy in range(self.grid_map.rows):
            for gx in range(self.grid_map.cols):
                if self.grid_map.grid[gy, gx]:
                    color = _COL_OBSTACLE
                elif phi is None:
                    color = _COL_FREE
                else:
                    color = _phi_to_color(float(phi[gy, gx]))
                sl.append(self._make_sprite(gx, gy, color))
        return sl

    # ------------------------------------------------------------------
    # Arcade callbacks
    # ------------------------------------------------------------------

    def on_update(self, delta_time: float) -> None:
        """Called every frame. Query solver then advance robot dynamics."""
        if self._paused:
            return
        v, omega = self.solver.solve(
            self.robot.state,
            self.vector_field,
            self.grid_map,
        )
        self._last_v = v
        self._last_omega = omega
        self.robot.step(v, omega, delta_time, self.grid_map)

    def on_draw(self) -> None:
        """Called every frame to render the scene."""
        self.clear()

        # Draw layers back-to-front
        self._run_ui()

    def _run_ui(self) -> None:
        self._draw_grid()
        self._draw_goal()
        self._draw_robot()
        if self._show_ui:
            if self._view_gradient:
                self._draw_colorbar()
            self._draw_hud()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.ESCAPE:
            self.close()
        elif symbol == arcade.key.V:
            self._view_gradient = not self._view_gradient
        elif symbol == arcade.key.Q:
            self._show_ui = not self._show_ui
        elif symbol == arcade.key.SPACE:
            self._paused = not self._paused
        elif symbol == arcade.key.TAB:
            self._solver_idx = (self._solver_idx + 1) % len(self._solvers)
            self.solver = self._solvers[self._solver_idx]
        else:
            self.solver.on_key_press(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.solver.on_key_release(symbol)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        # if the mouse is within the robot circle:
        sx, sy = self.world_to_screen(self.robot.x, self.robot.y)
        if _collides_circle((x, y), (sx, sy), self.robot.RADIUS_M * self.w2s_scale):
            # translate robot
            self.robot.x += dx / self.w2s_scale
            self.robot.y += dy / self.w2s_scale
            return

        # if the mouse is within the goal circle:
        gx, gy = self.grid_map.goal
        cx, cy = self.cell_center_screen(gx, gy)
        if _collides_circle((x, y), (cx, cy), self.robot.RADIUS_M * self.w2s_scale):
            # translate goal
            self.grid_map.goal = self.screen_to_cell(x + dx, y + dy)
            self.vector_field_invalidated = True

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.vector_field_invalidated:
            gx_g, gy_g = self.grid_map.goal
            goal_world = self.grid_map.grid_to_world(gx_g, gy_g)

            # recompute gradient
            self.vector_field.compute(
                self.grid_map,
                goal_world,
                repulsion_radius=self.repulsion_radius,
                repulsion_alpha=self.repulsion_alpha,
            )

            self._sprites_gradient = self._build_sprites_gradient()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        """Render the active sprite list based on current view mode."""
        if self._view_gradient:
            self._sprites_gradient.draw()
        else:
            self._sprites_normal.draw()

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

    # ------------------------------------------------------------------
    # Colour bar (gradient view only)
    # ------------------------------------------------------------------

    def _build_colorbar(self) -> tuple[arcade.SpriteList, list]:
        """
        Build a vertical colour-bar widget showing the φ → colour mapping.

        Layout (right side of screen, vertically centred):

            potential φ       ← title
            ─────────────
            1.0 ─┤ red   ┃   ← far from goal
                  ┃  ...  ┃
            0.5 ─┤ blue  ┃
                  ┃  ...  ┃
            0.0 ─┤ green ┃   ← goal
            ─────────────

        The bar is built once as a SpriteList (90 × 2 px strips) and the
        text labels as arcade.Text objects so nothing is re-created each frame.
        """
        BAR_W = 16
        BAR_H = 200
        N_STRIPS = 100  # strips of 2 px each  → 200 px total
        STRIP_H = BAR_H // N_STRIPS
        PAD_RIGHT = 12  # pixels from right screen edge to bar right
        FONT = 10

        bar_right = self.width - PAD_RIGHT
        bar_left = bar_right - BAR_W
        bar_bottom = (self.height - BAR_H) // 2

        # ── Coloured strips ───────────────────────────────────────────
        sl = arcade.SpriteList(use_spatial_hash=False)
        for i in range(N_STRIPS):
            phi = i / (N_STRIPS - 1)  # 0 (bottom) → 1 (top)
            color = _phi_to_color(phi)
            cx = bar_left + BAR_W / 2
            cy = bar_bottom + i * STRIP_H + STRIP_H / 2
            sl.append(
                arcade.SpriteSolidColor(
                    width=BAR_W,
                    height=STRIP_H,
                    center_x=cx,
                    center_y=cy,
                    color=color,
                )
            )

        # ── Tick labels (right-aligned to bar_left − 4) ───────────────
        label_x = bar_left - 4  # labels sit left of the bar
        tick_data = [
            (bar_bottom, "0.0", "goal"),
            (bar_bottom + BAR_H // 2, "0.5", ""),
            (bar_bottom + BAR_H, "1.0", "far"),
        ]

        texts = []
        for y, value, note in tick_data:
            label = f"{value} {note}".strip()
            texts.append(
                arcade.Text(
                    text=label,
                    x=label_x,
                    y=y - FONT // 2,
                    color=(230, 230, 230, 255),
                    font_size=FONT,
                    font_name="monospace",
                    anchor_x="right",
                )
            )

        # ── Title centred above the bar ───────────────────────────────
        texts.append(
            arcade.Text(
                text="potential  \u03c6",
                x=bar_left + BAR_W // 2,
                y=bar_bottom + BAR_H + 14,
                color=(230, 230, 230, 255),
                font_size=FONT,
                font_name="monospace",
                anchor_x="center",
            )
        )

        return sl, texts

    def _draw_colorbar(self) -> None:
        """Draw the background panel, coloured bar, tick labels, and title."""
        BAR_W = 16
        BAR_H = 200
        PAD_RIGHT = 12
        LABEL_W = 70  # approximate width of widest label ("0.0 goal")
        FONT = 10

        bar_right = self.width - PAD_RIGHT
        bar_left = bar_right - BAR_W
        bar_bottom = (self.height - BAR_H) // 2

        # Background panel sized to contain bar + labels + title
        panel_pad = 8
        panel_l = bar_left - LABEL_W - panel_pad
        panel_b = bar_bottom - panel_pad
        panel_w = LABEL_W + BAR_W + panel_pad * 2
        panel_h = BAR_H + FONT + 20 + panel_pad * 2  # bar + title row

        arcade.draw_lbwh_rectangle_filled(
            panel_l,
            panel_b,
            panel_w,
            panel_h,
            (20, 20, 30, 210),
        )

        # Thin outline around the bar itself
        arcade.draw_lbwh_rectangle_outline(
            bar_left,
            bar_bottom,
            BAR_W,
            BAR_H,
            (160, 160, 160, 200),
            border_width=1,
        )

        # Horizontal tick lines at 0, 0.5, 1
        tick_ys = [bar_bottom, bar_bottom + BAR_H // 2, bar_bottom + BAR_H]
        for ty in tick_ys:
            arcade.draw_line(
                bar_left - 4,
                ty,
                bar_left,
                ty,
                (200, 200, 200, 200),
                1,
            )

        self._colorbar_sprites.draw()

        for t in self._colorbar_texts:
            t.draw()

    def _build_hud_texts(self) -> list:
        """
        Create one arcade.Text object per HUD line.
        Created once in __init__, .text updated each frame.
        """
        font_size = 12
        line_h = 18
        pad = 8
        n_lines = 6
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

        view_label = "gradient φ" if self._view_gradient else "normal"
        paused_label = "PAUSED" if self._paused else "running"
        strings = [
            f"pos : ({x_m:.2f} m, {y_m:.2f} m)  cell: ({gx}, {gy})",
            f"\u03b8   : {th_d:+.1f}\u00b0",
            f"v   : {self._last_v:.3f} m/s    \u03c9: {self._last_omega:.3f} rad/s",
            f"solver: {self.solver.NAME:<12}  {paused_label}"
            + (
                f"  [{self.solver.last_status}]"
                if hasattr(self.solver, "last_status")
                else ""
            ),
            f"view  : {view_label}",
            "[ESC] quit  [SPACE] pause  [TAB] solver  [V] view  [Q] UI  [WASD] drive",
        ]

        for t, s in zip(self._hud_texts, strings):
            t.text = s

        # Semi-transparent dark panel — readable over any cell colour
        # while still letting the map show through at the edges.
        # draw_lbwh_rectangle_filled(left, bottom, width, height, color)
        line_h = 18
        pad = 8
        panel_h = len(strings) * line_h + pad * 2
        panel_w = 510
        arcade.draw_lbwh_rectangle_filled(
            0,
            self.height - panel_h,
            panel_w,
            panel_h,
            (20, 20, 30, 200),
        )

        for t in self._hud_texts:
            t.draw()
