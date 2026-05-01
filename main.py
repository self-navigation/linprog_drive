"""
LP OCP Robot Navigation Simulator
Entry point — parses CLI arguments and launches the arcade window.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Educational LP-based OCP robot navigation simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Map file format (text):
  size <cols> <rows>          grid dimensions (required, must come first)
  cell <metres>               metres per cell (default: 1.0)
  obstacle <x1,y1> <x2,y2>   obstacle line between two cells (0-indexed)
  start <x> <y>               robot start cell
  goal  <x> <y>               goal cell
  # ...                       comment lines

Example:
  size 30 20
  cell 0.5
  obstacle 10,2 10,15
  obstacle 20,5 20,18
  start 2 2
  goal 27 17
        """,
    )
    parser.add_argument(
        "map",
        metavar="MAP_FILE",
        help="Path to the text map definition file.",
    )
    parser.add_argument(
        "--cell-px",
        type=int,
        default=4,
        metavar="PX",
        help="Screen pixels per grid cell (default: 24).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        metavar="FPS",
        help="Target simulation frame rate (default: 60).",
    )
    parser.add_argument(
        "--repulsion-radius",
        type=float,
        default=0.6,
        metavar="M",
        help="Obstacle repulsion radius in metres (default: 0.6).",
    )
    parser.add_argument(
        "--repulsion-alpha",
        type=float,
        default=5.0,
        metavar="A",
        help="Repulsion cost weight at obstacle boundary (default: 5.0, 0=off).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here so argparse --help works without arcade installed
    try:
        import arcade  # noqa: F401
    except ImportError:
        print("ERROR: 'arcade' is not installed.  Run:  pip install arcade")
        sys.exit(1)

    try:
        from map import load_map_from_bitmap, load_map
    except ImportError as e:
        print(f"ERROR: could not import map module: {e}")
        sys.exit(1)

    # Load and validate map first (fast, gives clear errors before opening window)
    try:
        if Path(args.map).suffix == ".map":
            grid_map = load_map(args.map)
        else:
            grid_map = load_map_from_bitmap(args.map)
    except FileNotFoundError:
        print(f"ERROR: map file not found: {args.map!r}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: invalid map file: {e}")
        sys.exit(1)

    print(
        f"Map loaded: {grid_map.cols}×{grid_map.rows} cells, "
        f"{grid_map.cell_size} m/cell"
    )
    print(f"Start: {grid_map.start}  Goal: {grid_map.goal}")

    from simulation import Simulation

    sim = Simulation(
        grid_map,
        cell_px=args.cell_px,
        repulsion_radius=args.repulsion_radius,
        repulsion_alpha=args.repulsion_alpha,
    )
    sim.set_update_rate(1.0 / args.fps)
    arcade.run()


if __name__ == "__main__":
    main()
