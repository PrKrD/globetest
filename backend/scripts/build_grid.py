"""Build global grid and land/ocean mask artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import (
    COARSE_RENDER_RESOLUTION_DEG,
    DATA_DIR,
    FRONTEND_PUBLIC_DATA_DIR,
    GRID_RESOLUTION_DEG,
)
from app.export import export_binary_mask, export_coarse_grid_json
from app.grid import build_global_grid
from app.mask import build_land_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global ocean grid datasets")
    parser.add_argument("--resolution", type=float, default=GRID_RESOLUTION_DEG)
    parser.add_argument("--coarse-resolution", type=float, default=COARSE_RENDER_RESOLUTION_DEG)
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--frontend-output",
        type=str,
        default=str(FRONTEND_PUBLIC_DATA_DIR / f"coarse_grid_{COARSE_RENDER_RESOLUTION_DEG:.1f}deg.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid = build_global_grid(args.resolution)
    mask = build_land_mask(grid, DATA_DIR)

    output_dir = Path(args.output_dir)
    frontend_output = Path(args.frontend_output)

    mask_path, meta_path = export_binary_mask(grid, mask, output_dir)
    coarse_path = export_coarse_grid_json(
        fine_grid=grid,
        fine_is_land=mask,
        coarse_resolution_deg=args.coarse_resolution,
        output_path=frontend_output,
    )

    ocean_cells = int((~mask).sum())
    land_cells = int(mask.sum())
    print(f"Built grid @ {grid.resolution_deg}°: {grid.n_lat} x {grid.n_lon} = {grid.n_lat * grid.n_lon:,} cells")
    print(f"Land cells: {land_cells:,} | Ocean cells: {ocean_cells:,}")
    print(f"Wrote binary mask: {mask_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Wrote coarse frontend grid: {coarse_path}")


if __name__ == "__main__":
    main()
