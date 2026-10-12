"""Export layer for grid products.

Prototype format choice:
- Underlying 0.25° grid: compact binary mask + compact metadata JSON.
- Frontend preview grid: coarse JSON (small enough for Phase 2 rendering).

This keeps export replaceable for future Zarr/NetCDF/tiled services.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .grid import GridDefinition, build_global_grid


def export_binary_mask(grid: GridDefinition, is_land: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_land.shape != (grid.n_lat, grid.n_lon):
        raise ValueError("Mask shape does not match grid dimensions")

    mask_path = output_dir / "mask_u8.bin"
    meta_path = output_dir / "grid_meta.json"

    is_land.astype(np.uint8).tofile(mask_path)

    metadata = {
        "resolution_deg": grid.resolution_deg,
        "n_lat": grid.n_lat,
        "n_lon": grid.n_lon,
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": -180.0,
        "lon_max": 180.0,
        "indexing": {
            "order": "row-major",
            "cell_id_formula": "cell_id = lat_index * n_lon + lon_index",
            "mask_values": {"0": "ocean", "1": "land"},
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return mask_path, meta_path


def _coarsen_mask(is_land: np.ndarray, factor: int) -> np.ndarray:
    n_lat, n_lon = is_land.shape
    if n_lat % factor != 0 or n_lon % factor != 0:
        raise ValueError("Coarse resolution factor must divide fine grid dimensions")

    reshaped = is_land.reshape(n_lat // factor, factor, n_lon // factor, factor)
    land_fraction = reshaped.mean(axis=(1, 3))
    return (land_fraction >= 0.5).astype(np.uint8)


def export_coarse_grid_json(
    fine_grid: GridDefinition,
    fine_is_land: np.ndarray,
    coarse_resolution_deg: float,
    output_path: Path,
) -> Path:
    factor_raw = coarse_resolution_deg / fine_grid.resolution_deg
    factor = int(round(factor_raw))
    if not np.isclose(factor_raw, factor):
        raise ValueError("Coarse resolution must be an integer multiple of fine resolution")

    coarse_grid = build_global_grid(coarse_resolution_deg)
    coarse_land = _coarsen_mask(fine_is_land.astype(np.uint8), factor)

    cells = []
    for i in range(coarse_grid.n_lat):
        lat0 = float(coarse_grid.lat_edges[i])
        lat1 = float(coarse_grid.lat_edges[i + 1])
        clat = float(coarse_grid.lat_centers[i])
        for j in range(coarse_grid.n_lon):
            lon0 = float(coarse_grid.lon_edges[j])
            lon1 = float(coarse_grid.lon_edges[j + 1])
            cell_id = int(coarse_grid.cell_ids[i, j])
            cells.append(
                {
                    "id": cell_id,
                    "lat0": lat0,
                    "lat1": lat1,
                    "lon0": lon0,
                    "lon1": lon1,
                    "clat": clat,
                    "clon": float(coarse_grid.lon_centers[j]),
                    "is_land": int(coarse_land[i, j]),
                }
            )

    payload = {
        "resolution_deg": coarse_grid.resolution_deg,
        "n_lat": coarse_grid.n_lat,
        "n_lon": coarse_grid.n_lon,
        "cells": cells,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path
