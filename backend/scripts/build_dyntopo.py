"""Transform Maximenko mean dynamic topography into a globe-ready slope dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, FRONTEND_PUBLIC_DATA_DIR
from app.dyntopo import coordinate_edges, load_maximenko, spherical_gradient


GRAVITY_M_PER_S2 = 9.80665
EARTH_ROTATION_RAD_PER_S = 7.2921159e-5
MIN_GEOSTROPHIC_LATITUDE_DEG = 2.0
EQUATORIAL_UNCERTAINTY_LATITUDE_DEG = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Maximenko slope globe dataset")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--output", type=Path, default=FRONTEND_PUBLIC_DATA_DIR / "maximenko_slope.json")
    return parser.parse_args()


def rounded(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def main() -> None:
    args = parse_args()
    source = load_maximenko(str(args.input))
    grad_east, grad_north, magnitude = spherical_gradient(
        source.height_m, source.longitude_deg, source.latitude_deg
    )
    valid_gradient = np.isfinite(magnitude)
    coriolis = 2.0 * EARTH_ROTATION_RAD_PER_S * np.sin(np.deg2rad(source.latitude_deg))[:, None]
    valid_velocity = valid_gradient & (np.abs(source.latitude_deg)[:, None] >= MIN_GEOSTROPHIC_LATITUDE_DEG)
    velocity_east = np.full_like(grad_east, np.nan)
    velocity_north = np.full_like(grad_north, np.nan)
    np.divide(-GRAVITY_M_PER_S2 * grad_north, coriolis, out=velocity_east, where=valid_velocity)
    np.divide(GRAVITY_M_PER_S2 * grad_east, coriolis, out=velocity_north, where=valid_velocity)
    velocity_speed = np.hypot(velocity_east, velocity_north)
    velocity_p98 = float(np.nanpercentile(velocity_speed, 98.0))
    lat_edges = coordinate_edges(source.latitude_deg)
    lon_step = float(np.median(np.diff(source.longitude_deg)))

    cells: list[dict[str, float | int]] = []
    n_lon = source.longitude_deg.size
    for i, latitude in enumerate(source.latitude_deg):
        for j, longitude in enumerate(source.longitude_deg):
            is_land = int(not source.valid[i, j])
            has_velocity = int(valid_velocity[i, j])
            speed = float(velocity_speed[i, j]) if has_velocity else 0.0
            equatorial_uncertainty = int(abs(latitude) < EQUATORIAL_UNCERTAINTY_LATITUDE_DEG)
            cells.append(
                {
                    "id": i * n_lon + j,
                    "lat0": rounded(lat_edges[i], 4),
                    "lat1": rounded(lat_edges[i + 1], 4),
                    "lon0": rounded(longitude, 4),
                    "lon1": rounded(longitude + lon_step, 4),
                    "clat": rounded(latitude, 4),
                    "clon": rounded(longitude, 4),
                    "is_land": is_land,
                    "has_velocity": has_velocity,
                    "velocity_speed_m_per_s": rounded(speed, 6),
                    "equatorial_uncertainty": equatorial_uncertainty,
                }
            )

    payload = {
        "dataset_type": "maximenko_geostrophic_velocity",
        "units": {"velocity": "m/s"},
        "resolution_deg": lon_step,
        "n_lat": int(source.latitude_deg.size),
        "n_lon": int(n_lon),
        "velocity_p98_m_per_s": rounded(velocity_p98, 6),
        "min_geostrophic_latitude_deg": MIN_GEOSTROPHIC_LATITUDE_DEG,
        "equatorial_uncertainty_latitude_deg": EQUATORIAL_UNCERTAINTY_LATITUDE_DEG,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the currently served product intact until the full replacement has
    # been serialized. This matters because the browser can fetch this file
    # while a local rebuild is running.
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary_output.replace(args.output)
    print(f"Wrote {len(cells):,} cells to {args.output}")
    print(f"Native grid: {source.latitude_deg.size} x {n_lon} at {lon_step:g}°")
    print(f"Velocity p98: {velocity_p98:.3f} m/s; valid velocity cells: {int(valid_velocity.sum()):,}")


if __name__ == "__main__":
    main()
