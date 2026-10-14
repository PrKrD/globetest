"""Transform Maximenko mean dynamic topography into a globe-ready slope dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, FRONTEND_PUBLIC_DATA_DIR
from app.dyntopo import (
    coordinate_edges,
    equatorial_beta_plane_velocity,
    load_maximenko,
    spherical_gradient,
)
from app.mld import load_mld, maximum_mld, regrid_maximum_mld


GRAVITY_M_PER_S2 = 9.80665
EARTH_ROTATION_RAD_PER_S = 7.2921159e-5
# The curvature relationship is exact on the equator.  With this 0.5° grid,
# a 0.25° core selects only the 0° row; nearby rows blend immediately toward
# the conventional solution instead of creating an artificial tropical stripe.
BETA_PLANE_CORE_LATITUDE_DEG = 0.25
BETA_PLANE_BLEND_LATITUDE_DEG = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Maximenko slope globe dataset")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--mld-input", type=Path, default=DATA_DIR / "mld_dr003_ref10m_v2023.nc")
    parser.add_argument("--output", type=Path, default=FRONTEND_PUBLIC_DATA_DIR / "maximenko_slope.json")
    return parser.parse_args()


def rounded(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def smoothstep(value: np.ndarray) -> np.ndarray:
    """Return a cubic, zero-slope transition from 0 to 1."""
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def main() -> None:
    args = parse_args()
    source = load_maximenko(str(args.input))
    mld_source = load_mld(str(args.mld_input))
    mld_maximum = maximum_mld(mld_source.monthly_m)
    regridded_mld = regrid_maximum_mld(
        mld_maximum.maximum_m,
        mld_maximum.month_count,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        source.longitude_deg,
        source.latitude_deg,
        source.valid,
    )
    grad_east, grad_north, magnitude = spherical_gradient(
        source.height_m, source.longitude_deg, source.latitude_deg
    )
    valid_gradient = np.isfinite(magnitude)
    coriolis = 2.0 * EARTH_ROTATION_RAD_PER_S * np.sin(np.deg2rad(source.latitude_deg))[:, None]
    valid_f_plane = valid_gradient & (np.abs(source.latitude_deg)[:, None] > 0.0)
    velocity_east = np.full_like(grad_east, np.nan)
    velocity_north = np.full_like(grad_north, np.nan)
    np.divide(-GRAVITY_M_PER_S2 * grad_north, coriolis, out=velocity_east, where=valid_f_plane)
    np.divide(GRAVITY_M_PER_S2 * grad_east, coriolis, out=velocity_north, where=valid_f_plane)

    beta_east, beta_north, valid_beta = equatorial_beta_plane_velocity(
        source.height_m, source.longitude_deg, source.latitude_deg
    )
    latitude_abs = np.abs(source.latitude_deg)[:, None]
    blend_weight = np.broadcast_to(
        smoothstep(
            (latitude_abs - BETA_PLANE_CORE_LATITUDE_DEG)
            / (BETA_PLANE_BLEND_LATITUDE_DEG - BETA_PLANE_CORE_LATITUDE_DEG)
        ),
        source.height_m.shape,
    )
    use_beta_only = (latitude_abs <= BETA_PLANE_CORE_LATITUDE_DEG) & valid_beta
    use_blend = (
        (latitude_abs > BETA_PLANE_CORE_LATITUDE_DEG)
        & (latitude_abs < BETA_PLANE_BLEND_LATITUDE_DEG)
        & valid_beta
        & valid_f_plane
    )
    velocity_east[use_beta_only] = beta_east[use_beta_only]
    velocity_north[use_beta_only] = beta_north[use_beta_only]
    velocity_east[use_blend] = (
        (1.0 - blend_weight[use_blend]) * beta_east[use_blend]
        + blend_weight[use_blend] * velocity_east[use_blend]
    )
    velocity_north[use_blend] = (
        (1.0 - blend_weight[use_blend]) * beta_north[use_blend]
        + blend_weight[use_blend] * velocity_north[use_blend]
    )
    valid_velocity = np.isfinite(velocity_east) & np.isfinite(velocity_north)
    velocity_speed = np.hypot(velocity_east, velocity_north)
    # Keep the ordinary-ocean colour scale stable: the experimental values
    # grow rapidly as f approaches zero and would otherwise flatten detail in
    # the rest of the map.
    reference_velocity = velocity_speed[
        np.abs(source.latitude_deg) >= BETA_PLANE_BLEND_LATITUDE_DEG
    ]
    velocity_p98 = float(np.nanpercentile(reference_velocity, 98.0))
    mld_max_p98 = float(np.nanpercentile(regridded_mld.maximum_m[regridded_mld.valid], 98.0))
    lat_edges = coordinate_edges(source.latitude_deg)
    lon_step = float(np.median(np.diff(source.longitude_deg)))

    cells: list[dict[str, float | int]] = []
    n_lon = source.longitude_deg.size
    for i, latitude in enumerate(source.latitude_deg):
        for j, longitude in enumerate(source.longitude_deg):
            is_land = int(not source.valid[i, j])
            has_velocity = int(valid_velocity[i, j])
            speed = float(velocity_speed[i, j]) if has_velocity else 0.0
            equatorial_uncertainty = int(abs(latitude) < BETA_PLANE_BLEND_LATITUDE_DEG)
            equatorial_estimate = int(equatorial_uncertainty and has_velocity)
            has_mld = int(regridded_mld.valid[i, j])
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
                    "equatorial_estimate": equatorial_estimate,
                    "has_mld": has_mld,
                    "mld_max_m": rounded(regridded_mld.maximum_m[i, j], 3) if has_mld else 0.0,
                    "mld_month_count_min": int(regridded_mld.month_count_min[i, j]),
                }
            )

    payload = {
        "dataset_type": "maximenko_geostrophic_velocity",
        "units": {"velocity": "m/s", "mld_max": "m"},
        "resolution_deg": lon_step,
        "n_lat": int(source.latitude_deg.size),
        "n_lon": int(n_lon),
        "velocity_p98_m_per_s": rounded(velocity_p98, 6),
        "mld_max_p98_m": rounded(mld_max_p98, 3),
        "mld_source_resolution_deg": 1.0,
        "mld_temporal_statistic": "maximum_of_available_monthly_climatologies",
        "mld_regridding_method": "periodic_mask_aware_bilinear",
        "equatorial_method": "cubic_spline_beta_plane_blended_to_f_plane",
        "beta_plane_core_latitude_deg": BETA_PLANE_CORE_LATITUDE_DEG,
        "beta_plane_blend_latitude_deg": BETA_PLANE_BLEND_LATITUDE_DEG,
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
    print(f"Maximum MLD p98: {mld_max_p98:.1f} m; valid MLD cells: {int(regridded_mld.valid.sum()):,}")


if __name__ == "__main__":
    main()
