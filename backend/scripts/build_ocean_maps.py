"""Build the globe-ready velocity and maximum mixed-layer-depth maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, FRONTEND_PUBLIC_DATA_DIR
from app.dyntopo import coordinate_edges, load_maximenko
from app.mld import (
    load_mld,
    maximum_mld,
    regrid_maximum_mld,
    regrid_ocean_mask_by_overlap,
)
from app.subduction import geostrophic_subduction_rate
from app.velocity import (
    DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG,
    DEFAULT_BETA_PLANE_CORE_LATITUDE_DEG,
    reconstruct_geostrophic_velocity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the combined ocean velocity and maximum-MLD globe dataset"
    )
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--mld-input", type=Path, default=DATA_DIR / "mld_dr003_ref10m_v2023.nc")
    parser.add_argument(
        "--output",
        type=Path,
        default=FRONTEND_PUBLIC_DATA_DIR / "maximenko_slope.json",
    )
    return parser.parse_args()


def rounded(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


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
    physical_ocean = regrid_ocean_mask_by_overlap(
        mld_source.source_ocean,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        source.longitude_deg,
        source.latitude_deg,
    )
    velocity = reconstruct_geostrophic_velocity(
        source.height_m, source.longitude_deg, source.latitude_deg
    )
    velocity_speed = np.hypot(velocity.east, velocity.north)
    subduction = geostrophic_subduction_rate(
        regridded_mld.maximum_m,
        velocity.east,
        velocity.north,
        source.longitude_deg,
        source.latitude_deg,
        regridded_mld.valid & velocity.valid,
        physical_ocean,
    )
    # Keep the ordinary-ocean colour scale stable: the experimental values
    # grow rapidly as f approaches zero and would otherwise flatten detail in
    # the rest of the map.
    reference_velocity = velocity_speed[
        np.abs(source.latitude_deg) >= DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG
    ]
    velocity_p98 = float(np.nanpercentile(reference_velocity, 98.0))
    mld_max_p98 = float(np.nanpercentile(regridded_mld.maximum_m[regridded_mld.valid], 98.0))
    subduction_abs_p98 = float(
        np.nanpercentile(np.abs(subduction.rate_m_per_s[subduction.valid]), 98.0)
    )
    lat_edges = coordinate_edges(source.latitude_deg)
    lon_step = float(np.median(np.diff(source.longitude_deg)))

    cells: list[dict[str, float | int]] = []
    n_lon = source.longitude_deg.size
    for i, latitude in enumerate(source.latitude_deg):
        for j, longitude in enumerate(source.longitude_deg):
            is_land = int(not source.valid[i, j])
            has_velocity = int(velocity.valid[i, j])
            speed = float(velocity_speed[i, j]) if has_velocity else 0.0
            equatorial_uncertainty = int(
                abs(latitude) < DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG
            )
            equatorial_estimate = int(equatorial_uncertainty and has_velocity)
            has_mld = int(regridded_mld.valid[i, j])
            has_subduction = int(subduction.valid[i, j])
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
                    "has_subduction": has_subduction,
                    "subduction_m_per_s": rounded(subduction.rate_m_per_s[i, j], 12)
                    if has_subduction
                    else 0.0,
                }
            )

    payload = {
        "dataset_type": "maximenko_ocean_diagnostics",
        "units": {"velocity": "m/s", "mld_max": "m", "subduction": "m/s"},
        "resolution_deg": lon_step,
        "n_lat": int(source.latitude_deg.size),
        "n_lon": int(n_lon),
        "velocity_p98_m_per_s": rounded(velocity_p98, 6),
        "mld_max_p98_m": rounded(mld_max_p98, 3),
        "subduction_abs_p98_m_per_s": rounded(subduction_abs_p98, 12),
        "mld_source_resolution_deg": 1.0,
        "mld_temporal_statistic": "maximum_of_available_monthly_climatologies",
        "mld_regridding_method": "periodic_mask_aware_bilinear",
        "equatorial_method": "cubic_spline_beta_plane_blended_to_f_plane",
        "subduction_method": (
            "conservative_spherical_finite_volume_of_mld_times_geostrophic_velocity"
        ),
        "subduction_face_transport": "arithmetic_mean_Hu_Hv_on_shared_wet_faces",
        "subduction_boundary_policy": {
            "physical_coastline": "zero_normal_flux",
            "ocean_missing_data": "wet_side_transport",
            "north_south_domain_edge": "wet_side_transport",
        },
        "physical_ocean_mask_method": "mld_source_mask_spherical_overlap_majority_ties_ocean",
        "subduction_conservation": {
            "area_integrated_m3_per_s": rounded(
                subduction.conservation.area_integrated_subduction_m3_per_s, 3
            ),
            "outward_boundary_m3_per_s": rounded(
                subduction.conservation.total_outward_m3_per_s, 3
            ),
            "closure_residual_m3_per_s": rounded(
                subduction.conservation.closure_residual_m3_per_s, 6
            ),
        },
        "beta_plane_core_latitude_deg": DEFAULT_BETA_PLANE_CORE_LATITUDE_DEG,
        "beta_plane_blend_latitude_deg": DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG,
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
    print(
        f"Velocity p98: {velocity_p98:.3f} m/s; "
        f"valid velocity cells: {int(velocity.valid.sum()):,}"
    )
    print(
        f"Maximum MLD p98: {mld_max_p98:.1f} m; "
        f"valid MLD cells: {int(regridded_mld.valid.sum()):,}"
    )
    print(
        f"Subduction |p98|: {subduction_abs_p98:.3e} m/s; "
        f"valid subduction cells: {int(subduction.valid.sum()):,}"
    )
    print(
        "Subduction conservation residual: "
        f"{subduction.conservation.closure_residual_m3_per_s:.3e} m^3/s"
    )


if __name__ == "__main__":
    main()
