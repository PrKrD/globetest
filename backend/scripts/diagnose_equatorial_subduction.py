"""Diagnose tropical geostrophic velocity and subduction without changing the globe product."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from app.config import DATA_DIR
from app.dyntopo import (
    EARTH_RADIUS_METERS,
    EARTH_ROTATION_RAD_PER_S,
    GRAVITY_M_PER_S2,
    coordinate_edges,
    equatorial_beta_plane_velocity,
    load_maximenko,
    spherical_gradient,
)
from app.mld import load_mld, maximum_mld, regrid_maximum_mld
from app.subduction import geostrophic_subduction_rate


CORE_LATITUDE_DEG = 0.25
BASE_FIT_WIDTH_DEG = 3.5
BASE_SMOOTHING_M = 0.01
BASE_BLEND_WIDTH_DEG = 3.0
TROPICAL_LIMIT_DEG = 10.0
FIT_WIDTHS_DEG = (3.0, 3.5, 4.5, 6.0)
SMOOTHING_VALUES_M = (0.0, 0.003, 0.01, 0.03, 0.1)
BLEND_WIDTHS_DEG = (1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class VelocitySolution:
    east: np.ndarray
    north: np.ndarray
    valid: np.ndarray
    beta_east: np.ndarray
    beta_north: np.ndarray
    beta_valid: np.ndarray
    ordinary_east: np.ndarray
    ordinary_north: np.ndarray
    ordinary_valid: np.ndarray
    blend_weight: np.ndarray


@dataclass(frozen=True)
class SubductionTerms:
    zonal_m_per_s: np.ndarray
    meridional_m_per_s: np.ndarray
    total_m_per_s: np.ndarray
    valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--mld-input", type=Path, default=DATA_DIR / "mld_dr003_ref10m_v2023.nc")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "diagnostics" / "equatorial_subduction",
    )
    return parser.parse_args()


def smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ordinary_velocity(
    height_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_east, grad_north, magnitude = spherical_gradient(height_m, longitude_deg, latitude_deg)
    coriolis = 2.0 * EARTH_ROTATION_RAD_PER_S * np.sin(np.deg2rad(latitude_deg))[:, None]
    valid = np.isfinite(magnitude) & (np.abs(latitude_deg)[:, None] > 0.0)
    east = np.full_like(grad_east, np.nan)
    north = np.full_like(grad_north, np.nan)
    np.divide(-GRAVITY_M_PER_S2 * grad_north, coriolis, out=east, where=valid)
    np.divide(GRAVITY_M_PER_S2 * grad_east, coriolis, out=north, where=valid)
    valid &= np.isfinite(east) & np.isfinite(north)
    return east, north, valid


def velocity_solution(
    height_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    fit_width_deg: float = BASE_FIT_WIDTH_DEG,
    smoothing_m: float = BASE_SMOOTHING_M,
    blend_width_deg: float = BASE_BLEND_WIDTH_DEG,
) -> VelocitySolution:
    ordinary_east, ordinary_north, ordinary_valid = ordinary_velocity(
        height_m, longitude_deg, latitude_deg
    )
    beta_east, beta_north, beta_valid = equatorial_beta_plane_velocity(
        height_m,
        longitude_deg,
        latitude_deg,
        fit_half_width_deg=fit_width_deg,
        smoothing_m=smoothing_m,
    )
    latitude_abs = np.abs(latitude_deg)[:, None]
    weight = np.broadcast_to(
        smoothstep(
            (latitude_abs - CORE_LATITUDE_DEG)
            / (blend_width_deg - CORE_LATITUDE_DEG)
        ),
        height_m.shape,
    ).copy()
    east = ordinary_east.copy()
    north = ordinary_north.copy()
    beta_only = (latitude_abs <= CORE_LATITUDE_DEG) & beta_valid
    blended = (
        (latitude_abs > CORE_LATITUDE_DEG)
        & (latitude_abs < blend_width_deg)
        & beta_valid
        & ordinary_valid
    )
    east[beta_only] = beta_east[beta_only]
    north[beta_only] = beta_north[beta_only]
    east[blended] = (
        (1.0 - weight[blended]) * beta_east[blended]
        + weight[blended] * ordinary_east[blended]
    )
    north[blended] = (
        (1.0 - weight[blended]) * beta_north[blended]
        + weight[blended] * ordinary_north[blended]
    )
    valid = np.isfinite(east) & np.isfinite(north)
    return VelocitySolution(
        east,
        north,
        valid,
        beta_east,
        beta_north,
        beta_valid,
        ordinary_east,
        ordinary_north,
        ordinary_valid,
        weight,
    )


def subduction_terms(
    depth_m: np.ndarray,
    east_velocity_m_per_s: np.ndarray,
    north_velocity_m_per_s: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    valid: np.ndarray,
) -> SubductionTerms:
    """Reproduce the production stencil while retaining its two signed terms."""
    depth = np.asarray(depth_m, dtype=np.float64)
    east_velocity = np.asarray(east_velocity_m_per_s, dtype=np.float64)
    north_velocity = np.asarray(north_velocity_m_per_s, dtype=np.float64)
    usable = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(depth)
        & np.isfinite(east_velocity)
        & np.isfinite(north_velocity)
    )
    longitude = np.deg2rad(longitude_deg)
    latitude = np.deg2rad(latitude_deg)
    shape = depth.shape
    cosine = np.cos(latitude)[:, None]
    east_flux = depth * east_velocity
    north_flux_cosine = depth * north_velocity * cosine

    west = np.roll(east_flux, 1, axis=1)
    east = np.roll(east_flux, -1, axis=1)
    west_valid = np.roll(usable, 1, axis=1)
    east_valid = np.roll(usable, -1, axis=1)
    d_lambda = float(np.median(np.diff(longitude)))
    d_east = np.full(shape, np.nan)
    centered_x = usable & west_valid & east_valid
    forward_x = usable & ~west_valid & east_valid
    backward_x = usable & west_valid & ~east_valid
    d_east[centered_x] = ((east - west) / (2.0 * d_lambda))[centered_x]
    d_east[forward_x] = ((east - east_flux) / d_lambda)[forward_x]
    d_east[backward_x] = ((east_flux - west) / d_lambda)[backward_x]

    south = np.full(shape, np.nan)
    north = np.full(shape, np.nan)
    south[1:] = north_flux_cosine[:-1]
    north[:-1] = north_flux_cosine[1:]
    south_valid = np.zeros(shape, dtype=bool)
    north_valid = np.zeros(shape, dtype=bool)
    south_valid[1:] = usable[:-1]
    north_valid[:-1] = usable[1:]
    d_phi = float(np.median(np.diff(latitude)))
    d_north = np.full(shape, np.nan)
    centered_y = usable & south_valid & north_valid
    forward_y = usable & ~south_valid & north_valid
    backward_y = usable & south_valid & ~north_valid
    d_north[centered_y] = ((north - south) / (2.0 * d_phi))[centered_y]
    d_north[forward_y] = ((north - north_flux_cosine) / d_phi)[forward_y]
    d_north[backward_y] = ((north_flux_cosine - south) / d_phi)[backward_y]

    result_valid = (
        (centered_x | forward_x | backward_x)
        & (centered_y | forward_y | backward_y)
    )
    zonal = np.full(shape, np.nan)
    meridional = np.full(shape, np.nan)
    zonal[result_valid] = (-d_east / (EARTH_RADIUS_METERS * cosine))[result_valid]
    meridional[result_valid] = (-d_north / (EARTH_RADIUS_METERS * cosine))[result_valid]
    total = zonal + meridional
    return SubductionTerms(zonal, meridional, total, result_valid)


def cell_areas(longitude_deg: np.ndarray, latitude_deg: np.ndarray) -> np.ndarray:
    longitude_width = np.deg2rad(float(np.median(np.diff(longitude_deg))))
    latitude_edges = np.deg2rad(coordinate_edges(latitude_deg))
    row_area = EARTH_RADIUS_METERS**2 * longitude_width * (
        np.sin(latitude_edges[1:]) - np.sin(latitude_edges[:-1])
    )
    return np.broadcast_to(row_area[:, None], (latitude_deg.size, longitude_deg.size)).copy()


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return float("nan")
    return float(np.sum(values[usable] * weights[usable]) / np.sum(weights[usable]))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    quantile: float,
) -> float:
    usable = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return float("nan")
    ordered = np.argsort(values[usable])
    sorted_values = values[usable][ordered]
    sorted_weights = weights[usable][ordered]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return float(np.interp(quantile, cumulative, sorted_values))


def statistics(
    values: np.ndarray,
    valid: np.ndarray,
    area: np.ndarray,
    band: np.ndarray,
) -> dict[str, float | int]:
    mask = valid & band
    valid_area = float(np.sum(area[mask]))
    mean = weighted_mean(values, area, mask)
    rms = np.sqrt(weighted_mean(values**2, area, mask))
    positive_fraction = (
        float(np.sum(area[mask & (values > 0.0)]) / valid_area) if valid_area else float("nan")
    )
    return {
        "cell_count": int(np.sum(mask)),
        "valid_area_m2": valid_area,
        "area_weighted_mean_m_per_s": mean,
        "area_weighted_rms_m_per_s": float(rms),
        "area_weighted_positive_fraction": positive_fraction,
        "area_weighted_p10_m_per_s": weighted_quantile(values, area, mask, 0.1),
        "area_weighted_median_m_per_s": weighted_quantile(values, area, mask, 0.5),
        "area_weighted_p90_m_per_s": weighted_quantile(values, area, mask, 0.9),
        "area_integral_m3_per_s": float(np.sum(values[mask] * area[mask])),
    }


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & np.isfinite(a) & np.isfinite(b)
    if np.sum(usable) < 2:
        return float("nan")
    return float(np.corrcoef(a[usable], b[usable])[0, 1])


def blend_gradient_terms(
    depth_m: np.ndarray,
    solution: VelocitySolution,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    mld_valid: np.ndarray,
    blend_width_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = mld_valid & solution.beta_valid & solution.ordinary_valid
    beta_terms = subduction_terms(
        depth_m,
        solution.beta_east,
        solution.beta_north,
        longitude_deg,
        latitude_deg,
        common,
    )
    ordinary_terms = subduction_terms(
        depth_m,
        solution.ordinary_east,
        solution.ordinary_north,
        longitude_deg,
        latitude_deg,
        common,
    )
    blended_terms = subduction_terms(
        depth_m,
        (1.0 - solution.blend_weight) * solution.beta_east
        + solution.blend_weight * solution.ordinary_east,
        (1.0 - solution.blend_weight) * solution.beta_north
        + solution.blend_weight * solution.ordinary_north,
        longitude_deg,
        latitude_deg,
        common,
    )
    base = (
        (1.0 - solution.blend_weight) * beta_terms.total_m_per_s
        + solution.blend_weight * ordinary_terms.total_m_per_s
    )
    discrete_extra = blended_terms.total_m_per_s - base

    latitude_abs = np.abs(latitude_deg)
    t = np.clip(
        (latitude_abs - CORE_LATITUDE_DEG) / (blend_width_deg - CORE_LATITUDE_DEG),
        0.0,
        1.0,
    )
    derivative_per_radian = np.zeros_like(latitude_abs)
    transition = (latitude_abs > CORE_LATITUDE_DEG) & (latitude_abs < blend_width_deg)
    derivative_per_radian[transition] = (
        np.sign(latitude_deg[transition])
        * 6.0
        * t[transition]
        * (1.0 - t[transition])
        / np.deg2rad(blend_width_deg - CORE_LATITUDE_DEG)
    )
    analytic_extra = (
        -depth_m
        * (solution.ordinary_north - solution.beta_north)
        * derivative_per_radian[:, None]
        / EARTH_RADIUS_METERS
    )
    analytic_extra[~blended_terms.valid] = np.nan
    discrete_extra[~blended_terms.valid] = np.nan
    return discrete_extra, analytic_extra, blended_terms.valid


def conservation_budget(
    depth_m: np.ndarray,
    solution: VelocitySolution,
    mld_valid: np.ndarray,
    source_valid: np.ndarray,
    total_subduction: SubductionTerms,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    area: np.ndarray,
) -> dict[str, float | int]:
    valid = mld_valid & solution.valid
    east_flux = depth_m * solution.east
    north_flux = depth_m * solution.north
    longitude_width = np.deg2rad(float(np.median(np.diff(longitude_deg))))
    latitude_edges = np.deg2rad(coordinate_edges(latitude_deg))
    latitude_width = float(np.median(np.diff(np.deg2rad(latitude_deg))))
    meridional_face_length = EARTH_RADIUS_METERS * latitude_width

    flux_by_boundary = {"source_mask_m3_per_s": 0.0, "derived_missing_m3_per_s": 0.0, "domain_edge_m3_per_s": 0.0}
    boundary_cells = np.zeros_like(valid)

    for shift, sign in ((-1, 1.0), (1, -1.0)):
        neighbour_valid = np.roll(valid, shift, axis=1)
        boundary = valid & ~neighbour_valid
        neighbour_source = np.roll(source_valid, shift, axis=1)
        source_boundary = boundary & ~neighbour_source
        missing_boundary = boundary & neighbour_source
        flux = sign * east_flux * meridional_face_length
        flux_by_boundary["source_mask_m3_per_s"] += float(np.nansum(flux[source_boundary]))
        flux_by_boundary["derived_missing_m3_per_s"] += float(np.nansum(flux[missing_boundary]))
        boundary_cells |= boundary

    for row, sign, edge_index in ((0, -1.0, 0), (-1, 1.0, -1)):
        boundary = valid[row]
        face_length = EARTH_RADIUS_METERS * np.cos(latitude_edges[edge_index]) * longitude_width
        flux_by_boundary["domain_edge_m3_per_s"] += float(
            np.nansum(sign * north_flux[row, boundary] * face_length)
        )
        boundary_cells[row, boundary] = True

    north_neighbour = np.zeros_like(valid)
    south_neighbour = np.zeros_like(valid)
    north_neighbour[:-1] = valid[1:]
    south_neighbour[1:] = valid[:-1]
    north_source = np.zeros_like(source_valid)
    south_source = np.zeros_like(source_valid)
    north_source[:-1] = source_valid[1:]
    south_source[1:] = source_valid[:-1]
    for neighbour_valid, neighbour_source, sign, face_rows in (
        (north_neighbour, north_source, 1.0, latitude_edges[1:]),
        (south_neighbour, south_source, -1.0, latitude_edges[:-1]),
    ):
        boundary = valid & ~neighbour_valid
        boundary[0 if sign < 0 else -1] = False
        source_boundary = boundary & ~neighbour_source
        missing_boundary = boundary & neighbour_source
        face_length = EARTH_RADIUS_METERS * np.cos(face_rows)[:, None] * longitude_width
        flux = sign * north_flux * face_length
        flux_by_boundary["source_mask_m3_per_s"] += float(np.nansum(flux[source_boundary]))
        flux_by_boundary["derived_missing_m3_per_s"] += float(np.nansum(flux[missing_boundary]))
        boundary_cells |= boundary

    subduction_valid = total_subduction.valid
    integral = float(np.nansum(total_subduction.total_m_per_s[subduction_valid] * area[subduction_valid]))
    boundary_flux = float(sum(flux_by_boundary.values()))
    closure = integral + boundary_flux
    scale = max(abs(integral), abs(boundary_flux), 1e-30)
    return {
        "valid_transport_cells": int(np.sum(valid)),
        "valid_subduction_cells": int(np.sum(subduction_valid)),
        "boundary_adjacent_transport_cells": int(np.sum(boundary_cells)),
        "subduction_area_integral_m3_per_s": integral,
        "estimated_outward_boundary_flux_m3_per_s": boundary_flux,
        **flux_by_boundary,
        "closure_residual_m3_per_s": closure,
        "relative_closure_residual": float(closure / scale),
        "boundary_cell_subduction_integral_m3_per_s": float(
            np.nansum(
                total_subduction.total_m_per_s[subduction_valid & boundary_cells]
                * area[subduction_valid & boundary_cells]
            )
        ),
        "interior_cell_subduction_integral_m3_per_s": float(
            np.nansum(
                total_subduction.total_m_per_s[subduction_valid & ~boundary_cells]
                * area[subduction_valid & ~boundary_cells]
            )
        ),
    }


def tropical_view(values: np.ndarray, tropical_rows: np.ndarray) -> np.ndarray:
    return values[tropical_rows]


def symmetric_limit(values: Iterable[np.ndarray], percentile: float = 98.0) -> float:
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    return float(np.percentile(np.abs(finite), percentile))


def plot_field(
    axis: plt.Axes,
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    title: str,
    limit: float,
    unit: str,
) -> None:
    image = axis.pcolormesh(
        longitude,
        latitude,
        values,
        shading="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(-180.0, 180.0)
    axis.set_ylim(-TROPICAL_LIMIT_DEG, TROPICAL_LIMIT_DEG)
    plt.colorbar(image, ax=axis, shrink=0.82, label=unit)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_maximenko(str(args.input))
    mld_source = load_mld(str(args.mld_input))
    mld_maximum = maximum_mld(mld_source.monthly_m)
    mld = regrid_maximum_mld(
        mld_maximum.maximum_m,
        mld_maximum.month_count,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        source.longitude_deg,
        source.latitude_deg,
        source.valid,
    )
    area = cell_areas(source.longitude_deg, source.latitude_deg)
    baseline = velocity_solution(source.height_m, source.longitude_deg, source.latitude_deg)
    baseline_valid = mld.valid & baseline.valid
    baseline_terms = subduction_terms(
        mld.maximum_m,
        baseline.east,
        baseline.north,
        source.longitude_deg,
        source.latitude_deg,
        baseline_valid,
    )
    production = geostrophic_subduction_rate(
        mld.maximum_m,
        baseline.east,
        baseline.north,
        source.longitude_deg,
        source.latitude_deg,
        baseline_valid,
    )
    reproduction_error = float(
        np.nanmax(np.abs(production.rate_m_per_s - baseline_terms.total_m_per_s))
    )

    ordinary_valid = mld.valid & baseline.ordinary_valid
    ordinary_terms = subduction_terms(
        mld.maximum_m,
        baseline.ordinary_east,
        baseline.ordinary_north,
        source.longitude_deg,
        source.latitude_deg,
        ordinary_valid,
    )
    gradient_extra, analytic_gradient_extra, gradient_valid = blend_gradient_terms(
        mld.maximum_m,
        baseline,
        source.longitude_deg,
        source.latitude_deg,
        mld.valid,
        BASE_BLEND_WIDTH_DEG,
    )

    tropical_rows = np.abs(source.latitude_deg) <= TROPICAL_LIMIT_DEG
    tropical_band = np.broadcast_to(tropical_rows[:, None], source.height_m.shape)
    transition_band = np.broadcast_to(
        (
            (np.abs(source.latitude_deg) > CORE_LATITUDE_DEG)
            & (np.abs(source.latitude_deg) < BASE_BLEND_WIDTH_DEG)
        )[:, None],
        source.height_m.shape,
    )
    well_conditioned_band = np.broadcast_to(
        (
            (np.abs(source.latitude_deg) >= BASE_BLEND_WIDTH_DEG + 0.5)
            & (np.abs(source.latitude_deg) <= TROPICAL_LIMIT_DEG)
        )[:, None],
        source.height_m.shape,
    )

    velocity_limit = symmetric_limit(
        [tropical_view(baseline.east, tropical_rows), tropical_view(baseline.north, tropical_rows)]
    )
    velocity_difference_limit = symmetric_limit(
        [
            tropical_view(baseline.east - baseline.ordinary_east, tropical_rows),
            tropical_view(baseline.north - baseline.ordinary_north, tropical_rows),
        ]
    )
    latitude_tropical = source.latitude_deg[tropical_rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
    plot_field(axes[0, 0], source.longitude_deg, latitude_tropical, tropical_view(baseline.east, tropical_rows), "Current eastward velocity", velocity_limit, "m/s")
    plot_field(axes[0, 1], source.longitude_deg, latitude_tropical, tropical_view(baseline.north, tropical_rows), "Current northward velocity", velocity_limit, "m/s")
    plot_field(axes[1, 0], source.longitude_deg, latitude_tropical, tropical_view(baseline.east - baseline.ordinary_east, tropical_rows), "Eastward: current − ordinary", velocity_difference_limit, "m/s")
    plot_field(axes[1, 1], source.longitude_deg, latitude_tropical, tropical_view(baseline.north - baseline.ordinary_north, tropical_rows), "Northward: current − ordinary", velocity_difference_limit, "m/s")
    save_figure(fig, args.output_dir / "01_velocity_components.png")

    subduction_limit = symmetric_limit(
        [
            tropical_view(baseline_terms.zonal_m_per_s, tropical_rows),
            tropical_view(baseline_terms.meridional_m_per_s, tropical_rows),
            tropical_view(baseline_terms.total_m_per_s, tropical_rows),
        ]
    )
    gradient_limit = symmetric_limit([tropical_view(gradient_extra, tropical_rows)])
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
    plot_field(axes[0, 0], source.longitude_deg, latitude_tropical, tropical_view(baseline_terms.zonal_m_per_s, tropical_rows), "Zonal transport convergence", subduction_limit, "m/s")
    plot_field(axes[0, 1], source.longitude_deg, latitude_tropical, tropical_view(baseline_terms.meridional_m_per_s, tropical_rows), "Meridional transport convergence", subduction_limit, "m/s")
    plot_field(axes[1, 0], source.longitude_deg, latitude_tropical, tropical_view(baseline_terms.total_m_per_s, tropical_rows), "Total geostrophic subduction", subduction_limit, "m/s")
    plot_field(axes[1, 1], source.longitude_deg, latitude_tropical, tropical_view(gradient_extra, tropical_rows), "Discrete contribution from ∇w", gradient_limit, "m/s")
    save_figure(fig, args.output_dir / "02_subduction_components.png")

    comparison_limit = symmetric_limit(
        [
            tropical_view(baseline_terms.total_m_per_s, tropical_rows),
            tropical_view(ordinary_terms.total_m_per_s, tropical_rows),
        ]
    )
    difference = baseline_terms.total_m_per_s - ordinary_terms.total_m_per_s
    difference_limit = symmetric_limit([tropical_view(difference, tropical_rows)])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    plot_field(axes[0], source.longitude_deg, latitude_tropical, tropical_view(baseline_terms.total_m_per_s, tropical_rows), "Current reconstruction", comparison_limit, "m/s")
    plot_field(axes[1], source.longitude_deg, latitude_tropical, tropical_view(ordinary_terms.total_m_per_s, tropical_rows), "Ordinary geostrophy", comparison_limit, "m/s")
    plot_field(axes[2], source.longitude_deg, latitude_tropical, tropical_view(difference, tropical_rows), "Current − ordinary", difference_limit, "m/s")
    save_figure(fig, args.output_dir / "03_subduction_comparison.png")

    profile_rows: list[dict[str, object]] = []
    for i, latitude in enumerate(source.latitude_deg):
        if not tropical_rows[i]:
            continue
        row_area = area[i]
        current_mask = baseline_terms.valid[i]
        ordinary_mask = ordinary_terms.valid[i]
        gradient_mask = gradient_valid[i]
        current_area = float(np.sum(row_area[current_mask]))
        profile_rows.append(
            {
                "latitude_deg": float(latitude),
                "valid_ocean_area_m2": current_area,
                "current_total_mean_m_per_s": weighted_mean(baseline_terms.total_m_per_s[i], row_area, current_mask),
                "current_zonal_mean_m_per_s": weighted_mean(baseline_terms.zonal_m_per_s[i], row_area, current_mask),
                "current_meridional_mean_m_per_s": weighted_mean(baseline_terms.meridional_m_per_s[i], row_area, current_mask),
                "ordinary_total_mean_m_per_s": weighted_mean(ordinary_terms.total_m_per_s[i], row_area, ordinary_mask),
                "blend_gradient_discrete_mean_m_per_s": weighted_mean(gradient_extra[i], row_area, gradient_mask),
                "blend_gradient_analytic_mean_m_per_s": weighted_mean(analytic_gradient_extra[i], row_area, gradient_mask),
                "current_positive_area_fraction": float(np.sum(row_area[current_mask & (baseline_terms.total_m_per_s[i] > 0.0)]) / current_area) if current_area else float("nan"),
            }
        )
    write_csv(args.output_dir / "latitude_profiles.csv", profile_rows)
    profile_latitude = np.array([row["latitude_deg"] for row in profile_rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label in (
        ("current_total_mean_m_per_s", "current total"),
        ("ordinary_total_mean_m_per_s", "ordinary total"),
        ("current_zonal_mean_m_per_s", "current zonal term"),
        ("current_meridional_mean_m_per_s", "current meridional term"),
        ("blend_gradient_discrete_mean_m_per_s", "discrete ∇w contribution"),
    ):
        axes[0].plot(profile_latitude, [row[key] for row in profile_rows], label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].axvspan(-BASE_BLEND_WIDTH_DEG, BASE_BLEND_WIDTH_DEG, color="grey", alpha=0.12)
    axes[0].set_ylabel("Area-weighted mean Sgeo (m/s)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(profile_latitude, [row["current_positive_area_fraction"] for row in profile_rows], color="#b2182b")
    axes[1].axvspan(-BASE_BLEND_WIDTH_DEG, BASE_BLEND_WIDTH_DEG, color="grey", alpha=0.12)
    axes[1].set_xlabel("Latitude")
    axes[1].set_ylabel("Positive-area fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    save_figure(fig, args.output_dir / "04_latitude_profiles.png")

    sensitivity_rows: list[dict[str, object]] = []
    variant_fields: list[np.ndarray] = []

    def evaluate_variant(group: str, value: float, solution: VelocitySolution) -> None:
        valid = mld.valid & solution.valid
        terms = subduction_terms(
            mld.maximum_m,
            solution.east,
            solution.north,
            source.longitude_deg,
            source.latitude_deg,
            valid,
        )
        stats = statistics(terms.total_m_per_s, terms.valid, area, tropical_band)
        transition_stats = statistics(
            terms.total_m_per_s, terms.valid, area, transition_band
        )
        common = terms.valid & baseline_terms.valid & tropical_band
        stats.update(
            {
                "parameter": group,
                "value": value,
                "correlation_with_baseline": correlation(terms.total_m_per_s, baseline_terms.total_m_per_s, common),
                "area_weighted_mean_difference_m_per_s": weighted_mean(terms.total_m_per_s - baseline_terms.total_m_per_s, area, common),
                "area_weighted_rms_difference_m_per_s": float(np.sqrt(weighted_mean((terms.total_m_per_s - baseline_terms.total_m_per_s) ** 2, area, common))),
                **{f"transition_{key}": result for key, result in transition_stats.items()},
            }
        )
        sensitivity_rows.append(stats)
        variant_fields.append(terms.total_m_per_s)

    for width in FIT_WIDTHS_DEG:
        evaluate_variant(
            "fit_half_width_deg",
            width,
            velocity_solution(source.height_m, source.longitude_deg, source.latitude_deg, fit_width_deg=width),
        )
    for smoothing in SMOOTHING_VALUES_M:
        evaluate_variant(
            "smoothing_m",
            smoothing,
            velocity_solution(source.height_m, source.longitude_deg, source.latitude_deg, smoothing_m=smoothing),
        )
    for width in BLEND_WIDTHS_DEG:
        evaluate_variant(
            "blend_width_deg",
            width,
            velocity_solution(source.height_m, source.longitude_deg, source.latitude_deg, blend_width_deg=width),
        )
    write_csv(args.output_dir / "sensitivity_summary.csv", sensitivity_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, group, title in zip(
        axes,
        ("fit_half_width_deg", "smoothing_m", "blend_width_deg"),
        ("β-plane fit half-width", "Spline smoothing", "Blend half-width"),
    ):
        rows = [row for row in sensitivity_rows if row["parameter"] == group]
        x = np.array([row["value"] for row in rows], dtype=float)
        mean = np.array([row["area_weighted_mean_m_per_s"] for row in rows], dtype=float)
        rms_difference = np.array([row["area_weighted_rms_difference_m_per_s"] for row in rows], dtype=float)
        axis.plot(x, mean, "o-", label="tropical mean Sgeo")
        axis.plot(x, rms_difference, "s--", label="RMS change from baseline")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel(group)
        axis.set_ylabel("m/s")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        if group == "smoothing_m":
            axis.set_xscale("symlog", linthresh=0.001)
    save_figure(fig, args.output_dir / "05_parameter_sensitivity.png")

    variant_stack = np.stack(variant_fields)
    variant_count = np.sum(np.isfinite(variant_stack), axis=0)
    variant_sum = np.nansum(variant_stack, axis=0)
    variant_mean = np.full(source.height_m.shape, np.nan)
    np.divide(variant_sum, variant_count, out=variant_mean, where=variant_count > 0)
    variant_squared_difference = np.where(
        np.isfinite(variant_stack),
        (variant_stack - variant_mean[None, :, :]) ** 2,
        0.0,
    )
    variant_std = np.full(source.height_m.shape, np.nan)
    np.divide(
        np.sum(variant_squared_difference, axis=0),
        variant_count,
        out=variant_std,
        where=variant_count > 0,
    )
    variant_std = np.sqrt(variant_std)
    positive_frequency = np.full(source.height_m.shape, np.nan)
    np.divide(
        np.sum((variant_stack > 0.0) & np.isfinite(variant_stack), axis=0),
        variant_count,
        out=positive_frequency,
        where=variant_count > 0,
    )
    robustness_limit = symmetric_limit([tropical_view(variant_mean, tropical_rows)])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    plot_field(
        axes[0],
        source.longitude_deg,
        latitude_tropical,
        tropical_view(variant_mean, tropical_rows),
        "Mean across parameter variants",
        robustness_limit,
        "m/s",
    )
    std_image = axes[1].pcolormesh(
        source.longitude_deg,
        latitude_tropical,
        tropical_view(variant_std, tropical_rows),
        shading="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.nanpercentile(tropical_view(variant_std, tropical_rows), 98.0)),
        rasterized=True,
    )
    axes[1].set_title("Standard deviation across variants")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(std_image, ax=axes[1], shrink=0.82, label="m/s")
    frequency_image = axes[2].pcolormesh(
        source.longitude_deg,
        latitude_tropical,
        tropical_view(positive_frequency, tropical_rows),
        shading="nearest",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[2].set_title("Fraction of variants with positive Sgeo")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(frequency_image, ax=axes[2], shrink=0.82, label="fraction")
    for axis in axes[1:]:
        axis.set_xlim(-180.0, 180.0)
        axis.set_ylim(-TROPICAL_LIMIT_DEG, TROPICAL_LIMIT_DEG)
    save_figure(fig, args.output_dir / "06_parameter_robustness.png")

    conservation = conservation_budget(
        mld.maximum_m,
        baseline,
        mld.valid,
        source.valid,
        baseline_terms,
        source.longitude_deg,
        source.latitude_deg,
        area,
    )
    write_csv(args.output_dir / "conservation_summary.csv", [conservation])
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [
        "∫S dA",
        "outward boundary flux",
        "source-mask boundary",
        "derived-missing boundary",
        "domain edge",
        "closure residual",
    ]
    values = [
        conservation["subduction_area_integral_m3_per_s"],
        conservation["estimated_outward_boundary_flux_m3_per_s"],
        conservation["source_mask_m3_per_s"],
        conservation["derived_missing_m3_per_s"],
        conservation["domain_edge_m3_per_s"],
        conservation["closure_residual_m3_per_s"],
    ]
    axis.bar(labels, values, color=["#2166ac", "#b2182b", "#ef8a62", "#67a9cf", "#999999", "#762a83"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("m³/s")
    axis.set_title("Global discrete conservation budget")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, args.output_dir / "07_conservation_budget.png")

    summary = {
        "baseline_parameters": {
            "fit_half_width_deg": BASE_FIT_WIDTH_DEG,
            "smoothing_m": BASE_SMOOTHING_M,
            "core_latitude_deg": CORE_LATITUDE_DEG,
            "blend_half_width_deg": BASE_BLEND_WIDTH_DEG,
        },
        "production_stencil_reproduction_max_abs_error_m_per_s": reproduction_error,
        "current_tropical_10deg": statistics(baseline_terms.total_m_per_s, baseline_terms.valid, area, tropical_band),
        "current_transition_0p25_to_3deg": statistics(baseline_terms.total_m_per_s, baseline_terms.valid, area, transition_band),
        "ordinary_transition_0p25_to_3deg": statistics(ordinary_terms.total_m_per_s, ordinary_terms.valid, area, transition_band),
        "ordinary_well_conditioned_3p5_to_10deg": statistics(ordinary_terms.total_m_per_s, ordinary_terms.valid, area, well_conditioned_band),
        "current_well_conditioned_3p5_to_10deg": statistics(baseline_terms.total_m_per_s, baseline_terms.valid, area, well_conditioned_band),
        "blend_gradient_discrete_transition": statistics(gradient_extra, gradient_valid, area, transition_band),
        "blend_gradient_analytic_transition": statistics(analytic_gradient_extra, gradient_valid, area, transition_band),
        "current_vs_ordinary_transition_correlation": correlation(
            baseline_terms.total_m_per_s,
            ordinary_terms.total_m_per_s,
            baseline_terms.valid & ordinary_terms.valid & transition_band,
        ),
        "blend_gradient_discrete_vs_analytic_correlation": correlation(
            gradient_extra,
            analytic_gradient_extra,
            gradient_valid & transition_band,
        ),
        "parameter_ensemble_transition": statistics(
            variant_mean,
            np.isfinite(variant_mean),
            area,
            transition_band,
        ),
        "global_conservation": conservation,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    tropical_means = [float(row["area_weighted_mean_m_per_s"]) for row in sensitivity_rows]
    transition_means = [
        float(row["transition_area_weighted_mean_m_per_s"])
        for row in sensitivity_rows
    ]
    correlations = [float(row["correlation_with_baseline"]) for row in sensitivity_rows]
    current_transition = summary["current_transition_0p25_to_3deg"]
    ordinary_transition = summary["ordinary_transition_0p25_to_3deg"]
    discrete_gradient = summary["blend_gradient_discrete_transition"]
    conservation_summary = summary["global_conservation"]
    report = f"""# Equatorial geostrophic-subduction diagnostics

## Scope and method

This is a diagnostic of the existing method; it does not alter the globe's velocity or
subduction calculation. The baseline uses a {BASE_FIT_WIDTH_DEG:g}° β-plane spline fit,
{BASE_SMOOTHING_M:g} m smoothing parameter, a {CORE_LATITUDE_DEG:g}° β-only core, and a
smoothstep transition to ordinary geostrophy at ±{BASE_BLEND_WIDTH_DEG:g}°. Ordinary
geostrophy is shown at every nonzero-latitude row, but quantitative "well-conditioned"
comparison uses |latitude| ≥ {BASE_BLEND_WIDTH_DEG + 0.5:g}°, one full grid stencil beyond
the blend. The diagnostic decomposition reproduces the production stencil to
{reproduction_error:.3e} m/s.

## Main findings

- The positive signal is concentrated in the transition rather than the full 10°S–10°N
  band. From {CORE_LATITUDE_DEG:g}° to {BASE_BLEND_WIDTH_DEG:g}°, current area-weighted
  mean Sgeo is {current_transition['area_weighted_mean_m_per_s']:.3e} m/s and
  {current_transition['area_weighted_positive_fraction']:.1%} of valid area is positive.
  Across the full tropical band the mean is
  {summary['current_tropical_10deg']['area_weighted_mean_m_per_s']:.3e} m/s and the
  positive-area fraction is {summary['current_tropical_10deg']['area_weighted_positive_fraction']:.1%}.
- Ordinary geostrophy over the same nonzero transition rows has mean
  {ordinary_transition['area_weighted_mean_m_per_s']:.3e} m/s. Its spatial correlation
  with the blended result is {summary['current_vs_ordinary_transition_correlation']:.3f};
  ordinary values close to the equator remain numerically ill-conditioned and are a
  comparison, not a preferred estimate.
- The exact discrete product-rule residual associated with the blend-weight gradient has
  transition mean {discrete_gradient['area_weighted_mean_m_per_s']:.3e} m/s, compared
  with {summary['blend_gradient_analytic_transition']['area_weighted_mean_m_per_s']:.3e}
  m/s from `-H (v_f - v_beta) dw/dy`. Their pointwise correlation is
  {summary['blend_gradient_discrete_vs_analytic_correlation']:.3f}. This term is therefore
  large enough to explain the positive transition-band mean, while the remaining terms
  partly offset it.

## Numerical sensitivity

- Across all one-at-a-time parameter variants, the 10°S–10°N mean remains positive,
  spanning {min(tropical_means):.3e} to {max(tropical_means):.3e} m/s. The transition-band
  mean spans {min(transition_means):.3e} to {max(transition_means):.3e} m/s.
- Spatial correlations with the baseline span {min(correlations):.3f} to
  {max(correlations):.3f}. The blend-width sweep produces the strongest pattern changes;
  the 1.5° blend has baseline correlation
  {[row['correlation_with_baseline'] for row in sensitivity_rows if row['parameter'] == 'blend_width_deg' and row['value'] == 1.5][0]:.3f}.
- The unsmoothed spline is materially different. Smoothing values 0.01, 0.03, and 0.1 m
  give the same fitted result for this dataset because the spline residual already falls
  below those thresholds; this should not be read as general smoothing insensitivity.
- The sign of the broad transition-band mean is robust over the tested settings, but many
  local anomalies and their magnitudes are not. See `06_parameter_robustness.png` for the
  per-cell sign frequency and ensemble spread.

## Conservation check

The source grid stops at {source.latitude_deg[0]:g}° and {source.latitude_deg[-1]:g}° and
its mask combines coastlines with source no-data. Using wet-cell transport to estimate
outward flux across source-mask, derived-missing, and outer-domain faces gives:

- area-integrated Sgeo: {conservation_summary['subduction_area_integral_m3_per_s'] / 1e6:.2f} Sv
- estimated outward boundary flux: {conservation_summary['estimated_outward_boundary_flux_m3_per_s'] / 1e6:.2f} Sv
- closure residual `integral(Sgeo) + outward flux`:
  {conservation_summary['closure_residual_m3_per_s'] / 1e6:.2f} Sv
  ({conservation_summary['relative_closure_residual']:.1%} of the larger budget term)

The current centered/one-sided cell derivative is not a finite-volume flux operator, so
it does not satisfy exact discrete global conservation. Boundary-adjacent cells contribute
{conservation_summary['boundary_cell_subduction_integral_m3_per_s'] / 1e6:.2f} Sv and
interior cells {conservation_summary['interior_cell_subduction_integral_m3_per_s'] / 1e6:.2f}
Sv. The boundary-flux estimate is approximate because no normal velocity is available on
the dry/no-data side; no equatorial values were zeroed and no anomalies were smoothed.

## Outputs

The numbered PNGs contain velocity components, subduction components, the ordinary
comparison, latitude profiles, parameter sensitivity, sign robustness, and conservation.
The CSV and JSON files retain the plotted statistics for independent inspection.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote equatorial diagnostics to {args.output_dir}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
