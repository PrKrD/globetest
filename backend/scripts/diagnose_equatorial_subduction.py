"""Diagnose tropical geostrophic velocity and subduction without changing the globe product."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from app.config import DATA_DIR
from app.dyntopo import EARTH_RADIUS_METERS, load_maximenko
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
    DEFAULT_FIT_HALF_WIDTH_DEG,
    DEFAULT_SPLINE_SMOOTHING_M,
    GeostrophicVelocity,
    reconstruct_geostrophic_velocity,
)


CORE_LATITUDE_DEG = DEFAULT_BETA_PLANE_CORE_LATITUDE_DEG
BASE_FIT_WIDTH_DEG = DEFAULT_FIT_HALF_WIDTH_DEG
BASE_SMOOTHING_M = DEFAULT_SPLINE_SMOOTHING_M
BASE_BLEND_WIDTH_DEG = DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG
TROPICAL_LIMIT_DEG = 10.0
FIT_WIDTHS_DEG = (3.0, 3.5, 4.5, 6.0)
SMOOTHING_VALUES_M = (0.0, 0.003, 0.01, 0.03, 0.1)
BLEND_WIDTHS_DEG = (1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class LegacySubduction:
    rate_m_per_s: np.ndarray
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


def legacy_subduction_terms(
    depth_m: np.ndarray,
    east_velocity_m_per_s: np.ndarray,
    north_velocity_m_per_s: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    valid: np.ndarray,
) -> LegacySubduction:
    """Reproduce the retired centered/one-sided stencil for comparison."""
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
    return LegacySubduction(total, result_valid)


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
    solution: GeostrophicVelocity,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    mld_valid: np.ndarray,
    physical_ocean: np.ndarray,
    blend_width_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare component divergences on one deliberately common support."""
    common = mld_valid & solution.beta_valid & solution.ordinary_valid
    beta_subduction = geostrophic_subduction_rate(
        depth_m,
        solution.beta_east,
        solution.beta_north,
        longitude_deg,
        latitude_deg,
        common,
        physical_ocean,
    )
    ordinary_subduction = geostrophic_subduction_rate(
        depth_m,
        solution.ordinary_east,
        solution.ordinary_north,
        longitude_deg,
        latitude_deg,
        common,
        physical_ocean,
    )
    blended_subduction = geostrophic_subduction_rate(
        depth_m,
        solution.east,
        solution.north,
        longitude_deg,
        latitude_deg,
        common,
        physical_ocean,
    )
    base = (
        (1.0 - solution.blend_weight) * beta_subduction.rate_m_per_s
        + solution.blend_weight * ordinary_subduction.rate_m_per_s
    )
    discrete_extra = blended_subduction.rate_m_per_s - base

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
    analytic_extra[~blended_subduction.valid] = np.nan
    discrete_extra[~blended_subduction.valid] = np.nan
    return discrete_extra, analytic_extra, blended_subduction.valid


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
    latitude_limit: float | None = TROPICAL_LIMIT_DEG,
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
    if latitude_limit is not None:
        axis.set_ylim(-latitude_limit, latitude_limit)
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
    physical_ocean = regrid_ocean_mask_by_overlap(
        mld_source.source_ocean,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        source.longitude_deg,
        source.latitude_deg,
    )
    baseline = reconstruct_geostrophic_velocity(
        source.height_m,
        source.longitude_deg,
        source.latitude_deg,
    )
    baseline_valid = mld.valid & baseline.valid
    baseline_subduction = geostrophic_subduction_rate(
        mld.maximum_m,
        baseline.east,
        baseline.north,
        source.longitude_deg,
        source.latitude_deg,
        baseline_valid,
        physical_ocean,
    )
    area = baseline_subduction.area_m2
    legacy_terms = legacy_subduction_terms(
        mld.maximum_m,
        baseline.east,
        baseline.north,
        source.longitude_deg,
        source.latitude_deg,
        baseline_valid,
    )

    ordinary_valid = mld.valid & baseline.ordinary_valid
    ordinary_subduction = geostrophic_subduction_rate(
        mld.maximum_m,
        baseline.ordinary_east,
        baseline.ordinary_north,
        source.longitude_deg,
        source.latitude_deg,
        ordinary_valid,
        physical_ocean,
    )
    gradient_extra, analytic_gradient_extra, gradient_valid = blend_gradient_terms(
        mld.maximum_m,
        baseline,
        source.longitude_deg,
        source.latitude_deg,
        mld.valid,
        physical_ocean,
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
    velocity_panels = (
        (baseline.east, "Current eastward velocity", velocity_limit),
        (baseline.north, "Current northward velocity", velocity_limit),
        (
            baseline.east - baseline.ordinary_east,
            "Eastward: current − ordinary",
            velocity_difference_limit,
        ),
        (
            baseline.north - baseline.ordinary_north,
            "Northward: current − ordinary",
            velocity_difference_limit,
        ),
    )
    for axis, (field, title, limit) in zip(axes.flat, velocity_panels):
        plot_field(
            axis,
            source.longitude_deg,
            latitude_tropical,
            tropical_view(field, tropical_rows),
            title,
            limit,
            "m/s",
        )
    save_figure(fig, args.output_dir / "01_velocity_components.png")

    subduction_limit = symmetric_limit(
        [
            tropical_view(baseline_subduction.zonal_rate_m_per_s, tropical_rows),
            tropical_view(baseline_subduction.meridional_rate_m_per_s, tropical_rows),
            tropical_view(baseline_subduction.rate_m_per_s, tropical_rows),
        ]
    )
    gradient_limit = symmetric_limit([tropical_view(gradient_extra, tropical_rows)])
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
    subduction_panels = (
        (
            baseline_subduction.zonal_rate_m_per_s,
            "Zonal transport convergence",
            subduction_limit,
        ),
        (
            baseline_subduction.meridional_rate_m_per_s,
            "Meridional transport convergence",
            subduction_limit,
        ),
        (
            baseline_subduction.rate_m_per_s,
            "Total geostrophic subduction",
            subduction_limit,
        ),
        (gradient_extra, "Discrete contribution from ∇w", gradient_limit),
    )
    for axis, (field, title, limit) in zip(axes.flat, subduction_panels):
        plot_field(
            axis,
            source.longitude_deg,
            latitude_tropical,
            tropical_view(field, tropical_rows),
            title,
            limit,
            "m/s",
        )
    save_figure(fig, args.output_dir / "02_subduction_components.png")

    comparison_limit = symmetric_limit(
        [
            tropical_view(baseline_subduction.rate_m_per_s, tropical_rows),
            tropical_view(ordinary_subduction.rate_m_per_s, tropical_rows),
        ]
    )
    difference = baseline_subduction.rate_m_per_s - ordinary_subduction.rate_m_per_s
    difference_limit = symmetric_limit([tropical_view(difference, tropical_rows)])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    comparison_panels = (
        (baseline_subduction.rate_m_per_s, "Current reconstruction", comparison_limit),
        (ordinary_subduction.rate_m_per_s, "Ordinary geostrophy", comparison_limit),
        (difference, "Current − ordinary", difference_limit),
    )
    for axis, (field, title, limit) in zip(axes, comparison_panels):
        plot_field(
            axis,
            source.longitude_deg,
            latitude_tropical,
            tropical_view(field, tropical_rows),
            title,
            limit,
            "m/s",
        )
    save_figure(fig, args.output_dir / "03_subduction_comparison.png")

    profile_rows: list[dict[str, object]] = []
    for i, latitude in enumerate(source.latitude_deg):
        if not tropical_rows[i]:
            continue
        row_area = area[i]
        current_mask = baseline_subduction.valid[i]
        ordinary_mask = ordinary_subduction.valid[i]
        gradient_mask = gradient_valid[i]
        current_area = float(np.sum(row_area[current_mask]))
        profile_rows.append(
            {
                "latitude_deg": float(latitude),
                "valid_ocean_area_m2": current_area,
                "current_total_mean_m_per_s": weighted_mean(
                    baseline_subduction.rate_m_per_s[i], row_area, current_mask
                ),
                "current_zonal_mean_m_per_s": weighted_mean(
                    baseline_subduction.zonal_rate_m_per_s[i], row_area, current_mask
                ),
                "current_meridional_mean_m_per_s": weighted_mean(
                    baseline_subduction.meridional_rate_m_per_s[i], row_area, current_mask
                ),
                "ordinary_total_mean_m_per_s": weighted_mean(
                    ordinary_subduction.rate_m_per_s[i], row_area, ordinary_mask
                ),
                "blend_gradient_discrete_mean_m_per_s": weighted_mean(
                    gradient_extra[i], row_area, gradient_mask
                ),
                "blend_gradient_analytic_mean_m_per_s": weighted_mean(
                    analytic_gradient_extra[i], row_area, gradient_mask
                ),
                "current_positive_area_fraction": (
                    float(
                        np.sum(
                            row_area[
                                current_mask
                                & (baseline_subduction.rate_m_per_s[i] > 0.0)
                            ]
                        )
                        / current_area
                    )
                    if current_area
                    else float("nan")
                ),
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
    axes[1].plot(
        profile_latitude,
        [row["current_positive_area_fraction"] for row in profile_rows],
        color="#b2182b",
    )
    axes[1].axvspan(-BASE_BLEND_WIDTH_DEG, BASE_BLEND_WIDTH_DEG, color="grey", alpha=0.12)
    axes[1].set_xlabel("Latitude")
    axes[1].set_ylabel("Positive-area fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    save_figure(fig, args.output_dir / "04_latitude_profiles.png")

    sensitivity_rows: list[dict[str, object]] = []
    variant_fields: list[np.ndarray] = []

    def evaluate_variant(
        group: str,
        value: float,
        solution: GeostrophicVelocity,
    ) -> None:
        valid = mld.valid & solution.valid
        subduction = geostrophic_subduction_rate(
            mld.maximum_m,
            solution.east,
            solution.north,
            source.longitude_deg,
            source.latitude_deg,
            valid,
            physical_ocean,
        )
        stats = statistics(
            subduction.rate_m_per_s,
            subduction.valid,
            area,
            tropical_band,
        )
        transition_stats = statistics(
            subduction.rate_m_per_s,
            subduction.valid,
            area,
            transition_band,
        )
        common = subduction.valid & baseline_subduction.valid & tropical_band
        difference_from_baseline = (
            subduction.rate_m_per_s - baseline_subduction.rate_m_per_s
        )
        stats.update(
            {
                "parameter": group,
                "value": value,
                "correlation_with_baseline": correlation(
                    subduction.rate_m_per_s,
                    baseline_subduction.rate_m_per_s,
                    common,
                ),
                "area_weighted_mean_difference_m_per_s": weighted_mean(
                    difference_from_baseline,
                    area,
                    common,
                ),
                "area_weighted_rms_difference_m_per_s": float(
                    np.sqrt(
                        weighted_mean(
                            difference_from_baseline**2,
                            area,
                            common,
                        )
                    )
                ),
                **{
                    f"transition_{key}": result
                    for key, result in transition_stats.items()
                },
            }
        )
        sensitivity_rows.append(stats)
        variant_fields.append(subduction.rate_m_per_s)

    for width in FIT_WIDTHS_DEG:
        evaluate_variant(
            "fit_half_width_deg",
            width,
            reconstruct_geostrophic_velocity(
                source.height_m,
                source.longitude_deg,
                source.latitude_deg,
                fit_half_width_deg=width,
            ),
        )
    for smoothing in SMOOTHING_VALUES_M:
        evaluate_variant(
            "smoothing_m",
            smoothing,
            reconstruct_geostrophic_velocity(
                source.height_m,
                source.longitude_deg,
                source.latitude_deg,
                smoothing_m=smoothing,
            ),
        )
    for width in BLEND_WIDTHS_DEG:
        evaluate_variant(
            "blend_width_deg",
            width,
            reconstruct_geostrophic_velocity(
                source.height_m,
                source.longitude_deg,
                source.latitude_deg,
                blend_half_width_deg=width,
            ),
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
        rms_difference = np.array(
            [row["area_weighted_rms_difference_m_per_s"] for row in rows],
            dtype=float,
        )
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

    conservation = {
        "valid_subduction_cells": int(np.sum(baseline_subduction.valid)),
        **asdict(baseline_subduction.conservation),
        "relative_closure_residual": (
            baseline_subduction.conservation.relative_closure_residual
        ),
    }
    write_csv(args.output_dir / "conservation_summary.csv", [conservation])
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [
        "∫S dA",
        "outward boundary flux",
        "coastline",
        "missing-data boundary",
        "open domain",
        "closure residual",
    ]
    values = [
        conservation["area_integrated_subduction_m3_per_s"],
        conservation["total_outward_m3_per_s"],
        conservation["coastline_outward_m3_per_s"],
        conservation["missing_data_outward_m3_per_s"],
        conservation["open_domain_outward_m3_per_s"],
        conservation["closure_residual_m3_per_s"],
    ]
    axis.bar(
        labels,
        values,
        color=["#2166ac", "#b2182b", "#ef8a62", "#67a9cf", "#999999", "#762a83"],
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("m³/s")
    axis.set_title("Global discrete conservation budget")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, args.output_dir / "07_conservation_budget.png")

    common_old_new = baseline_subduction.valid & legacy_terms.valid
    finite_volume_difference = np.where(
        common_old_new,
        baseline_subduction.rate_m_per_s - legacy_terms.rate_m_per_s,
        np.nan,
    )
    north_wet = np.zeros_like(baseline_subduction.valid)
    south_wet = np.zeros_like(baseline_subduction.valid)
    north_wet[:-1] = baseline_subduction.valid[1:]
    south_wet[1:] = baseline_subduction.valid[:-1]
    wet_interior = (
        common_old_new
        & np.roll(baseline_subduction.valid, 1, axis=1)
        & np.roll(baseline_subduction.valid, -1, axis=1)
        & north_wet
        & south_wet
    )
    boundary_adjacent = common_old_new & ~wet_interior
    interior_difference = np.where(
        wet_interior,
        finite_volume_difference,
        np.nan,
    )
    global_limit = symmetric_limit([finite_volume_difference])
    global_interior_limit = symmetric_limit([interior_difference])
    tropical_difference = tropical_view(finite_volume_difference, tropical_rows)
    tropical_interior_difference = tropical_view(interior_difference, tropical_rows)
    tropical_limit = symmetric_limit([tropical_difference])
    tropical_interior_limit = symmetric_limit([tropical_interior_difference])

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    plot_field(
        axes[0, 0],
        source.longitude_deg,
        source.latitude_deg,
        finite_volume_difference,
        "Full domain: all common cells",
        global_limit,
        "m/s",
        latitude_limit=None,
    )
    plot_field(
        axes[0, 1],
        source.longitude_deg,
        source.latitude_deg,
        interior_difference,
        "Full domain: wet interior scale",
        global_interior_limit,
        "m/s",
        latitude_limit=None,
    )
    plot_field(
        axes[1, 0],
        source.longitude_deg,
        latitude_tropical,
        tropical_difference,
        "10°S–10°N: all common cells",
        tropical_limit,
        "m/s",
    )
    plot_field(
        axes[1, 1],
        source.longitude_deg,
        latitude_tropical,
        tropical_interior_difference,
        "10°S–10°N: wet interior scale",
        tropical_interior_limit,
        "m/s",
    )
    for axis in axes[1]:
        axis.axhline(0.0, color="black", linewidth=0.6)
    fig.suptitle("Finite volume minus legacy finite difference")
    save_figure(fig, args.output_dir / "08_finite_volume_difference.png")

    summary = {
        "baseline_parameters": {
            "fit_half_width_deg": BASE_FIT_WIDTH_DEG,
            "smoothing_m": BASE_SMOOTHING_M,
            "core_latitude_deg": CORE_LATITUDE_DEG,
            "blend_half_width_deg": BASE_BLEND_WIDTH_DEG,
        },
        "finite_volume_minus_finite_difference_global": statistics(
            finite_volume_difference,
            common_old_new,
            area,
            common_old_new,
        ),
        "finite_volume_minus_finite_difference_tropical_10deg": statistics(
            finite_volume_difference,
            common_old_new,
            area,
            common_old_new & tropical_band,
        ),
        "finite_volume_minus_finite_difference_global_interior": statistics(
            finite_volume_difference,
            wet_interior,
            area,
            wet_interior,
        ),
        "finite_volume_minus_finite_difference_global_boundary": statistics(
            finite_volume_difference,
            boundary_adjacent,
            area,
            boundary_adjacent,
        ),
        "finite_volume_minus_finite_difference_tropical_interior": statistics(
            finite_volume_difference,
            wet_interior,
            area,
            wet_interior & tropical_band,
        ),
        "finite_volume_minus_finite_difference_tropical_boundary": statistics(
            finite_volume_difference,
            boundary_adjacent,
            area,
            boundary_adjacent & tropical_band,
        ),
        "current_tropical_10deg": statistics(
            baseline_subduction.rate_m_per_s,
            baseline_subduction.valid,
            area,
            tropical_band,
        ),
        "current_transition_0p25_to_3deg": statistics(
            baseline_subduction.rate_m_per_s,
            baseline_subduction.valid,
            area,
            transition_band,
        ),
        "ordinary_transition_0p25_to_3deg": statistics(
            ordinary_subduction.rate_m_per_s,
            ordinary_subduction.valid,
            area,
            transition_band,
        ),
        "ordinary_well_conditioned_3p5_to_10deg": statistics(
            ordinary_subduction.rate_m_per_s,
            ordinary_subduction.valid,
            area,
            well_conditioned_band,
        ),
        "current_well_conditioned_3p5_to_10deg": statistics(
            baseline_subduction.rate_m_per_s,
            baseline_subduction.valid,
            area,
            well_conditioned_band,
        ),
        "blend_gradient_discrete_transition": statistics(
            gradient_extra,
            gradient_valid,
            area,
            transition_band,
        ),
        "blend_gradient_analytic_transition": statistics(
            analytic_gradient_extra,
            gradient_valid,
            area,
            transition_band,
        ),
        "current_vs_ordinary_transition_correlation": correlation(
            baseline_subduction.rate_m_per_s,
            ordinary_subduction.rate_m_per_s,
            baseline_subduction.valid & ordinary_subduction.valid & transition_band,
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
    interior_difference_summary = summary[
        "finite_volume_minus_finite_difference_global_interior"
    ]
    boundary_difference_summary = summary[
        "finite_volume_minus_finite_difference_global_boundary"
    ]
    tropical_positive_fraction = summary["current_tropical_10deg"][
        "area_weighted_positive_fraction"
    ]
    narrow_blend_correlation = next(
        float(row["correlation_with_baseline"])
        for row in sensitivity_rows
        if row["parameter"] == "blend_width_deg" and row["value"] == 1.5
    )
    report = f"""# Equatorial geostrophic-subduction diagnostics

## Scope and method

This diagnostic uses the same conservative finite-volume subduction operator as the
globe build. The baseline uses a {BASE_FIT_WIDTH_DEG:g}° β-plane spline fit,
{BASE_SMOOTHING_M:g} m smoothing parameter, a {CORE_LATITUDE_DEG:g}° β-only core, and a
smoothstep transition to ordinary geostrophy at ±{BASE_BLEND_WIDTH_DEG:g}°. Ordinary
geostrophy is shown at every nonzero-latitude row, but quantitative "well-conditioned"
comparison uses |latitude| ≥ {BASE_BLEND_WIDTH_DEG + 0.5:g}°, one full grid stencil beyond
the blend.

## Main findings

- The positive signal is concentrated in the transition rather than the full 10°S–10°N
  band. From {CORE_LATITUDE_DEG:g}° to {BASE_BLEND_WIDTH_DEG:g}°, current area-weighted
  mean Sgeo is {current_transition['area_weighted_mean_m_per_s']:.3e} m/s and
  {current_transition['area_weighted_positive_fraction']:.1%} of valid area is positive.
  Across the full tropical band the mean is
  {summary['current_tropical_10deg']['area_weighted_mean_m_per_s']:.3e} m/s and the
  positive-area fraction is {tropical_positive_fraction:.1%}.
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
  {narrow_blend_correlation:.3f}.
- The unsmoothed spline is materially different. Smoothing values 0.01, 0.03, and 0.1 m
  give the same fitted result for this dataset because the spline residual already falls
  below those thresholds; this should not be read as general smoothing insensitivity.
- The sign of the broad transition-band mean is robust over the tested settings, but many
  local anomalies and their magnitudes are not. See `06_parameter_robustness.png` for the
  per-cell sign frequency and ensemble spread.

## Finite-volume comparison

The finite-volume and legacy centered operators are nearly identical in the fully wet
interior, where their RMS difference is
{interior_difference_summary['area_weighted_rms_m_per_s']:.3e} m/s. Boundary-adjacent
cells change materially because the finite-volume method applies explicit face policies;
their RMS difference is
{boundary_difference_summary['area_weighted_rms_m_per_s']:.3e} m/s.
`08_finite_volume_difference.png` shows separate all-cell and wet-interior colour scales
for both the full domain and 10°S–10°N.

## Conservation check

The source grid stops at {source.latitude_deg[0]:g}° and {source.latitude_deg[-1]:g}° and
the remapped MLD ocean mask separates physical coastlines from oceanic missing data.
The cell integral and boundary budget use the exact same finite-volume face fluxes:

- area-integrated Sgeo: {conservation_summary['area_integrated_subduction_m3_per_s'] / 1e6:.2f} Sv
- outward boundary flux: {conservation_summary['total_outward_m3_per_s'] / 1e6:.2f} Sv
- coastline / missing-data / open-domain flux:
  {conservation_summary['coastline_outward_m3_per_s'] / 1e6:.2f} /
  {conservation_summary['missing_data_outward_m3_per_s'] / 1e6:.2f} /
  {conservation_summary['open_domain_outward_m3_per_s'] / 1e6:.2f} Sv
- closure residual `integral(Sgeo) + outward flux`:
  {conservation_summary['closure_residual_m3_per_s'] / 1e6:.2f} Sv
  ({conservation_summary['relative_closure_residual']:.1%} of the larger budget term)

Coastline faces are impermeable. Missing-data and north/south open-domain faces use the
adjacent wet-cell transport. Shared wet faces use one arithmetic-mean transport, so their
contributions cancel pairwise. The finite-volume minus legacy finite-difference comparison
is shown globally and across 10°S–10°N in the final numbered figure.

## Outputs

The numbered PNGs contain velocity components, subduction components, the ordinary
comparison, latitude profiles, parameter sensitivity, sign robustness, conservation, and
the old/new discretization difference. The CSV and JSON files retain the plotted
statistics for independent inspection.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote equatorial diagnostics to {args.output_dir}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
