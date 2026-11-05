"""Combine the existing geostrophic transport convergence with ERA5 Ekman contribution.

Run from backend with PYTHONPATH="$PWD" python scripts/build_total_subduction.py ...
The saved Ekman field uses the sign convention in ekman_pumping.py:
Mx=+tau_y/(rho*f), My=-tau_x/(rho*f), hence w_saved=-w_upward_standard.
Therefore S_total = S_geo + w_saved for a downward-positive convergence rate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from app.config import DATA_DIR
from app.dyntopo import load_maximenko
from app.mld import load_mld, maximum_mld, regrid_maximum_mld, regrid_ocean_mask_by_overlap
from app.subduction import geostrophic_subduction_rate
from app.velocity import reconstruct_geostrophic_velocity

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def regrid_ekman(lat_in, lon_in, field_in, lat_out, lon_out):
    """Periodic bilinear interpolation, only where all contributing source values are valid."""
    lat = np.asarray(lat_in, dtype=float)
    lon = np.mod(np.asarray(lon_in, dtype=float), 360.0)
    field = np.asarray(field_in, dtype=float)
    if field.shape != (len(lat), len(lon)):
        raise ValueError("Ekman field and coordinate shapes do not match")
    if lat[0] > lat[-1]:
        lat, field = lat[::-1], field[::-1]
    if np.any(np.diff(lat) <= 0):
        raise ValueError("Ekman latitudes must be strictly monotonic")
    order = np.argsort(lon)
    lon, field = lon[order], field[:, order]
    # Remove duplicate periodic seam if 0 and 360 are both present.
    keep = np.r_[True, np.diff(lon) > 1e-8]
    lon, field = lon[keep], field[:, keep]
    if len(lon) < 2 or np.any(np.diff(lon) <= 0):
        raise ValueError("Ekman longitudes are invalid")
    lon_ext = np.r_[lon[-1] - 360, lon, lon[0] + 360]
    field_ext = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)
    finite = np.isfinite(field_ext)
    filled = np.where(finite, field_ext, 0.0)
    points_lon, points_lat = np.meshgrid(np.mod(lon_out, 360), lat_out)
    query = np.column_stack((points_lat.ravel(), points_lon.ravel()))
    value_interp = RegularGridInterpolator((lat, lon_ext), filled, bounds_error=False, fill_value=np.nan)
    mask_interp = RegularGridInterpolator((lat, lon_ext), finite.astype(float), bounds_error=False, fill_value=0.0)
    values = value_interp(query).reshape(points_lat.shape)
    coverage = mask_interp(query).reshape(points_lat.shape)
    # Avoid filling land or excluded equatorial bands by interpolating across gaps.
    return np.where(coverage >= 1 - 1e-9, values, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--mld-input", type=Path, default=DATA_DIR / "mld_dr003_ref10m_v2023.nc")
    parser.add_argument("--ekman", type=Path, required=True, help="ekman_pumping_ocean.npz (preferred) or ekman_pumping.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/total_subduction"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = load_maximenko(str(args.input))
    mld_source = load_mld(str(args.mld_input))
    mld_max = maximum_mld(mld_source.monthly_m)
    physical_ocean = regrid_ocean_mask_by_overlap(
        mld_source.source_ocean, mld_source.longitude_deg, mld_source.latitude_deg,
        source.longitude_deg, source.latitude_deg,
    )
    mld = regrid_maximum_mld(
        mld_max.maximum_m, mld_max.month_count,
        mld_source.longitude_deg, mld_source.latitude_deg,
        source.longitude_deg, source.latitude_deg, physical_ocean,
    )
    velocity = reconstruct_geostrophic_velocity(source.height_m, source.longitude_deg, source.latitude_deg)



    geo = geostrophic_subduction_rate(
        mld.maximum_m, velocity.east, velocity.north,
        source.longitude_deg, source.latitude_deg,
        mld.valid & velocity.valid, physical_ocean,
    )

    with np.load(args.ekman) as ek:
        lat_ek, lon_ek = ek["latitude"], ek["longitude"]
        if "w_Ek" in ek:
            w_saved = ek["w_Ek"]  # m/s, negative of standard upward-positive pumping
        elif "w_Ek_m_per_year" in ek:
            w_saved = ek["w_Ek_m_per_year"] / SECONDS_PER_YEAR
        else:
            raise KeyError("Expected w_Ek or w_Ek_m_per_year in Ekman NPZ")
    ek_contribution = regrid_ekman(lat_ek, lon_ek, w_saved, source.latitude_deg, source.longitude_deg)
    common = geo.valid & np.isfinite(ek_contribution)
    geo_field = np.where(common, geo.rate_m_per_s, np.nan)
    ek_field = np.where(common, ek_contribution, np.nan)
    total_field = geo_field + ek_field

    np.savez_compressed(
        args.output_dir / "total_subduction.npz",
        latitude=source.latitude_deg, longitude=source.longitude_deg,
        S_geo_m_per_s=geo_field, S_ekman_m_per_s=ek_field,
        S_total_m_per_s=total_field, valid=common,
        H_m=mld.maximum_m,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    for ax, field, label in zip(
        axes, (geo_field, ek_field, total_field),
        ("Geostrophic contribution", "Ekman contribution (saved sign)", "Combined two-term subduction"),
    ):
        yearly = field * SECONDS_PER_YEAR
        finite = np.abs(yearly[np.isfinite(yearly)])
        limit = float(np.percentile(finite, 98)) if finite.size else 1.0
        if limit <= 0:
            limit = 1.0
        im = ax.pcolormesh(source.longitude_deg, source.latitude_deg, yearly,
                           shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{label} (98th-percentile scale: ±{limit:.2g} m/year)")
        ax.set_xlabel("Longitude (degrees east)")
        ax.set_ylabel("Latitude (degrees north)")
        fig.colorbar(im, ax=ax, label="m/year")
    fig.savefig(args.output_dir / "total_subduction_components.png", dpi=180)
    plt.close(fig)

    print(f"Valid common cells: {common.sum():,} / {common.size:,}")
    for name, field in (("Geostrophic", geo_field), ("Ekman", ek_field), ("Combined", total_field)):
        vals = field[np.isfinite(field)] * SECONDS_PER_YEAR
        if vals.size:
            print(f"{name}: median={np.median(vals):.3g}, |p98|={np.percentile(np.abs(vals), 98):.3g} m/year")
    print(f"Saved to {args.output_dir.resolve()}")
    print("Note: this is a two-term approximation, not a complete mixed-layer subduction budget.")


if __name__ == "__main__":
    main()
