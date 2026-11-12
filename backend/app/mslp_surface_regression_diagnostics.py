"""Numerical acceptance, area-weighted summaries and global regression figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import xarray as xr

from app.mslp_surface_regression import COMPARE_TOL, COUNTS, cell_areas

REGIONS = {"global_ocean": (-90, 90), "northern_extratropics": (20, 90),
           "southern_extratropics": (-90, -20), "tropics": (-20, 20)}
_MESH_CACHE = {}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
    path.write_text(json.dumps(clean(obj), indent=2, default=str, allow_nan=False)+"\n")


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(a, weights):
    valid = np.isfinite(a) & (weights > 0)
    return float(np.sum(a[valid]*weights[valid])/np.sum(weights[valid])) if valid.any() else np.nan


def weighted_correlation(a, b, weights):
    valid = np.isfinite(a) & np.isfinite(b) & (weights > 0)
    if valid.sum() < 2:
        return np.nan
    a, b, w = a[valid], b[valid], weights[valid]
    a, b = a-np.average(a, weights=w), b-np.average(b, weights=w)
    denom = np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
    return float(np.sum(w*a*b)/denom) if denom > 0 else np.nan


def numerical_summary(out: Path, kinds: list[str], pc_diagnostics: dict) -> dict:
    comparisons, regions, reconstruction, physical = [], [], [], []
    for hemi, directory in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        for kind in kinds:
            root = out/directory
            with xr.open_dataset(root/"individual"/f"{kind}.nc") as individual, \
                 xr.open_dataset(root/"joint"/f"{kind}.nc") as joint, \
                 xr.open_dataset(root/"comparison"/f"{kind}.nc") as comparison:
                area = cell_areas(joint.latitude.values, joint.longitude.values)
                target_var = joint.target_variance.values
                scale = max(np.sqrt(weighted_mean(target_var, area)), 1e-12)
                expected = pc_diagnostics[hemi]["orthogonal"]
                for n in range(12):
                    a = individual.coefficient.isel(mode=n).values
                    b = joint.coefficient.isel(mode=n).values
                    delta = comparison.coefficient_difference.isel(mode=n).values
                    finite = np.isfinite(delta)
                    ref_scale = max(np.sqrt(weighted_mean(a*a, area)), scale*1e-12)
                    rmse = np.sqrt(weighted_mean(delta*delta, area))
                    passed = bool(np.all(np.abs(delta[finite]) <= COMPARE_TOL*(scale+np.abs(a[finite])))) if finite.any() else False
                    comparisons.append({"hemisphere": hemi, "variable": kind, "mode": n+1,
                        "valid_cells": int(finite.sum()), "max_absolute_difference": float(np.max(np.abs(delta[finite]))) if finite.any() else np.nan,
                        "mean_absolute_difference": float(np.mean(np.abs(delta[finite]))) if finite.any() else np.nan,
                        "area_weighted_rmse": rmse, "relative_coefficient_error": rmse/ref_scale,
                        "spatial_pattern_correlation": weighted_correlation(a, b, area),
                        "target_sd_reference_scale": scale, "atol": COMPARE_TOL*scale, "rtol": COMPARE_TOL,
                        "equivalence_expected": expected, "agreement_pass": passed})
                pred = comparison.reconstruction_max_absolute_difference.values
                rdiff = comparison.r_squared_sum_difference.values
                valid = np.isfinite(pred)
                rvalid = np.isfinite(rdiff)
                reconstruction.append({"hemisphere": hemi, "variable": kind,
                    "max_reconstruction_difference": float(np.max(pred[valid])) if valid.any() else np.nan,
                    "max_r_squared_sum_difference": float(np.max(np.abs(rdiff[rvalid]))) if rvalid.any() else np.nan,
                    "reconstruction_pass": bool(valid.any() and np.all(pred[valid] <= COMPARE_TOL*scale)),
                    "r_squared_sum_pass": bool(rvalid.any() and np.all(np.abs(rdiff[rvalid]) <= COMPARE_TOL)),
                    "correlated_design_identity_max_error": float(comparison.correlated_design_identity_error.max()),
                    "equivalence_expected": expected})
                nested = joint.nested_r_squared.values
                nested_mse = joint.nested_rmse.values**2
                land = joint.land_fraction.values
                for region, bounds in REGIONS.items():
                    region_area = cell_areas(joint.latitude.values, joint.longitude.values, bounds)
                    pc1 = joint.coefficient.isel(mode=0).values
                    physical.append({"hemisphere": hemi, "variable": kind, "region": region,
                                     "pc1_area_mean_coefficient": weighted_mean(pc1, region_area),
                                     "pc1_area_rms_coefficient": float(np.sqrt(weighted_mean(pc1*pc1, region_area))),
                                     "units": joint.coefficient.attrs["units"]})
                    for policy in ("primary", "pure_ocean"):
                        weights = region_area*(land < .5 if policy == "primary" else land <= .05)
                        previous = 0.
                        for index, k in enumerate(COUNTS):
                            r2 = nested[index]
                            eligible = np.isfinite(r2) & (weights > 0)
                            denom = np.sum(weights[eligible]*target_var[eligible])
                            fraction = (1-np.sum(weights[eligible]*nested_mse[index][eligible]) /
                                        (denom*(int(joint.attrs["n_samples"])-1)/int(joint.attrs["n_samples"]))) if denom > 0 else np.nan
                            regions.append({"hemisphere": hemi, "variable": kind, "region": region,
                                "mask_policy": policy, "retained_modes": k,
                                "valid_cells": int(eligible.sum()), "valid_area_m2": float(weights[eligible].sum()),
                                "mean_local_r_squared": weighted_mean(r2, weights),
                                "total_variance_fraction_explained": float(fraction),
                                "incremental_variance_fraction": float(fraction-previous),
                                "area_weighted_rmse": float(np.sqrt(weighted_mean(nested_mse[index], weights)))})
                            previous = fraction
    write_rows(out/"reports"/"individual_joint_equivalence.csv", comparisons)
    write_rows(out/"variance_diagnostics"/"regional_variance.csv", regions)
    write_rows(out/"reports"/"reconstruction_equivalence.csv", reconstruction)
    write_rows(out/"reports"/"pc1_regional_associations.csv", physical)
    summary = {"coefficient_comparisons": comparisons, "regional_variance": regions,
               "reconstruction_comparisons": reconstruction, "pc1_regional_associations": physical,
               "all_coefficient_agreement_tests_pass": all(r["agreement_pass"] for r in comparisons),
               "all_reconstruction_tests_pass": all(r["reconstruction_pass"] and r["r_squared_sum_pass"] for r in reconstruction)}
    write_json(out/"reports"/"summary.json", summary)
    return summary


def turbulent_consistency(out: Path) -> list[dict]:
    rows = []
    for directory in ("northern_hemisphere", "southern_hemisphere"):
        for method in ("individual", "joint"):
            root = out/directory/method
            with xr.open_dataset(root/"latent.nc") as latent, xr.open_dataset(root/"sensible.nc") as sensible, \
                 xr.open_dataset(root/"turbulent.nc") as turbulent:
                for mode in range(12):
                    a, b, c = [ds.coefficient.isel(mode=mode).values for ds in (latent, sensible, turbulent)]
                    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
                    error = np.abs(c[valid]-(a[valid]+b[valid]))
                    scale = max(float(np.max(np.abs(c[valid]))) if valid.any() else 0, 1e-12)
                    rows.append({"hemisphere": directory, "method": method, "mode": mode+1,
                                 "valid_cells": int(valid.sum()), "max_absolute_difference": float(error.max()) if valid.any() else np.nan,
                                 "agreement_pass": bool(valid.any() and np.all(error <= COMPARE_TOL*scale))})
    write_rows(out/"reports"/"turbulent_linearity.csv", rows)
    return rows


def save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def quiver_samples(lat, lon, u, v, step_y, step_x):
    """Geographic arrows: normalised longitudes, finite components, no exact poles."""
    yy, xx = np.meshgrid(lat[::step_y], lon[::step_x], indexing="ij")
    u, v = u[::step_y, ::step_x], v[::step_y, ::step_x]
    valid = np.isfinite(u) & np.isfinite(v) & (np.abs(yy) < 90)
    return (xx[valid]+180)%360-180, yy[valid], u[valid], v[valid]


def map_panel(ax, data, lat, lon, title, limit=None, sequential=False, coastlines=True):
    import cartopy.crs as ccrs
    # Project exact native cell edges once per grid/projection, not once per
    # panel. Split the dateline cell into two matching half-cells; no values
    # are averaged, interpolated or downsampled. Drawing a projected QuadMesh
    # directly avoids Cartopy's repeated global seam detection/transformation.
    key = (np.asarray(lat).tobytes(), np.asarray(lon).tobytes(), str(ax.projection))
    if key not in _MESH_CACHE:
        normalized = (np.asarray(lon)+180)%360-180
        order = np.argsort(normalized)
        centres = normalized[order]
        # The primary ERA5 grid has a cell centred on the dateline. For
        # synthetic shifted grids, choose a cyclic start at its native edge.
        spacing = float(centres[1]-centres[0])
        if np.isclose(centres[0], -180):
            xedges = np.r_[-180, centres+spacing/2, 180]
            order = np.r_[order, order[0]]
        else:
            xedges = np.r_[centres-spacing/2, centres[-1]+spacing/2]
        half_lat = float(lat[1]-lat[0])/2
        yedges = np.clip(np.r_[np.asarray(lat)-half_lat, lat[-1]+half_lat], -90, 90)
        xx, yy = np.meshgrid(xedges, yedges)
        projected = ax.projection.transform_points(ccrs.PlateCarree(), xx, yy)
        _MESH_CACHE[key] = (projected[..., 0], projected[..., 1], order)
    xx, yy, order = _MESH_CACHE[key]
    data = np.asarray(data)[:, order]
    kwargs = {"vmin": 0, "vmax": limit or 1, "cmap": "viridis"} if sequential else {
        "norm": TwoSlopeNorm(vmin=-max(limit or 1e-15, 1e-30), vcenter=0, vmax=max(limit or 1e-15, 1e-30)), "cmap": "RdBu_r"}
    image = Axes.pcolormesh(ax, xx, yy, data, shading="flat", rasterized=True, **kwargs)
    ax.set_global()
    if coastlines:
        ax.coastlines(resolution="110m", linewidth=.4)
    ax.gridlines(linewidth=.3, alpha=.4)
    ax.set_title(title, fontsize=9)
    return image


def make_figures(out: Path, kinds: list[str], pc_diagnostics: dict, summary: dict,
                 dpi: int = 300, coastlines: bool = True, progress=print):
    import cartopy.crs as ccrs
    for hemi, directory in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        root = out/directory
        fig, ax = plt.subplots(figsize=(7, 6))
        c = np.array(pc_diagnostics[hemi]["correlation"])
        image = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=range(12), yticks=range(12), xticklabels=range(1, 13), yticklabels=range(1, 13),
               xlabel="PC", ylabel="PC", title=f"{hemi.upper()} actual regression sample; max off-diagonal |r|={pc_diagnostics[hemi]['maximum_offdiagonal_correlation']:.2e}")
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        fig.tight_layout()
        save(fig, root/"comparison"/"pc_correlation.png", dpi)
        for kind in kinds:
            progress(f"Figures: {hemi} {kind}")
            with xr.open_dataset(root/"individual"/f"{kind}.nc") as individual, \
                 xr.open_dataset(root/"joint"/f"{kind}.nc") as joint, \
                 xr.open_dataset(root/"comparison"/f"{kind}.nc") as comparison:
                lat, lon = joint.latitude.values, joint.longitude.values
                # One target's two 12-map coefficient collections, never time-space cubes.
                ai, aj = individual.coefficient.values, joint.coefficient.values
                limit = max(float(np.nanmax(np.abs(ai))), float(np.nanmax(np.abs(aj))), 1e-15)
                unit = str(joint.coefficient.attrs["units"])
                period = f"{joint.attrs['first_month']}–{joint.attrs['last_month']}"
                for method, maps in (("individual", ai), ("joint", aj)):
                    fig, axes = plt.subplots(4, 3, figsize=(15, 12), subplot_kw={"projection": ccrs.Robinson()})
                    for n, ax in enumerate(axes.flat):
                        image = map_panel(ax, maps[n], lat, lon, f"PC{n+1}", limit, coastlines=coastlines)
                    fig.suptitle(f"{hemi.upper()} PCs → global {kind.replace('_', ' ')} | {method} | {period}")
                    fig.subplots_adjust(left=.03, right=.9, top=.94, bottom=.04, hspace=.16)
                    fig.colorbar(image, ax=list(axes.flat), shrink=.6, label=unit)
                    save(fig, root/method/f"{kind}_modes_1_12.png", dpi)
                rows = [r for r in summary["coefficient_comparisons"] if r["hemisphere"] == hemi and r["variable"] == kind]
                comparison_figure(root, hemi, kind, rows, ai, aj, comparison, dpi, coastlines)
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()})
                image = map_panel(ax, joint.r_squared.values, lat, lon,
                                  f"{hemi.upper()} {kind} | joint 12-PC in-sample R² | {period}", 1, True, coastlines)
                fig.colorbar(image, ax=ax, shrink=.7, label="R²")
                save(fig, out/"variance_diagnostics"/f"{hemi}_{kind}_r_squared.png", dpi)
        if {"eastward_stress", "northward_stress"}.issubset(kinds):
            stress_vectors_figure(out, directory, hemi, dpi, coastlines)
    for kind in kinds:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, hemi in zip(axes, ("north", "south")):
            for region in REGIONS:
                rows = [r for r in summary["regional_variance"] if r["hemisphere"] == hemi and
                        r["variable"] == kind and r["region"] == region and r["mask_policy"] == "primary"]
                ax.plot(COUNTS, [r["total_variance_fraction_explained"]*100 for r in rows], marker="o", label=region.replace("_", " "))
            ax.set(title=f"{hemi.upper()} | {kind}", xlabel="Number of PCs", xticks=COUNTS, ylabel="Area-weighted total variance explained (%)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, out/"variance_diagnostics"/f"{kind}_retained_mode_comparison.png", dpi)
    physical_figure(out, kinds, dpi, coastlines)


def comparison_figure(root, hemi, kind, rows, ai, aj, comparison, dpi, coastlines):
    import cartopy.crs as ccrs
    lat, lon = comparison.latitude.values, comparison.longitude.values
    period = f"{comparison.attrs['first_month']}–{comparison.attrs['last_month']}"
    unit = comparison.coefficient_difference.attrs["units"]
    limit = max(float(np.nanmax(np.abs(ai))), float(np.nanmax(np.abs(aj))), 1e-15)
    worst = max(rows, key=lambda r: r["max_absolute_difference"])["mode"]-1
    selected = sorted({0, 2, 5, 11, worst})
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.0*len(selected)),
                             subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)
    # Set subplot geometry BEFORE colorbars shrink the parent axes.
    fig.subplots_adjust(top=.94, bottom=.05, hspace=.32, wspace=.3)
    for row, n in enumerate(selected):
        diff = comparison.coefficient_difference.isel(mode=n).values
        images = []
        for ax, data, label, lim in zip(axes[row], (ai[n], aj[n], diff),
            ("Individual", "Joint", "Joint − individual"), (limit, limit, max(float(np.nanmax(np.abs(diff))), 1e-30))):
            images.append(map_panel(ax, data, lat, lon, f"PC{n+1}: {label}", lim, coastlines=coastlines))
        fig.colorbar(images[0], ax=list(axes[row, :2]), shrink=.65, label=unit)
        fig.colorbar(images[2], ax=axes[row, 2], shrink=.65, label=unit)
        r = rows[n]
        axes[row, 2].text(.5, -.13, f"max |Δ|={r['max_absolute_difference']:.2e}; weighted RMSE={r['area_weighted_rmse']:.2e}",
                          transform=axes[row, 2].transAxes, ha="center", fontsize=7)
    fig.suptitle(f"{hemi.upper()} {kind} | {period} | difference scales are numerical, not physical signals")
    save(fig, root/"comparison"/f"{kind}_individual_joint.png", dpi)


def stress_vectors_figure(out: Path, directory: str, hemi: str, dpi: int, coastlines: bool):
    import cartopy.crs as ccrs
    root = out/directory
    with xr.open_dataset(root/"joint"/"eastward_stress.nc") as east, xr.open_dataset(root/"joint"/"northward_stress.nc") as north:
        lat, lon = east.latitude.values, east.longitude.values
        u, v = east.coefficient.values, north.coefficient.values
        magnitude = np.hypot(u, v)
        limit = max(float(np.nanmax(magnitude)), 1e-12)
        arrow = max(float(np.nanpercentile(magnitude, 95)), 1e-12)
        stepy = max(1, round(5/abs(lat[1]-lat[0])))
        stepx = max(1, round(7.5/abs(lon[1]-lon[0])))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), subplot_kw={"projection": ccrs.Robinson()})
        for n, ax in enumerate(axes.flat):
            im = map_panel(ax, magnitude[n], lat, lon, f"PC{n+1}", limit, True, coastlines)
            xx, yy, uu, vv = quiver_samples(lat, lon, u[n], v[n], stepy, stepx)
            quiver = ax.quiver(xx, yy, uu, vv, transform=ccrs.PlateCarree(), scale=arrow*20, width=.002)
            quiver.set_path_effects([path_effects.Stroke(linewidth=.6, foreground="white"), path_effects.Normal()])
            ax.quiverkey(quiver, .83, -.08, arrow, f"{arrow:.3g} N m⁻² / PC SD", labelpos="S", fontproperties={"size": 7})
        fig.suptitle(f"{hemi.upper()} joint stress regressions | positive saved-PC phase | arrows thinned; exact poles omitted")
        fig.subplots_adjust(top=.9, bottom=.08, hspace=.25)
        fig.colorbar(im, ax=list(axes.flat), shrink=.6, label="Stress magnitude, N m⁻² per PC SD")
        save(fig, root/"joint"/"stress_vectors_modes_1_6.png", dpi)


def physical_figure(out: Path, kinds: list[str], dpi: int, coastlines: bool):
    """Compare measured AO-like/SAM-like PC1 heat and vector associations."""
    import cartopy.crs as ccrs
    heat = [k for k in ("latent", "sensible", "turbulent") if k in kinds]
    has_stress = {"eastward_stress", "northward_stress"}.issubset(kinds)
    columns = len(heat)+int(has_stress)
    if not columns:
        return
    data = {}
    for hemi, directory in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        for kind in heat+(["eastward_stress", "northward_stress"] if has_stress else []):
            with xr.open_dataset(out/directory/"joint"/f"{kind}.nc") as ds:
                data[hemi, kind] = ds.coefficient.isel(mode=0).values
                lat, lon = ds.latitude.values, ds.longitude.values
    fig, axes = plt.subplots(2, columns, figsize=(5*columns, 7), squeeze=False,
                             subplot_kw={"projection": ccrs.Robinson()})
    fig.subplots_adjust(top=.88, bottom=.12, hspace=.25, wspace=.35)
    for col, kind in enumerate(heat):
        limit = max(float(np.nanmax(np.abs(data[h, kind]))) for h in ("north", "south"))
        for row, hemi in enumerate(("north", "south")):
            image = map_panel(axes[row, col], data[hemi, kind], lat, lon,
                              f"{'AO-like NH' if hemi == 'north' else 'SAM-like SH'} PC1: {kind}", limit, coastlines=coastlines)
        fig.colorbar(image, ax=list(axes[:, col]), shrink=.65, label="W m⁻² per PC SD; downward positive")
    if has_stress:
        mags = {h: np.hypot(data[h, "eastward_stress"], data[h, "northward_stress"]) for h in ("north", "south")}
        limit = max(float(np.nanmax(v)) for v in mags.values())
        arrow = max(float(np.nanpercentile(v, 95)) for v in mags.values())
        sy = max(1, round(5/abs(lat[1]-lat[0])))
        sx = max(1, round(7.5/abs(lon[1]-lon[0])))
        for row, hemi in enumerate(("north", "south")):
            ax = axes[row, -1]
            image = map_panel(ax, mags[hemi], lat, lon, f"{hemi.upper()} PC1: stress", limit, True, coastlines)
            xx, yy, uu, vv = quiver_samples(lat, lon, data[hemi, "eastward_stress"], data[hemi, "northward_stress"], sy, sx)
            q = ax.quiver(xx, yy, uu, vv, transform=ccrs.PlateCarree(), scale=arrow*20, width=.002)
            q.set_path_effects([path_effects.Stroke(linewidth=.6, foreground="white"), path_effects.Normal()])
            ax.quiverkey(q, .8, -.1, arrow, f"{arrow:.3g} N m⁻² / PC SD", labelpos="S", fontproperties={"size": 7})
        fig.colorbar(image, ax=list(axes[:, -1]), shrink=.65, label="N m⁻² per PC SD")
    fig.suptitle("Calculated global associations with positive saved AO-like/SAM-like PC1 | joint regression")
    save(fig, out/"reports"/"ao_sam_surface_associations.png", dpi)


def write_report(out: Path, metadata: dict, summary: dict, linearity: list[dict]):
    comparisons = summary["coefficient_comparisons"]
    lines = ["# MSLP PC–surface regression analysis", "", "## Scope and provenance", "",
        f"Common sample: **{metadata['first_month']}–{metadata['last_month']}**, {metadata['n_samples']} months. "
        "Independent NH and SH models use the original undetrended 12-mode 1979 EOF bases. "
        "All surface targets are global on the native grid. Saved PC signs and training-period scaling are preserved.", "",
        "Monthly climatologies are calculated over the common interval. No detrending, smoothing, filtering, regridding, "
        "regularisation or EOF recomputation is used. Dates are joined by calendar year and month, not timestamp hour.", "",
        "Input paths, fingerprints, parameter IDs, units, expver labels, software versions, command, masks and numerical "
        "tolerances are in [metadata.json](metadata.json). Full numerical results are in [summary.json](summary.json).", "",
        "## Predictor checks and numerical equivalence", "",
        "| Hemisphere | Max off-diagonal correlation | Rank | Condition number |", "|---|---:|---:|---:|"]
    for hemi, diag in metadata["pc_diagnostics"].items():
        lines.append(f"| {hemi} | {diag['maximum_offdiagonal_correlation']:.3e} | {diag['rank']} | {diag['condition_number']:.12g} |")
    worst = max(comparisons, key=lambda r: r["max_absolute_difference"])
    lines += ["", f"All coefficient agreement tests pass: **{summary['all_coefficient_agreement_tests_pass']}**. "
        f"All reconstruction and R²-additivity tests pass: **{summary['all_reconstruction_tests_pass']}**.", "",
        f"Largest absolute coefficient difference: {worst['max_absolute_difference']:.3e} "
        f"({worst['hemisphere']}, {worst['variable']}, PC{worst['mode']}). Differences across physical variables have "
        "different units; consult the per-field normalized errors and tolerances before comparing them.", "",
        "For centred orthogonal predictors, the Gram matrix is diagonal and individual OLS slopes equal joint slopes. "
        "The joint centred prediction equals the sum of individual centred contributions, and individual R² values add. "
        "An overall intercept is added once. These identities may fail after sample truncation or with correlated predictors; "
        "the general identity b_individual = diag(G)⁻¹ G b_joint is also checked. PCs are never reorthogonalised.", "",
        "[Coefficient comparison table](individual_joint_equivalence.csv); "
        "[reconstruction comparisons](reconstruction_equivalence.csv).", "", "## Explained surface variability", "",
        "Percentages below are fractions of total area-weighted field variance, not area-weighted means of local R². "
        "Both statistics, pure-ocean coastal sensitivity, all regions and nested 1/3/6/12-mode fits appear in "
        "[regional_variance.csv](../variance_diagnostics/regional_variance.csv).", "",
        "| Hemisphere | Variable | Global variance explained, 12 PCs (%) | Gain from PCs 7–12 (percentage points) | Unexplained (%) |",
        "|---|---|---:|---:|---:|"]
    for hemi in ("north", "south"):
        kinds = sorted({r["variable"] for r in comparisons})
        for kind in kinds:
            rows = [r for r in summary["regional_variance"] if r["hemisphere"] == hemi and r["variable"] == kind and r["region"] == "global_ocean" and r["mask_policy"] == "primary"]
            frac = {r["retained_modes"]: r["total_variance_fraction_explained"] for r in rows}
            lines.append(f"| {hemi} | {kind} | {100*frac[12]:.3f} | {100*(frac[12]-frac[6]):.3f} | {100*(1-frac[12]):.3f} |")
    lines += ["", "Pressure-variance rank is not necessarily surface-variance rank. A descriptive ranking of individual "
        "mode associations is provided below, based on area-weighted reconstruction variance over the global ocean.", ""]
    for hemi, directory in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        for kind in sorted({r["variable"] for r in comparisons}):
            with xr.open_dataset(out/directory/"individual"/f"{kind}.nc") as ds:
                area = cell_areas(ds.latitude.values, ds.longitude.values)
                variance = np.array(metadata["pc_diagnostics"][hemi]["sample_variance"])
                strength = np.array([weighted_mean(ds.coefficient.isel(mode=n).values**2, area)*variance[n] for n in range(12)])
                order = np.argsort(strength)[::-1][:3]+1
                lines.append(f"- {hemi.upper()} {kind}: strongest individual associations are PCs {', '.join(map(str, order))}.")
    lines += ["", "## Calculated PC1 regional patterns", "",
        "These signed area means describe the broad heat-exchange direction associated with a one-SD positive "
        "saved AO-like/SAM-like PC1. A positive heat value means reduced upward loss or increased downward exchange; "
        "a negative value means greater upward loss. Spatial cancellation can make regional means small even when "
        "local patterns are strong. Stress directions and local structures should be read from the vector maps. "
        "All component means and RMS amplitudes are in [pc1_regional_associations.csv](pc1_regional_associations.csv).", "",
        "| Hemisphere | Heat variable | Northern extratropics | Tropics | Southern extratropics |",
        "|---|---|---:|---:|---:|"]
    for hemi in ("north", "south"):
        for kind in ("latent", "sensible", "turbulent"):
            rows = [r for r in summary["pc1_regional_associations"] if r["hemisphere"] == hemi and r["variable"] == kind]
            if rows:
                means = {r["region"]: r["pc1_area_mean_coefficient"] for r in rows}
                lines.append(f"| {hemi} | {kind} | {means['northern_extratropics']:+.3f} | {means['tropics']:+.3f} | {means['southern_extratropics']:+.3f} |")
    lines += ["", "Values are W m⁻² per original PC SD. These broad regions do not establish a unique mechanism; "
        "regional signs must be interpreted with the full spatial maps and the retained sea-ice policy.", "",
        "## Masks, signs and missing data", "",
        "Ocean cells have ERA5 land fraction <0.5. Mixed coastal cells are retained; summaries also show the ≤0.05 "
        "land-fraction subset. The repeated mask is checked for temporal invariance. Cells missing any common month "
        "are excluded from both regression methods; finite and accepted sample counts are saved. SST is not an input.", "",
        "Sea-ice-covered ocean locations are retained with a fixed mask. Fluxes there describe surface exchange that "
        "may be through ice; they must not be interpreted as direct open-water ocean heating. No ice-specific separation "
        "is claimed without an ice-concentration ancillary dataset.", "",
        "Positive latent/sensible heat anomalies mean increased downward exchange or reduced upward loss. Negative latent "
        "anomalies mean enhanced evaporative loss. Stress is positive eastward/northward. Mean-rate inputs are not divided "
        "by time; verified daily-mean monthly accumulated alternatives are divided by 86400 seconds.", "",
        "## Turbulent heat exchange", ""]
    if linearity:
        lines.append(f"Latent plus sensible heat flux is analysed on their common complete-cell mask. All coefficient "
                     f"linearity tests pass: **{all(r['agreement_pass'] for r in linearity)}**. "
                     "Its R² is computed from the summed target, including component covariance, never by adding component R². "
                     "See [turbulent_linearity.csv](turbulent_linearity.csv).")
    else:
        lines.append("The optional turbulent sum was not requested.")
    lines += ["", "## Figures and physical interpretation", "",
        "Individual and joint coefficient figures use the same physical colour limits. Difference maps have their own "
        "numerical scales; roundoff patterns are not physical signals. Stress arrows use the geographic components, a "
        "shared reference scale and display-only thinning. Global targets are shown for both predictor hemispheres.", "",
        "Regional interpretation should use the calculated PC1 maps and their saved positive AO/SAM convention. "
        "The report's mode ranking and regional variance summaries are calculated from these data; they do not assign "
        "causal forcing or named climate identities to higher PCs.", "", "## Limitations", "",
        "These are in-sample statistical associations. Numerical equivalence does not establish high explanatory power "
        "or predictive skill. Flux anomalies include SST feedback; radiation is excluded. Turbulent exchange and its "
        "residual are not the complete stochastic forcing F′. Stress is not converted into Ekman transport or pumping. "
        "Monthly means omit shorter variability. NH and SH components can overlap and must not be added as a combined model.", "",
        "Strict predictive validation would fit EOFs, climatologies and regressions within training data only. "
        "No IID significance claims or unbiased residual uncertainty estimates are made; saved residual sample variance "
        "is descriptive and does not adjust for monthly-climatology estimation or serial correlation.", ""]
    if (out/"reports"/"ao_sam_surface_associations.png").exists():
        lines += ["## Figure navigation", "", "![Calculated AO-like/SAM-like surface associations](ao_sam_surface_associations.png)", "",
                  "[Northern Hemisphere maps](../northern_hemisphere/) · "
                  "[Southern Hemisphere maps](../southern_hemisphere/) · "
                  "[Explained-variance figures](../variance_diagnostics/)", ""]
    (out/"reports"/"surface_regression_report.md").write_text("\n".join(lines))
