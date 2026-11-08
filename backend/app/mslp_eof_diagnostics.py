"""Figures, tables and evidence-based reports for the MSLP EOF workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import xarray as xr

from app.mslp_eof import latitude_weights

LOG = logging.getLogger(__name__)


def write_csv(path: Path, columns: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def save_dataset(ds: xr.Dataset, path: Path) -> None:
    encoding = {key: {"zlib": True, "complevel": 4} for key in ds.data_vars}
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def _save(fig, path: Path, dpi: int) -> None:
    # Fixed canvas avoids Cartopy/Matplotlib tight-bbox clipping of polar axes.
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _map(ax, values, ds, hemisphere, title, limit, coastlines):
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    data, lon = add_cyclic_point(values, coord=ds.longitude.values)
    image = ax.pcolormesh(lon, ds.latitude.values, data, transform=ccrs.PlateCarree(),
                          shading="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), rasterized=True)
    ax.set_extent([-180, 180, 20, 90] if hemisphere == "north" else
                  [-180, 180, -90, -20], crs=ccrs.PlateCarree())
    if coastlines:
        ax.coastlines(resolution="110m", linewidth=0.5)
    angles = np.linspace(0, 2*np.pi, 181)
    ax.set_boundary(MatplotlibPath(np.column_stack([np.sin(angles), np.cos(angles)])*.5+.5),
                    transform=ax.transAxes)
    ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.45,
                 xlocs=np.arange(-180, 181, 60), ylocs=[-80, -60, -40, -20, 20, 40, 60, 80])
    # Explicit geographic labels and axes-relative headings avoid automatic
    # polar gridliner/title positioning differences across Matplotlib releases.
    sign = 1 if hemisphere == "north" else -1
    for longitude in range(-180, 180, 60):
        label = f"{abs(longitude)}°" + ("W" if longitude < 0 and longitude != -180 else
                                       "E" if longitude > 0 else "")
        ax.text(longitude, sign*24, label, transform=ccrs.PlateCarree(), fontsize=6,
                ha="center", va="center", bbox={"facecolor": "white", "alpha": .7, "pad": .3, "edgecolor": "none"})
    for latitude in (40, 60, 80):
        ax.text(135, sign*latitude, f"{latitude}°{'N' if sign == 1 else 'S'}",
                transform=ccrs.PlateCarree(), fontsize=6, ha="center", va="center",
                bbox={"facecolor": "white", "alpha": .7, "pad": .3, "edgecolor": "none"})
    ax.text(.5, 1.04, title, transform=ax.transAxes, fontsize=9, ha="center", va="bottom")
    return image


def plot_maps(ds: xr.Dataset, hemisphere: str, path: Path, dpi: int,
              coastlines: bool = True) -> None:
    import cartopy.crs as ccrs
    projection = ccrs.NorthPolarStereo() if hemisphere == "north" else ccrs.SouthPolarStereo()
    n = ds.sizes["mode"]
    fig, axes = plt.subplots((n+2)//3, 3, figsize=(13, 4.3*((n+2)//3)),
                             subplot_kw={"projection": projection}, squeeze=False)
    limit = max(float(np.abs(ds.regression_map).max()), 1e-10)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        image = _map(ax, ds.regression_map.values[k], ds, hemisphere,
                     f"EOF {k+1}: {ds.explained_variance_percent.values[k]:.2f}%", limit, coastlines)
    fig.suptitle(f"{hemisphere.capitalize()} hemisphere | {ds.attrs['first_month']}–{ds.attrs['last_month']}\n"
                 "MSLP regression on standardised PC")
    fig.subplots_adjust(top=.84, bottom=.06, left=.04, right=.93, hspace=.28, wspace=.16)
    fig.colorbar(image, ax=list(axes.flat), shrink=0.65, pad=0.07,
                 label="hPa per one-standard-deviation PC")
    _save(fig, path, dpi)


def plot_diagnostics(ds: xr.Dataset, directory: Path, dpi: int,
                     running_mean: bool = True) -> None:
    n = ds.sizes["mode"]
    label = f"{ds.attrs['hemisphere'].capitalize()} | {ds.attrs['first_month']}–{ds.attrs['last_month']}"
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.8*n), sharex=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        pc = ds.standardized_pc.isel(mode=k)
        ax.plot(ds.time.values, pc, color="0.55", linewidth=0.5, label="Monthly")
        if running_mean:
            ax.plot(ds.time.values, pc.rolling(time=12, center=True, min_periods=12).mean(),
                    color="#174f79", linewidth=1.1, label="12-month centred mean")
        ax.axhline(0, color="0.3", linewidth=0.5)
        ax.set_ylabel(f"EOF {k+1}\nPC (SD)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Calendar year")
    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, directory / "principal_components.png", dpi)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ds.mode, ds.explained_variance_percent, "o-")
    axes[0].set_ylabel("Explained variance (%)")
    axes[1].plot(ds.mode, ds.cumulative_variance_percent, "o-")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    for ax in axes:
        ax.set_xlabel("EOF mode")
        ax.set_xticks(ds.mode.values)
        ax.grid(alpha=0.25)
    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, directory / "variance.png", dpi)
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(n):
        ax.plot(ds.lag, ds.acf.values[:, k], label=f"EOF {k+1}")
    ax.axhline(0, color="0.3", linewidth=0.5)
    ax.axhline(np.exp(-1), color="0.5", linestyle=":", label="1/e")
    ax.set(xlabel="Lag (months)", ylabel="Sample autocorrelation", title=label)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    _save(fig, directory / "autocorrelation.png", dpi)


STATISTICS = ["eigenvalue", "explained_variance_percent", "cumulative_variance_percent",
              "lag_one_acf", "first_efold_crossing_months", "effective_sample_size_heuristic",
              "north_error_iid", "north_error_adjusted_heuristic", "next_eigenvalue",
              "separated_from_next_heuristic"]


def pattern_structure(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Describe annular structure without assigning atmospheric mode identities.

    Band means use cos(latitude); zonal energy is the area-weighted squared
    zonal-mean loading divided by total squared regression loading (not a
    fraction of temporal variance).
    """
    maps = ds.regression_map.values
    absolute_latitude = np.abs(ds.latitude.values)
    weights = latitude_weights(ds.latitude.values)**2
    zonal = maps.mean(axis=2)
    energy = np.sum(weights[None, :]*zonal**2, axis=1) * ds.sizes["longitude"]
    total = np.sum(weights[None, :, None]*maps**2, axis=(1, 2))
    values = {"zonal_loading_energy_fraction": energy/total}
    for label, low, high in (("polar_cap_hpa", 65, 90), ("midlatitude_band_hpa", 35, 55)):
        band = (absolute_latitude >= low) & (absolute_latitude <= high) & (weights > 0)
        values[label] = (np.average(zonal[:, band], weights=weights[band], axis=1)
                         if np.any(band) else np.full(len(maps), np.nan))
    return values


def export_analysis(ds: xr.Dataset, directory: Path, hemisphere: str, dpi: int = 300,
                    maps: bool = True, coastlines: bool = True,
                    running_mean: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_dataset(ds, directory / "eof.nc")
    write_csv(directory / "variance_and_persistence.csv",
              {"mode": ds.mode.values, **{name: ds[name].values for name in STATISTICS}})
    write_csv(directory / "pattern_structure.csv", {"mode": ds.mode.values, **pattern_structure(ds)})
    write_csv(directory / "standardized_pcs.csv",
              {"time": ds.time.values, **{f"EOF_{k+1}": ds.standardized_pc.values[:, k]
                                          for k in range(ds.sizes["mode"])}})
    plot_diagnostics(ds, directory, dpi, running_mean)
    if maps:
        plot_maps(ds, hemisphere, directory / "eof_regression_maps.png", dpi, coastlines)


def export_comparison(full: xr.Dataset, modern: xr.Dataset, comparison: xr.Dataset,
                      directory: Path, hemisphere: str, dpi: int = 300,
                      maps: bool = True, coastlines: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_dataset(comparison, directory / f"{hemisphere}_comparison.nc")
    names = [name for name in comparison.data_vars if comparison[name].dims == ("match",)]
    write_csv(directory / f"{hemisphere}_matching.csv", {name: comparison[name].values for name in names})
    np.savetxt(directory / f"{hemisphere}_spatial_correlation_matrix.csv",
               comparison.spatial_correlation_matrix.values, delimiter=",",
               header="rows=full modes; columns=modern modes")
    indices = comparison.matched_modern_mode.values.astype(int)-1
    rows = comparison.full_mode_number.values.astype(int)-1
    signs = comparison.modern_sign.values
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(rows))
    ax.bar(positions-0.2, comparison.full_variance_percent, 0.4, label=full.attrs["first_month"])
    ax.bar(positions+0.2, comparison.modern_variance_percent, 0.4, label=modern.attrs["first_month"])
    ax.set_xticks(positions, [f"{i+1} ↔ {j+1}" for i, j in zip(rows, indices)])
    ax.set(xlabel="Matched EOF: full ↔ modern", ylabel="Explained variance (%)", title=hemisphere.capitalize())
    ax.legend()
    fig.tight_layout()
    _save(fig, directory / f"{hemisphere}_variance_comparison.png", dpi)
    a, b = xr.align(full.standardized_pc, modern.standardized_pc, join="inner")
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2*len(rows)), sharex=True, squeeze=False)
    for ax, i, j, s, r in zip(axes.flat, rows, indices, signs, comparison.common_pc_correlation.values):
        ax.plot(a.time.values, a.values[:, i], linewidth=0.6, label="Full basis")
        ax.plot(b.time.values, b.values[:, j]*s, linewidth=0.6, alpha=0.7, label="Modern basis, aligned")
        ax.set_ylabel(f"{i+1} ↔ {j+1}\nPC (SD)")
        ax.set_title(f"Common-interval r = {r:.3f}", fontsize=8)
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Calendar year")
    fig.suptitle(f"{hemisphere.capitalize()}: separate period climatologies and PC normalisations")
    fig.tight_layout()
    _save(fig, directory / f"{hemisphere}_common_pcs.png", dpi)
    if maps:
        import cartopy.crs as ccrs
        projection = ccrs.NorthPolarStereo() if hemisphere == "north" else ccrs.SouthPolarStereo()
        for i, j, sign, r in zip(rows, indices, signs, comparison.aligned_spatial_correlation.values):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": projection})
            fields = (full.regression_map.values[i], modern.regression_map.values[j]*sign)
            limit = max(float(np.max(np.abs(x))) for x in fields)
            for ax, values, ds, k in zip(axes, fields, (full, modern), (i, j)):
                image = _map(ax, values, ds, hemisphere,
                             f"EOF {k+1}: {ds.explained_variance_percent.values[k]:.2f}%\n"
                             f"{ds.attrs['first_month']}–{ds.attrs['last_month']}", limit, coastlines)
            fig.suptitle(f"{hemisphere.capitalize()} | matched spatial r = {r:.3f}")
            fig.subplots_adjust(top=.78, bottom=.08, left=.04, right=.94, wspace=.2)
            fig.colorbar(image, ax=list(axes), shrink=0.65, pad=0.08, label="hPa per one-standard-deviation PC")
            _save(fig, directory / f"{hemisphere}_full_{i+1}_modern_{j+1}.png", dpi)


def interpretation_scope(results: dict, metadata: dict) -> dict:
    """Bind manually inspected interpretation to exact inputs and computed bases."""
    return {"files": metadata["files"], "analyses": [
        {"hemisphere": h, "start_year": int(year),
         "first_month": ds.attrs["first_month"], "last_month": ds.attrs["last_month"],
         "detrended": int(ds.attrs["detrended"]),
         "regression_sha256": hashlib.sha256(ds.regression_map.values.tobytes()).hexdigest(),
         "eigenvalues": ds.eigenvalue.values.tolist()}
        for (h, year), ds in sorted(results.items())]}


def write_report(results: dict, comparisons: dict, metadata: dict, directory: Path,
                 summaries: list[tuple[str, str]] | None = None) -> Path:
    """Report only computed quantities; mode labels require independent validation."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# ERA5 monthly MSLP EOF analysis", "", "## 1. Research objective", "",
             "Identify extratropical pressure variability separately in 20–90°N and 90–20°S, "
             "and assess stability across historical periods. Both land and ocean are included. "
             "No stochastic heat-flux forcing F′ is estimated.", "", "## 2. Dataset", "",
             f"Input coverage: {metadata['first_month']} through {metadata['last_month']}. "
             f"Native grid spacing: {metadata['grid_spacing_deg']:g}°. "
             "Exact input paths, sizes, modification times, dimensions, expver and source attributes "
             "are recorded in the adjacent metadata JSON. No downsampling, interpolation or smoothing "
             "is applied to the analysed pressure fields.", "", "## 3. Preprocessing", "",
             "Pressure is converted to hPa using its declared units. Every analysis requires a "
             "continuous monthly record beginning in January of its requested year and containing "
             "all twelve calendar months. A partial final year is retained. Missing values, duplicate "
             "months and gaps cause an error rather than interpolation. Complementary expver values "
             "are merged; conflicting finite overlaps cause an error.", "",
             "Each period has its own twelve monthly climatologies. Differences therefore describe "
             "variability about different seasonal baselines, not a change of mean pressure. "
             "Optional detrending is recorded per analysis; the primary default retains trends and "
             "decadal variability. When requested, monthly climatology and linear trend are fitted "
             "jointly to avoid reintroducing seasonal means.", "", "## 4. EOF methodology", "",
             "Anomalies are multiplied by sqrt(cos(latitude)). Covariance EOFs use the exact temporal "
             "Gram matrix, accumulated in float64 over spatial slabs. No grid-cell standard deviations "
             "are divided out. Eigenvalues use T−1 and percentages use total variance across all modes. "
             "Weighted EOF vectors are orthonormal; PCs have variance equal to their eigenvalue. "
             "Standardised PCs have sample variance one. Regression maps are unweighted pressure "
             "regressions in hPa per one-standard-deviation PC. Multiplying regressions by standardised "
             "PCs reconstructs the retained anomaly component, including the pole without division by zero. "
             "The exact pole has zero area weight. EOF signs initially make the largest absolute "
             "regression loading positive. Before export, PC1 and its paired EOF and regression "
             "map are oriented so positive PC1 means lower 65–90° cap pressure relative to the "
             "35–55° band, following AO/SAM. Higher modes keep the initial convention; sign "
             "alignment alone does not establish mode identity.", ""]
    for number, hemisphere in ((5, "north"), (6, "south")):
        name = "Northern" if hemisphere == "north" else "Southern"
        lines.extend([f"## {number}. {name} Hemisphere results", ""])
        selected = [(year, ds) for (h, year), ds in results.items() if h == hemisphere]
        if not selected:
            lines.extend(["Not requested in this run.", ""])
        for year, ds in selected:
            lines.extend([f"### {ds.attrs['first_month']} to {ds.attrs['last_month']}", "",
                          f"{ds.sizes['time']} months; detrending enabled: {bool(ds.attrs['detrended'])}.", "",
                          "| EOF | Eigenvalue (hPa²) | Variance (%) | Cumulative (%) | Lag-1 ACF | First 1/e crossing (months) |",
                          "|---|---:|---:|---:|---:|---:|"])
            for k in range(ds.sizes["mode"]):
                crossing = ds.first_efold_crossing_months.values[k]
                lines.append(f"| {k+1} | {ds.eigenvalue.values[k]:.4g} | "
                             f"{ds.explained_variance_percent.values[k]:.3f} | "
                             f"{ds.cumulative_variance_percent.values[k]:.3f} | "
                             f"{ds.lag_one_acf.values[k]:.3f} | "
                             f"{int(crossing) if np.isfinite(crossing) else 'not reached'} |")
            separation = bool(ds.separated_from_next_heuristic.values[0])
            structure = pattern_structure(ds)
            lines.extend(["", "Spatial structure of the physical regression maps:", "",
                          "| EOF | 65–90° cap mean (hPa/SD) | 35–55° band mean (hPa/SD) | Zonal loading energy fraction |",
                          "|---|---:|---:|---:|"])
            for k in range(ds.sizes["mode"]):
                lines.append(f"| {k+1} | {structure['polar_cap_hpa'][k]:+.3f} | "
                             f"{structure['midlatitude_band_hpa'][k]:+.3f} | "
                             f"{structure['zonal_loading_energy_fraction'][k]:.3f} |")
            strongest = int(np.argmax(structure["zonal_loading_energy_fraction"]))
            cap = structure["polar_cap_hpa"][strongest]
            mid = structure["midlatitude_band_hpa"][strongest]
            lines.extend(["", f"Among retained modes, EOF {strongest+1} has the largest zonal loading "
                          f"energy fraction ({structure['zonal_loading_energy_fraction'][strongest]:.3f}). "
                          f"Its cap and midlatitude band loadings {'have opposite signs' if cap*mid < 0 else 'do not have opposite signs'}. "
                          "Opposite signs support a polar–midlatitude pressure contrast; the zonal energy "
                          "fraction measures longitudinal coherence of the regression loading, not temporal "
                          "variance explained. These summaries support map interpretation without assigning "
                          "AO, NAO or SAM identities."])
            lines.extend(["", f"EOF 1 heuristic North intervals {'do not overlap' if separation else 'overlap'} "
                          "with EOF 2. This is a sampling diagnostic, not a formal significance test.", "",
                          f"[Figures, tables and NetCDF](../{'northern' if hemisphere == 'north' else 'southern'}_hemisphere/"
                          f"{year}_{ds.attrs['last_month'][:4]}/)", ""])
            map_relative = (f"../{'northern' if hemisphere == 'north' else 'southern'}_hemisphere/"
                            f"{year}_{ds.attrs['last_month'][:4]}/eof_regression_maps.png")
            if (directory / map_relative).exists():
                lines.extend([f"![{name} regression maps, {year} start]({map_relative})", ""])
    lines.extend(["## 7. Historical-period comparison", "",
                  "A maximum-total-absolute-correlation assignment matches retained regression maps using "
                  "area-weighted spatial Pearson correlation. Modern maps and PCs are sign-aligned for "
                  "comparison only; source NetCDF bases retain their exported sign convention. The matching "
                  "is restricted to retained modes and need not recover a mode outside that set. "
                  "Variance differences use each period's own total variance. Common-period PC correlations "
                  "use each period's own EOF basis, climatology and full-period SD. These are descriptive "
                  "correlations, not coefficients of the same physical basis.", ""])
    for hemisphere, comparison in comparisons.items():
        lines.extend([f"### {hemisphere.capitalize()} hemisphere", "",
                      "| Full EOF | Modern EOF | Aligned spatial r | Common PC r | Modern−full variance (percentage points) |",
                      "|---|---|---:|---:|---:|"])
        for k in range(comparison.sizes["match"]):
            lines.append(f"| {int(comparison.full_mode_number.values[k])} | "
                         f"{int(comparison.matched_modern_mode.values[k])} | "
                         f"{comparison.aligned_spatial_correlation.values[k]:.3f} | "
                         f"{comparison.common_pc_correlation.values[k]:.3f} | "
                         f"{comparison.variance_difference_percentage_points.values[k]:+.3f} |")
        lines.extend(["", "Retained-subspace principal angles (degrees): " +
                      ", ".join(f"{v:.2f}" for v in comparison.principal_angle_degrees.values) + ". "
                      "Small angles indicate similar subspaces even where individual modes rotate or reorder.", ""])
        ranked = np.argsort(-comparison.aligned_spatial_correlation.values)
        lines.extend([f"Within this retained set, full EOF {int(comparison.full_mode_number.values[ranked[0]])} "
                      f"has the highest matched spatial correlation ({comparison.aligned_spatial_correlation.values[ranked[0]]:.3f}); "
                      f"full EOF {int(comparison.full_mode_number.values[ranked[-1]])} has the lowest "
                      f"({comparison.aligned_spatial_correlation.values[ranked[-1]]:.3f}). "
                      "This ordering does not establish a statistically significant change.", ""])
    lines.extend(["## 8. Physical interpretation", "",
                  "Interpret regression maps against Thompson and Wallace (2000), *Annular Modes in "
                  "the Extratropical Circulation. Part I: Month-to-Month Variability*, "
                  "[doi:10.1175/1520-0442(2000)013<1000:AMITEC>2.0.CO;2](https://doi.org/10.1175/1520-0442%282000%29013%3C1000%3AAMITEC%3E2.0.CO%3B2). "
                  "A north polar versus midlatitude pressure contrast is an AO candidate; an Atlantic "
                  "Iceland–Azores dipole is an NAO candidate. A south polar versus circumpolar midlatitude "
                  "contrast is a SAM candidate. Inspect longitudinal coherence, centres of action and "
                  "independent indices before naming a mode. No automated AO/NAO/SAM identifications "
                  "are made by this report. Statistical patterns do not establish physical causes.", ""])
    interpretation = directory / "interpretation_notes.md"
    scope = directory / "interpretation_scope.json"
    if (interpretation.exists() and scope.exists() and
            json.loads(scope.read_text()) == interpretation_scope(results, metadata)):
        lines.extend([interpretation.read_text(), ""])
    lines.extend(["## 9. Uncertainty and limitations", "",
                  "The 20° cutoff, latitude weighting and inclusion of land determine the covariance "
                  "metric. Year-round sampling combines seasonally different circulation regimes. Early "
                  "ERA5 circulation, particularly Southern Hemisphere observations before satellite coverage, "
                  "is less constrained. Comparing nested periods is not an independent-samples test. "
                  "Closely spaced eigenvalues allow rotations and reordering, so consider principal angles "
                  "alongside matched modes. No bootstrap confidence intervals or significance claims are made.", "",
                  "North et al. (1982), *Sampling Errors in the Estimation of Empirical Orthogonal Functions*, "
                  "[doi:10.1175/1520-0493(1982)110<0699:SEITEO>2.0.CO;2](https://doi.org/10.1175/1520-0493%281982%29110%3C0699%3ASEITEO%3E2.0.CO%3B2), "
                  "motivates the eigenvalue ± lambda sqrt(2/N) diagnostic. Both IID and heuristic "
                  "effective-sample errors are stored. Effective N uses the positive sequence of PC ACF "
                  "up to the requested lag, stops at the first nonpositive lag, and is bounded to [2,T]. "
                  "Adjacent error intervals use the current mode's N for both eigenvalues; this is an "
                  "approximation. Truncation, seasonality, retained trends and nonstationarity limit this diagnostic. "
                  "The first sampled 1/e crossing is a whole-month descriptive timescale, not an exponential "
                  "fit; oscillatory modes need not be AR1. Monthly means cannot resolve a ten-day atmospheric "
                  "decorrelation timescale.", "",
                  "## 10. Relevance to stochastic SST forcing", "",
                  "These circulation PCs can later be compared with independently calculated surface heat-flux "
                  "anomalies, including lag structure, seasonality and shared SST feedback. Pressure alone "
                  "does not identify F′ in Qs′ = F′ − gamma Theta′. A future forcing model would need heat-flux "
                  "data, a defensible feedback separation and out-of-sample validation; none is implemented here.", ""])
    path = directory / "mslp_eof_report.md"
    for number, (title, body) in enumerate(summaries or [], start=11):
        lines.extend([f"## {number}. {title}", "", body, ""])
    path.write_text("\n".join(lines))
    return path
