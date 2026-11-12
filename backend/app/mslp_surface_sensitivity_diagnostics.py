"""Figures and calculated scientific report for the two sensitivity experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from app.mslp_surface_regression import COUNTS
from app.mslp_surface_regression_diagnostics import map_panel, quiver_samples, save, write_json, write_rows
from app.mslp_surface_sensitivity import HEMISPHERES, MODELS, SEASONS


def read_rows(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


def select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def headline_region(hemisphere):
    return "northern_extratropics" if hemisphere == "north" else "southern_extratropics"


def check_outputs(out, baseline, rows, metadata):
    baseline_rows = read_rows(baseline/"variance_diagnostics/regional_variance.csv")
    errors = []
    for r in select(rows, model="year_round", evaluation="monthly_full"):
        b = select(baseline_rows, **{key: r[key] for key in ("hemisphere", "variable", "retained_modes", "region", "mask_policy")})[0]
        for name in ("total_variance_fraction_explained", "mean_local_r_squared", "area_weighted_rmse"):
            error = abs(r[name]-b[name])
            if not np.isclose(r[name], b[name], atol=1e-10*max(abs(b[name]), 1), rtol=1e-10):
                raise ValueError(f"Regional baseline consistency failed: {r['hemisphere']} {r['variable']} {name}")
            errors.append(error)
    linearity = []
    for hemi in HEMISPHERES:
        root = out/"seasonal_regressions"/hemi
        with xr.open_dataset(baseline/HEMISPHERES[hemi]/"joint/turbulent.nc") as target:
            target_sd_scale = max(float(np.sqrt(target.target_variance.max())), 1e-12)
        with xr.open_dataset(root/"latent_coefficients.nc") as l, xr.open_dataset(root/"sensible_coefficients.nc") as h, xr.open_dataset(root/"turbulent_coefficients.nc") as q:
            for k in COUNTS:
                for variable in (f"coefficient_{k}", f"intercept_{k}"):
                    maximum, scale = 0., target_sd_scale
                    for j in range(0, q.sizes["latitude"], 121):
                        a, b, c = [ds[variable].isel(latitude=slice(j, j+121)).values for ds in (l, h, q)]
                        finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
                        if finite.any():
                            maximum = max(maximum, float(np.max(np.abs(c[finite]-a[finite]-b[finite]))))
                            scale = max(scale, float(np.max(np.abs(c[finite]))))
                    if maximum > 1e-10*scale:
                        raise ValueError("Seasonal turbulent heat linearity failed")
                    linearity.append({"hemisphere": hemi, "retained_modes": k, "field": variable,
                                      "max_absolute_error": maximum, "tolerance": 1e-10*scale, "pass": True})
    write_rows(out/"reports/turbulent_linearity.csv", linearity)
    checks = {"regional_baseline_pass": True, "maximum_regional_baseline_error": max(errors),
              "seasonal_turbulent_linearity_pass": True,
              "baseline_map_max_errors": {h: {kind: item["baseline_max_errors"][h] for kind, item in metadata["processing"].items()} for h in HEMISPHERES}}
    write_json(out/"reports/acceptance.json", checks)
    return checks


def chart_figures(out, rows, series, kinds, dpi):
    for kind in kinds:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, hemi in zip(axes, HEMISPHERES):
            region = headline_region(hemi)
            for mi, model in enumerate(MODELS):
                for ei, evaluation in enumerate(("monthly_complete", "annual")):
                    r = select(rows, hemisphere=hemi, variable=kind, region=region, mask_policy="primary", model=model, evaluation=evaluation)
                    ax.plot(COUNTS, [select(r, retained_modes=k)[0]["total_variance_fraction_explained"]*100 for k in COUNTS],
                            marker="o", linestyle="-" if ei == 0 else "--", color=f"C{mi}", label=f"{model.replace('_', ' ')} / {evaluation.replace('_', ' ')}")
            ax.set(title=f"{hemi.upper()} PCs → {region.replace('_', ' ')}", xlabel="Number of PCs", xticks=COUNTS, ylabel="Total field variance explained (%)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.suptitle(f"{kind.replace('_', ' ')} | common years 1979–2025")
        fig.tight_layout()
        save(fig, out/"annual_seasonal_comparison"/f"{kind}_mode_curves.png", dpi)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
        for col, hemi in enumerate(HEMISPHERES):
            region = headline_region(hemi)
            unit = "W m⁻²" if kind in ("latent", "sensible", "turbulent") else "N m⁻²"
            for model in MODELS:
                r = select(series, hemisphere=hemi, variable=kind, model=model, region=region, mask_policy="primary", retained_modes=12, evaluation="annual")
                years = [v["date"] for v in r]
                if model == "year_round":
                    axes[0, col].plot(years, [v["observed"] for v in r], color="black", label="ERA5")
                axes[0, col].plot(years, [v["reconstructed"] for v in r], label=model.replace("_", " "))
                axes[1, col].plot(years, [v["reconstructed"]-v["observed"] for v in r], label=model.replace("_", " "))
            axes[0, col].set_title(f"{hemi.upper()} PCs | {region.replace('_', ' ')}")
            axes[0, col].set_ylabel(f"Annual anomaly ({unit})")
            axes[0, col].legend(fontsize=8)
            axes[1, col].axhline(0, color="gray", linewidth=.7)
            axes[1, col].set(xlabel="Year", ylabel=f"Reconstruction − observed ({unit})")
        fig.suptitle(f"{kind} | annual regional means, 12 PCs | spatial cancellation differs from field-variance skill")
        fig.tight_layout()
        save(fig, out/"annual_variability"/f"{kind}_regional_timeseries.png", dpi)
    for hemi in HEMISPHERES:
        region = headline_region(hemi)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(kinds))
        for j, (model, evaluation) in enumerate(( (m, e) for m in MODELS for e in ("monthly_complete", "annual") )):
            values = [select(rows, hemisphere=hemi, variable=k, model=model, evaluation=evaluation, retained_modes=12, region=region, mask_policy="primary")[0] for k in kinds]
            label = f"{model.replace('_', ' ').capitalize()} / {'monthly' if evaluation == 'monthly_complete' else 'annual'}"
            axes[0].bar(x+(j-1.5)*.19, [r["total_variance_fraction_explained"]*100 for r in values], width=.19, label=label)
            axes[1].bar(x+(j-1.5)*.19, [r["sd_ratio"] for r in values], width=.19)
        for ax in axes:
            ax.set_xticks(x, [k.replace("_", "\n") for k in kinds])
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Total field variance explained (%)")
        axes[1].set_ylabel("Reconstructed / observed SD (field variance)")
        axes[1].axhline(1, color="gray", linewidth=.7)
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{hemi.upper()} | {region.replace('_', ' ')} | 1979–2025, 12 PCs")
        fig.tight_layout()
        save(fig, out/"annual_seasonal_comparison"/f"{hemi}_monthly_annual_bars.png", dpi)
        fig, ax = plt.subplots(figsize=(11, 5))
        labels = ("monthly_full", *SEASONS)
        for j, kind in enumerate(kinds):
            diffs = []
            for evaluation in labels:
                a, b = [select(rows, hemisphere=hemi, variable=kind, model=m, evaluation=evaluation, retained_modes=12, region=region, mask_policy="primary")[0] for m in MODELS]
                diffs.append(100*(b["total_variance_fraction_explained"]-a["total_variance_fraction_explained"]))
            ax.bar(np.arange(5)+(j-2)*.15, diffs, .15, label=kind.replace("_", " "))
        ax.set(xticks=np.arange(5), xticklabels=labels, ylabel="Seasonal − year-round R² (percentage points)", title=f"{hemi.upper()} | 12 PCs | {region.replace('_', ' ')}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, out/"seasonal_regressions"/f"{hemi}_seasonal_gains.png", dpi)


def map_figures(out, baseline, kinds, dpi, coastlines, progress):
    import cartopy.crs as ccrs
    projection = ccrs.Robinson()
    for hemi, directory in HEMISPHERES.items():
        for kind in kinds:
            progress(f"Maps: {hemi} {kind}")
            with xr.open_dataset(out/"annual_variability"/hemi/f"{kind}_diagnostics.nc") as annual, xr.open_dataset(out/"seasonal_regressions"/hemi/f"{kind}_diagnostics.nc") as monthly:
                lat, lon = annual.latitude.values, annual.longitude.values
                maps = [monthly.r_squared.isel(model=mi, evaluation=1, retained_modes=3).values for mi in range(2)]
                maps += [annual.r_squared.isel(model=mi, retained_modes=3).values for mi in range(2)]
                limit = max(1., max(float(np.nanmax(np.abs(a))) for a in maps))
                fig, axes = plt.subplots(2, 2, figsize=(13, 7), subplot_kw={"projection": projection})
                for ax, a, title in zip(axes.flat, maps, ("Year-round / monthly", "Seasonal / monthly", "Year-round / annual", "Seasonal / annual")):
                    im = map_panel(ax, a, lat, lon, title, limit, coastlines=coastlines)
                fig.suptitle(f"{hemi.upper()} {kind} | prediction R², 12 PCs, 1979–2025 | negative values retained")
                fig.subplots_adjust(right=.87, top=.91, bottom=.03)
                fig.colorbar(im, ax=list(axes.flat), shrink=.65, label="Prediction R²")
                save(fig, out/"annual_seasonal_comparison"/f"{hemi}_{kind}_r_squared.png", dpi)
                delta = [100*(monthly.r_squared.isel(model=1, evaluation=ei, retained_modes=3).values-monthly.r_squared.isel(model=0, evaluation=ei, retained_modes=3).values) for ei in (0, 2, 3, 4, 5)]
                limit = max(max(float(np.nanmax(np.abs(a))) for a in delta), 1e-12)
                fig, axes = plt.subplots(2, 3, figsize=(15, 5.6), subplot_kw={"projection": projection})
                for ax, a, title in zip(axes.flat, delta, ("All months", *SEASONS)):
                    im = map_panel(ax, a, lat, lon, title, limit, coastlines=coastlines)
                axes.flat[-1].set_visible(False)
                fig.suptitle(f"{hemi.upper()} {kind} | seasonal − year-round R², percentage points | 12 PCs")
                fig.subplots_adjust(right=.88, top=.92, bottom=.03)
                fig.colorbar(im, ax=list(axes.flat)[:-1], shrink=.65, label="Percentage points")
                save(fig, out/"seasonal_regressions"/f"{hemi}_{kind}_delta_r_squared.png", dpi)
            with xr.open_dataset(out/"seasonal_regressions"/hemi/f"{kind}_coefficients.nc") as seasonal, xr.open_dataset(baseline/directory/"joint"/f"{kind}.nc") as fixed:
                for mode in ((1, 2) if hemi == "north" else (1, 11)):
                    year = fixed.coefficient.sel(mode=mode).values
                    season = seasonal.coefficient_12.isel(pc_12=mode-1).values
                    limit = max(float(np.nanmax(np.abs(year))), float(np.nanmax(np.abs(season))), 1e-12)
                    diff = season-year
                    dlimit = max(float(np.nanmax(np.abs(diff))), 1e-12)
                    fig, axes = plt.subplots(2, 5, figsize=(19, 4.8), subplot_kw={"projection": projection})
                    direct = (year, *season)
                    for ax, a, title in zip(axes[0], direct, ("Year-round", *SEASONS)):
                        im = map_panel(ax, a, lat, lon, title, limit, coastlines=coastlines)
                    axes[1, 0].set_visible(False)
                    for ax, a, title in zip(axes[1, 1:], diff, SEASONS):
                        dim = map_panel(ax, a, lat, lon, f"{title} − year-round", dlimit, coastlines=coastlines)
                    fig.suptitle(f"{hemi.upper()} PC{mode} → {kind} | joint 12-PC coefficients per original PC SD")
                    fig.subplots_adjust(top=.86, bottom=.10, hspace=.22)
                    unit = "W m⁻² / PC SD" if kind in ("latent", "sensible", "turbulent") else "N m⁻² / PC SD"
                    fig.colorbar(im, ax=list(axes[0]), shrink=.8, label=unit)
                    fig.colorbar(dim, ax=list(axes[1, 1:]), shrink=.8, label="Δ "+unit)
                    save(fig, out/"seasonal_regressions"/f"{hemi}_{kind}_pc{mode}_patterns.png", dpi)
        if {"eastward_stress", "northward_stress"}.issubset(kinds):
            vector_figures(out, baseline, hemi, directory, dpi, coastlines)


def vector_figures(out, baseline, hemi, directory, dpi, coastlines):
    import cartopy.crs as ccrs
    with xr.open_dataset(out/"seasonal_regressions"/hemi/"eastward_stress_coefficients.nc") as east, xr.open_dataset(out/"seasonal_regressions"/hemi/"northward_stress_coefficients.nc") as north, xr.open_dataset(baseline/directory/"joint/eastward_stress.nc") as ey, xr.open_dataset(baseline/directory/"joint/northward_stress.nc") as ny:
        lat, lon = east.latitude.values, east.longitude.values
        for mode in ((1, 2) if hemi == "north" else (1, 11)):
            u = np.concatenate((ey.coefficient.sel(mode=mode).values[None], east.coefficient_12.isel(pc_12=mode-1).values))
            v = np.concatenate((ny.coefficient.sel(mode=mode).values[None], north.coefficient_12.isel(pc_12=mode-1).values))
            mag = np.hypot(u, v)
            limit, reference = max(float(np.nanmax(mag)), 1e-12), max(float(np.nanpercentile(mag, 95)), 1e-12)
            fig, axes = plt.subplots(2, 3, figsize=(15, 6), subplot_kw={"projection": ccrs.Robinson()})
            for si, (ax, title) in enumerate(zip(axes.flat, ("Year-round", *SEASONS))):
                im = map_panel(ax, mag[si], lat, lon, title, limit, True, coastlines)
                xx, yy, uu, vv = quiver_samples(lat, lon, u[si], v[si], max(1, round(5/abs(lat[1]-lat[0]))), max(1, round(7.5/abs(lon[1]-lon[0]))))
                q = ax.quiver(xx, yy, uu, vv, transform=ccrs.PlateCarree(), scale=reference*20,
                              width=.002, color="white", edgecolors="black", linewidths=.15)
                ax.quiverkey(q, .8, -.08, reference, f"{reference:.3g} N m⁻² / PC SD", labelpos="S", fontproperties={"size": 7})
            axes.flat[-1].set_visible(False)
            fig.suptitle(f"{hemi.upper()} PC{mode} | stress magnitude and direction | shared vector scale; display-only thinning")
            fig.subplots_adjust(top=.9, bottom=.08, hspace=.3)
            fig.colorbar(im, ax=list(axes.flat)[:-1], shrink=.6, label="N m⁻² per original PC SD")
            save(fig, out/"seasonal_regressions"/f"{hemi}_pc{mode}_stress_vectors.png", dpi)


def pc_figures(out, metadata, dpi):
    for hemi in HEMISPHERES:
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, season in zip(axes, SEASONS):
            d = metadata["seasonal_pc_diagnostics"][hemi][season]
            im = ax.imshow(d["correlation"], vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set(title=f"{season}, N={d['samples']}\nmax |off-diagonal r|={d['maximum_offdiagonal_correlation']:.3f}", xticks=[0, 5, 11], xticklabels=[1, 6, 12], yticks=[0, 5, 11], yticklabels=[1, 6, 12], xlabel="PC", ylabel="PC")
        fig.suptitle(f"{hemi.upper()} original PCs within seasons | no restandardization or rotation")
        fig.subplots_adjust(right=.89, top=.8, bottom=.15)
        fig.colorbar(im, ax=list(axes), label="Correlation", shrink=.7)
        save(fig, out/"seasonal_regressions"/f"{hemi}_pc_correlations.png", dpi)


def report(out, rows, patterns, metadata, checks):
    lines = ["# Annual variability and seasonal regression experiments", "",
             "## Inputs and method", "",
             f"Original undetrended, independently fitted NH/SH 12-PC bases; {metadata['first_month']}–{metadata['last_month']}, {metadata['n_samples']} months. Annual evaluation uses {metadata['annual_samples']} complete calendar years ({metadata['annual_years'][0]}–{metadata['annual_years'][-1]}); matched monthly evaluation uses {metadata['complete_year_months']} months. Seasonal coefficients use the complete monthly sample.", "",
             "Native 0.25° global targets; fixed ERA5 lsm <0.5 primary mask and lsm ≤0.05 sensitivity. Exact spherical cell areas are clipped at ±20°. Full-record finite cells and original monthly climatologies are unchanged for every comparison. Equal monthly weights are used for annual means. Heat is positive downward; stresses positive eastward/northward. Sea-ice locations remain included and do not necessarily describe direct open-water ocean heating.", "",
             "Year-round coefficients/intercept are reused without refitting, including for 1/3/6-PC truncations. Seasonal OLS fits each nested predictor set independently using cached QR factors, original PC scaling and one intercept per season. No annual fit, EOF fit, detrending, reorthogonalization, regularization or filtering is performed.", "",
             "Prediction R² = 1 − SSE/SST, using each evaluation sample's observed mean only in SST. Predictions are never recentered. Regional field R² pools area-weighted SSE/SST; it differs from area-mean local R² and regional-mean-series R². Negative R² is retained. Squared correlation and reconstructed/observed variance ratio are not prediction R². Residual variance is centred about its own mean; RMSE includes bias. Sample variances use N−1 and are descriptive, not unbiased error estimates after climatology fitting and serial dependence.", "",
             "## Primary extratropical results", "",
             "All percentages below are total area-weighted field variance explained, primary mask, 12 PCs. Monthly columns use 1979–2025. Δ is seasonal minus year-round in percentage points.", "",
             "| PCs / region | Variable | Monthly year-round % | Annual year-round % | Monthly seasonal % | Annual seasonal % | Monthly Δ pp | Annual Δ pp |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    summary = []
    for hemi in HEMISPHERES:
        region = headline_region(hemi)
        for kind in metadata["variables"]:
            get = lambda model, ev, k=12: select(rows, hemisphere=hemi, variable=kind, model=model, evaluation=ev, retained_modes=k, region=region, mask_policy="primary")[0]
            ym, ya, sm, sa = [get(m, e) for m, e in (("year_round", "monthly_complete"), ("year_round", "annual"), ("seasonal", "monthly_complete"), ("seasonal", "annual"))]
            values = [100*r["total_variance_fraction_explained"] for r in (ym, ya, sm, sa)]
            lines.append(f"| {hemi.upper()} / own extratropics | {kind} | "+" | ".join(f"{v:.3f}" for v in (*values, values[2]-values[0], values[3]-values[1]))+" |")
            summary.append({"hemisphere": hemi, "variable": kind, "monthly_year_round_percent": values[0], "annual_year_round_percent": values[1], "monthly_seasonal_percent": values[2], "annual_seasonal_percent": values[3]})
    lines += ["", "## Annual timing, amplitude and higher modes", "",
              "Field SD ratios compare area-weighted temporal variances; correlations below describe annual regional-mean series, where spatial cancellation can change the apparent agreement. The higher-mode gain retains fixed year-round coefficients.", "",
              "| Hemisphere | Variable | Year-round annual SD ratio | Seasonal annual SD ratio | Regional-mean annual r, year-round | Regional-mean annual r, seasonal | Annual year-round gain PCs 7–12, pp |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for kind in metadata["variables"]:
            a = select(rows, hemisphere=hemi, variable=kind, model="year_round", evaluation="annual", retained_modes=12, region=headline_region(hemi), mask_policy="primary")[0]
            b = select(rows, hemisphere=hemi, variable=kind, model="seasonal", evaluation="annual", retained_modes=12, region=headline_region(hemi), mask_policy="primary")[0]
            c = select(rows, hemisphere=hemi, variable=kind, model="year_round", evaluation="annual", retained_modes=6, region=headline_region(hemi), mask_policy="primary")[0]
            lines.append(f"| {hemi} | {kind} | {a['sd_ratio']:.3f} | {b['sd_ratio']:.3f} | {a['regional_mean_correlation']:.3f} | {b['regional_mean_correlation']:.3f} | {100*(a['total_variance_fraction_explained']-c['total_variance_fraction_explained']):+.3f} |")
    lines += ["", "## Seasonal gains over the original full monthly sample", "",
              "| Hemisphere | Variable | All months Δ pp | DJF Δ pp | MAM Δ pp | JJA Δ pp | SON Δ pp |", "|---|---|---:|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for kind in metadata["variables"]:
            changes = []
            for ev in ("monthly_full", *SEASONS):
                a, b = [select(rows, hemisphere=hemi, variable=kind, model=m, evaluation=ev, retained_modes=12, region=headline_region(hemi), mask_policy="primary")[0] for m in MODELS]
                changes.append(100*(b["total_variance_fraction_explained"]-a["total_variance_fraction_explained"]))
            lines.append(f"| {hemi} | {kind} | "+" | ".join(f"{v:+.3f}" for v in changes)+" |")
    lines += ["", "Within-season year-round R² can be negative because its coefficients were optimized over all months. A large seasonal gain can therefore correct poor seasonal calibration without implying an equally large seasonal R². The paired values below make that distinction explicit; each pair uses identical months and the same SST denominator.", "",
              "| Hemisphere | Variable | "+" | ".join(f"{s} year-round → seasonal % (N={metadata['seasonal_pc_diagnostics']['north'][s]['samples']})" for s in SEASONS)+" |",
              "|---|---|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for kind in metadata["variables"]:
            pairs = []
            for season in SEASONS:
                a, b = [select(rows, hemisphere=hemi, variable=kind, model=m, evaluation=season, retained_modes=12, region=headline_region(hemi), mask_policy="primary")[0] for m in MODELS]
                pairs.append(f"{100*a['total_variance_fraction_explained']:.3f} → {100*b['total_variance_fraction_explained']:.3f}")
            lines.append(f"| {hemi} | {kind} | "+" | ".join(pairs)+" |")
    lines += ["", "## Seasonal predictor diagnostics", "", "| Hemisphere | Season | N | Rank | Condition (centred X) | Condition ([1,X]) | Max off-diagonal correlation |", "|---|---|---:|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for season in SEASONS:
            d = metadata["seasonal_pc_diagnostics"][hemi][season]
            lines.append(f"| {hemi} | {season} | {d['samples']} | {d['rank']} | {d['condition_number']:.4f} | {d['augmented_condition_number']:.4f} | {d['maximum_offdiagonal_correlation']:.4f} |")
    lines += ["", "Seasonal predictor means/SDs, full matrices and nested conditioning are in [seasonal_pc_diagnostics.json](seasonal_pc_diagnostics.json) and [PC diagnostics NetCDF](../seasonal_regressions/pc_diagnostics.nc). Seasonal correlations invalidate individual/joint coefficient equivalence; joint fits are used throughout.", "",
              "## Selected regression patterns", "",
              "PC1 uses the saved positive AO/SAM convention: polar-cap minus midlatitude pressure loading is negative. No named identity is assigned to higher modes. Pattern statistics below are global-ocean area-weighted; RMS measures coefficient amplitude per original PC SD, not realized seasonal forcing amplitude. DJF is NH winter and SH summer; JJA is SH winter and NH summer.", "",
              "| Hemisphere | PC | Variable | Year-round RMS | DJF RMS | MAM RMS | JJA RMS | SON RMS | Minimum seasonal pair pattern r |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for mode in ((1, 2) if hemi == "north" else (1, 11)):
            for kind in metadata["variables"]:
                pp = select(patterns, hemisphere=hemi, mode=mode, variable=kind, region="global_ocean", mask_policy="primary")
                rms = [select(pp, pattern=s, comparison_pattern=s)[0]["rms_coefficient"] for s in ("year_round", *SEASONS)]
                pair = [r["weighted_pattern_correlation"] for r in pp if r["pattern"] in SEASONS and r["comparison_pattern"] in SEASONS and r["pattern"] != r["comparison_pattern"]]
                lines.append(f"| {hemi} | {mode} | {kind} | "+" | ".join(f"{v:.4g}" for v in (*rms, min(pair)))+" |")
    lines += ["", "## Where annual averaging changes agreement", "",
              "Each entry is annual minus matched-monthly year-round field R² in percentage points (12 PCs). This tests whether gains are localized to the predictor hemisphere rather than inferring localization from its headline region alone.", "",
              "| Hemisphere | Variable | Global ocean | Northern extratropics | Southern extratropics | Tropics |",
              "|---|---|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        for kind in metadata["variables"]:
            changes = []
            for region in ("global_ocean", "northern_extratropics", "southern_extratropics", "tropics"):
                monthly, annual = [select(rows, hemisphere=hemi, variable=kind, model="year_round", evaluation=ev, retained_modes=12, region=region, mask_policy="primary")[0] for ev in ("monthly_complete", "annual")]
                changes.append(100*(annual["total_variance_fraction_explained"]-monthly["total_variance_fraction_explained"]))
            lines.append(f"| {hemi} | {kind} | "+" | ".join(f"{v:+.3f}" for v in changes)+" |")
    lines += ["", "## PC1 winter/summer coefficient amplitudes", "",
              "RMS is evaluated over each predictor's own extratropical ocean region, per original PC SD. It measures the regression response, not seasonal PC variance or causal forcing. Heat and stress have different physical units and their absolute RMS values should not be compared directly.", "",
              "| Hemisphere | Variable | Winter RMS | Summer RMS | Winter / summer RMS | Winter pattern r with year-round | Summer pattern r with year-round |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for hemi in HEMISPHERES:
        winter, summer = ("DJF", "JJA") if hemi == "north" else ("JJA", "DJF")
        for kind in metadata["variables"]:
            pp = select(patterns, hemisphere=hemi, variable=kind, mode=1, region=headline_region(hemi), mask_policy="primary")
            if not pp:
                continue
            w, s = [select(pp, pattern=season, comparison_pattern="year_round")[0] for season in (winter, summer)]
            ratio = w["rms_coefficient"]/s["rms_coefficient"] if s["rms_coefficient"] else np.nan
            lines.append(f"| {hemi} | {kind} | {w['rms_coefficient']:.4g} | {s['rms_coefficient']:.4g} | {ratio:.3f} | {w['weighted_pattern_correlation']:.3f} | {s['weighted_pattern_correlation']:.3f} |")
    lines += ["", "## Calculated interpretation", ""]
    for hemi in HEMISPHERES:
        hh = [r for r in summary if r["hemisphere"] == hemi]
        increased = [r["variable"] for r in hh if r["annual_year_round_percent"] > r["monthly_year_round_percent"]]
        decreased = [r["variable"] for r in hh if r["annual_year_round_percent"] <= r["monthly_year_round_percent"]]
        lines.append(f"{hemi.upper()} own extratropics: annual averaging increases fixed-model field R² for {', '.join(increased) or 'none of the targets'}; decreases or leaves it unchanged for {', '.join(decreased) or 'none'}. These comparisons use identical years and original coefficients. The annual amplitude/timing table identifies whether agreement reflects timing, amplitude, or both.")
        lines.append("")
        improved = [r["variable"] for r in hh if r["annual_seasonal_percent"] > r["annual_year_round_percent"]]
        lines.append(f"Seasonal coefficients improve annual field R² for {', '.join(improved) or 'none of the targets'} in the {hemi.upper()} own extratropics. Full-record monthly gains should be read with the nested-model parameter-count qualification below. Seasonal gains, mode-count curves and the regional table retain cases that differ from this headline result.")
        lines.append("")
        for season in SEASONS:
            gains = []
            for kind in metadata["variables"]:
                a, b = [select(rows, hemisphere=hemi, variable=kind, model=m, evaluation=season, retained_modes=12, region=headline_region(hemi), mask_policy="primary")[0] for m in MODELS]
                gains.append((100*(b["total_variance_fraction_explained"]-a["total_variance_fraction_explained"]), kind))
            best = max(gains)
            lines.append(f"- {hemi.upper()} {season}: largest own-extratropical gain is {best[1]} ({best[0]:+.3f} percentage points).")
        lines.append("")
    lines += ["## Higher modes in the seasonal model", "",
              "These gains compare independently fitted seasonal 12-PC and 6-PC models. Monthly fitting-sample improvement is expected for nested OLS; annual improvement is not guaranteed.", "",
              "| Hemisphere | Variable | Full monthly gain PCs 7–12, pp | Annual gain PCs 7–12, pp |",
              "|---|---|---:|---:|"]
    for hemi in HEMISPHERES:
        for kind in metadata["variables"]:
            gains = []
            for evaluation in ("monthly_full", "annual"):
                a, b = [select(rows, hemisphere=hemi, variable=kind, model="seasonal", evaluation=evaluation, retained_modes=k, region=headline_region(hemi), mask_policy="primary")[0] for k in (6, 12)]
                gains.append(100*(b["total_variance_fraction_explained"]-a["total_variance_fraction_explained"]))
            lines.append(f"| {hemi} | {kind} | {gains[0]:+.3f} | {gains[1]:+.3f} |")
    lines += ["", "## Heat-flux versus stress pattern seasonality", "",
              "The mean of the six distinct seasonal-pair spatial correlations measures pattern resemblance; a lower value indicates stronger shape changes. This is dimensionless, unlike coefficient RMS, and does not capture a pure amplitude change. Latent and sensible are listed separately from their derived sum. These comparisons describe the selected joint coefficients, not all retained modes.", "",
              "| Hemisphere | PC | Region | Latent mean pair r | Sensible mean pair r | Turbulent mean pair r | Eastward stress mean pair r | Northward stress mean pair r |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    season_order = list(SEASONS)
    for hemi in HEMISPHERES:
        for mode in ((1, 2) if hemi == "north" else (1, 11)):
            for region in ("global_ocean", headline_region(hemi)):
                means = []
                for kind in ("latent", "sensible", "turbulent", "eastward_stress", "northward_stress"):
                    pp = select(patterns, hemisphere=hemi, mode=mode, variable=kind, region=region, mask_policy="primary")
                    pair = [r["weighted_pattern_correlation"] for r in pp if r["pattern"] in SEASONS and r["comparison_pattern"] in SEASONS and season_order.index(r["pattern"]) < season_order.index(r["comparison_pattern"])]
                    means.append(float(np.mean(pair)) if pair else np.nan)
                lines.append(f"| {hemi} | {mode} | {region} | "+" | ".join(f"{v:.3f}" for v in means)+" |")
    lines += ["", "## Numerical acceptance and resources", "",
              f"Regional baseline checks pass: {checks['regional_baseline_pass']}; maximum absolute regional diagnostic difference: {checks['maximum_regional_baseline_error']:.3g}. Seasonal latent+sensible coefficient and intercept linearity passes: {checks['seasonal_turbulent_linearity_pass']}. Per-map checks, thresholds and errors are in [acceptance.json](acceptance.json). Turbulent R² uses the summed observed field, including component covariance; component R² values are never added.", "",
              f"Numerical runtime: {metadata['analysis_elapsed_seconds']:.1f} s; peak process RSS: {metadata['peak_process_rss_bytes']/2**30:.2f} GiB; numerical NetCDF storage: {metadata['numerical_output_bytes']/2**30:.2f} GiB. These exclude subsequent figure rendering. Tiles are {metadata['tile_shape']}; no full monthly reconstruction cubes are saved.", "",
              "## Interpretation and limitations", "",
              "Annual averaging changes the covariance between reconstructed and unresolved components. Improved annual agreement is consistent with unresolved monthly fluctuations averaging out, but does not establish that mechanism; deterioration indicates slower variability or amplitude/bias errors remain unresolved. Annual PC orthogonality is not assumed, and individual annual explained fractions are not added. The 47-year sample gives limited information about decadal/interdecadal behavior and does not establish long-term predictability.", "",
              "Seasonal models have K+1 coefficients per season versus K+1 year-round: 8/16/28/52 total seasonal coefficients for K=1/3/6/12, compared with 2/4/7/13 year-round. Seasonal/full-sample in-sample SSE cannot increase because the year-round model is nested; an R² gain alone does not demonstrate predictive skill. Seasonal and full-sample R² use their own observed variance denominator; full-sample R² is pooled directly, never averaged from four seasonal values. The matched-year subset and annual evaluation have no guaranteed improvement.", "",
              "NH and SH reconstructions overlap and must not be added as a combined model. Global, tropical, cross-hemisphere and pure-ocean results are preserved in the tables. Turbulent heat flux includes SST feedback; radiation is excluded. Neither the reconstruction nor its residual is complete stochastic forcing F′. No Ekman transport, ocean-model integration, five/ten-year averaging or full held-out validation is included. Future predictive validation must estimate EOFs, climatologies and regression coefficients using training observations only.", "",
              "## Output navigation", "",
              "[All regional metrics](regional_metrics.csv) · [Model differences](model_differences.csv) · [Regional series](regional_timeseries.csv) · [Pattern statistics](../seasonal_regressions/pattern_diagnostics.csv) · [Annual outputs](../annual_variability/) · [Seasonal outputs](../seasonal_regressions/) · [Four-way comparison](../annual_seasonal_comparison/) · [Metadata](metadata.json)", ""]
    gallery = [
        ("NH monthly/annual comparison", "annual_seasonal_comparison/north_monthly_annual_bars.png"),
        ("SH monthly/annual comparison", "annual_seasonal_comparison/south_monthly_annual_bars.png"),
        ("Turbulent heat: retained modes", "annual_seasonal_comparison/turbulent_mode_curves.png"),
        ("Turbulent heat: annual regional means", "annual_variability/turbulent_regional_timeseries.png"),
        ("NH turbulent heat: monthly and annual R²", "annual_seasonal_comparison/north_turbulent_r_squared.png"),
        ("SH turbulent heat: monthly and annual R²", "annual_seasonal_comparison/south_turbulent_r_squared.png"),
    ]
    available = [(label, path) for label, path in gallery if (out/path).is_file()]
    if available:
        lines += ["## Selected figures", ""]
        for label, path in available:
            lines += [f"![{label}](../{path})", ""]
    (out/"reports/sensitivity_report.md").write_text("\n".join(lines))
    write_json(out/"reports/summary.json", {"headline": summary, "acceptance": checks})


def render(out, baseline, dpi=300, coastlines=True, figures=True, progress=print):
    metadata = json.loads((out/"reports/metadata.json").read_text())
    if metadata.get("pilot") or not metadata.get("completed"):
        raise ValueError("Cannot render pilot/incomplete outputs as a global analysis")
    rows = read_rows(out/"reports/regional_metrics.csv")
    patterns = read_rows(out/"seasonal_regressions/pattern_diagnostics.csv")
    checks = check_outputs(out, baseline, rows, metadata)
    if figures:
        series = read_rows(out/"reports/regional_timeseries.csv")
        chart_figures(out, rows, series, metadata["variables"], dpi)
        pc_figures(out, metadata, dpi)
        map_figures(out, baseline, metadata["variables"], dpi, coastlines, progress)
    metadata["rendering"] = {"dpi": dpi, "coastlines": coastlines,
                             "figures": len(list(out.rglob("*.png")))}
    write_json(out/"reports/metadata.json", metadata)
    report(out, rows, patterns, metadata, checks)
    return checks
