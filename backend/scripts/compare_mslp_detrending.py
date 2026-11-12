"""Compare saved baseline and detrended EOFs on an identical grid and time window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.mslp_eof import DETRENDING_METHOD, compare_periods
from app.mslp_eof_diagnostics import _map, _save, save_dataset, write_csv


def compare(baseline_root: Path, detrended_root: Path, output: Path, dpi: int) -> Path:
    import cartopy.crs as ccrs

    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    report = ["# Linear detrending sensitivity: MSLP EOFs", "",
              "Baseline and detrended analyses use identical months, native grids, latitude weights, "
              "and 12 retained modes. Detrending jointly removes calendar-month effects and a "
              "linear time term at each grid cell. Maps are hPa per PC sample standard deviation.", "",
              "Modes are matched by maximum total absolute area-weighted spatial Pearson correlation. "
              "The same sign alignment is applied to maps and PCs. PC correlations use unsmoothed "
              "monthly series; these compare separately fitted EOFs, not projections on one fixed basis.", ""]
    for hemisphere, folder in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        relative = Path(folder) / "1979_2026" / "eof.nc"
        baseline_path, detrended_path = baseline_root / relative, detrended_root / relative
        with xr.open_dataset(baseline_path, engine="h5netcdf") as source:
            baseline = source.load()
        with xr.open_dataset(detrended_path, engine="h5netcdf") as source:
            detrended = source.load()
        for coordinate in ("time", "latitude", "longitude", "mode"):
            if not np.array_equal(baseline[coordinate], detrended[coordinate]):
                raise ValueError(f"Mismatched {hemisphere} {coordinate}")
        if baseline.sizes["mode"] != 12 or baseline.attrs["detrended"] != 0:
            raise ValueError("Expected a 12-mode undetrended baseline")
        if detrended.attrs.get("detrending_method") != DETRENDING_METHOD:
            raise ValueError("Expected joint monthly/linear detrending")
        if baseline.attrs["source_files"] != detrended.attrs["source_files"]:
            raise ValueError("Source fingerprints differ")
        # Verify zero calendar-month means and zero linear trend in saved PCs.
        t = np.arange(detrended.sizes["time"], dtype=float)
        t -= t.mean()
        residual_slope = t @ detrended.standardized_pc.values / (t @ t)
        month_means = detrended.standardized_pc.groupby("time.month").mean("time").values
        if np.max(np.abs(residual_slope)) > 1e-10 or np.max(np.abs(month_means)) > 1e-9:
            raise ValueError("Detrended PCs failed residual checks")
        comparison = compare_periods(baseline, detrended)
        comparison.attrs.update({"baseline_path": str(baseline_path.resolve()),
                                 "detrended_path": str(detrended_path.resolve()),
                                 "comparison_type": "identical-period linear detrending sensitivity"})
        save_dataset(comparison, output / f"{hemisphere}_comparison.nc")
        rows = comparison.full_mode_number.values.astype(int) - 1
        cols = comparison.matched_modern_mode.values.astype(int) - 1
        signs = comparison.modern_sign.values
        base_eigen = baseline.eigenvalue.values[rows]
        det_eigen = detrended.eigenvalue.values[cols]
        table = {"baseline_mode": rows + 1, "detrended_mode": cols + 1,
                 "detrended_alignment_sign": signs,
                 "spatial_r": comparison.aligned_spatial_correlation.values,
                 "pc_r": comparison.common_pc_correlation.values,
                 "baseline_variance_percent": baseline.explained_variance_percent.values[rows],
                 "detrended_variance_percent": detrended.explained_variance_percent.values[cols],
                 "difference_percentage_points": comparison.variance_difference_percentage_points.values,
                 "baseline_eigenvalue_hpa2": base_eigen,
                 "detrended_eigenvalue_hpa2": det_eigen,
                 "eigenvalue_change_percent": 100 * (det_eigen / base_eigen - 1)}
        write_csv(output / f"{hemisphere}_matching.csv", table)
        np.savetxt(output / f"{hemisphere}_spatial_correlation_matrix.csv",
                   comparison.spatial_correlation_matrix.values, delimiter=",",
                   header="rows=baseline modes; columns=detrended modes")
        total_base = float(baseline.attrs["total_weighted_variance_hpa2"])
        total_det = float(detrended.attrs["total_weighted_variance_hpa2"])
        if total_det > total_base * (1 + 1e-10):
            raise ValueError("Detrending unexpectedly increased total variance")
        summary = {"first_month": baseline.attrs["first_month"],
                   "last_month": baseline.attrs["last_month"], "n_months": len(t),
                   "baseline_total_weighted_variance_hpa2": total_base,
                   "detrended_total_weighted_variance_hpa2": total_det,
                   "total_variance_removed_percent": 100 * (1 - total_det / total_base),
                   "baseline_retained_percent": float(baseline.cumulative_variance_percent.values[-1]),
                   "detrended_retained_percent": float(detrended.cumulative_variance_percent.values[-1]),
                   "principal_angles_degrees": comparison.principal_angle_degrees.values.tolist(),
                   "maximum_absolute_residual_pc_slope_per_month": float(np.max(np.abs(residual_slope))),
                   "maximum_absolute_residual_pc_monthly_mean": float(np.max(np.abs(month_means))),
                   "matches": [{name: np.asarray(values)[k].item() for name, values in table.items()}
                               for k in range(len(rows))]}
        summaries[hemisphere] = summary
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        mode = baseline.mode.values
        for ds, label in ((baseline, "Baseline"), (detrended, "Linear detrended")):
            axes[0].plot(mode, ds.explained_variance_percent, "o-", label=label)
            axes[1].plot(mode, ds.cumulative_variance_percent, "o-", label=label)
            axes[2].plot(mode, ds.eigenvalue, "o-", label=label)
        for ax, ylabel in zip(axes, ("Explained variance (%)", "Cumulative variance (%)", "Weighted eigenvalue (hPa²)")):
            ax.set(xlabel="Mode rank", ylabel=ylabel, xticks=mode)
            ax.grid(alpha=.2)
        axes[0].legend()
        fig.suptitle(f"{hemisphere.capitalize()} | {summary['first_month']}–{summary['last_month']} | "
                     f"total variance removed: {summary['total_variance_removed_percent']:.2f}%")
        fig.tight_layout()
        _save(fig, output / f"{hemisphere}_variance_comparison.png", dpi)
        fig, axes = plt.subplots(12, 1, figsize=(13, 24), sharex=True)
        for ax, i, j, sign, r in zip(axes, rows, cols, signs, table["pc_r"]):
            ax.plot(baseline.time, baseline.standardized_pc.values[:, i], lw=.6, label="Baseline")
            ax.plot(detrended.time, detrended.standardized_pc.values[:, j] * sign,
                    lw=.6, alpha=.75, label="Detrended, sign aligned")
            ax.set_ylabel(f"{i+1} ↔ {j+1}\nPC (SD)")
            ax.set_title(f"Monthly PC correlation = {r:.4f}", fontsize=8)
        axes[0].legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel("Calendar year")
        fig.suptitle(f"{hemisphere.capitalize()} | Baseline versus linear detrended PCs")
        fig.tight_layout()
        _save(fig, output / f"{hemisphere}_pc_comparison.png", dpi)
        projection = ccrs.NorthPolarStereo() if hemisphere == "north" else ccrs.SouthPolarStereo()
        for i, j, sign, r in zip(rows, cols, signs, table["spatial_r"]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": projection})
            fields = (baseline.regression_map.values[i], detrended.regression_map.values[j] * sign)
            limit = max(float(np.max(np.abs(field))) for field in fields)
            for ax, field, ds, k, label in zip(axes, fields, (baseline, detrended), (i, j), ("Baseline", "Linear detrended")):
                image = _map(ax, field, ds, hemisphere,
                             f"{label}: EOF {k+1} ({ds.explained_variance_percent.values[k]:.2f}%)",
                             limit, True)
            fig.suptitle(f"{hemisphere.capitalize()} | {summary['first_month']}–{summary['last_month']} | spatial r = {r:.4f}")
            fig.subplots_adjust(top=.78, bottom=.08, left=.04, right=.94, wspace=.2)
            fig.colorbar(image, ax=list(axes), shrink=.65, pad=.08, label="hPa per PC standard deviation")
            _save(fig, output / f"{hemisphere}_baseline_{i+1}_detrended_{j+1}.png", dpi)
        report += [f"## {hemisphere.capitalize()}", "",
                   f"{summary['first_month']}–{summary['last_month']}; {len(t)} monthly samples.", "",
                   f"Total weighted variance decreases from {total_base:.3f} to {total_det:.3f} hPa² "
                   f"({summary['total_variance_removed_percent']:.3f}% removed). The first 12 modes explain "
                   f"{summary['baseline_retained_percent']:.3f}% before and "
                   f"{summary['detrended_retained_percent']:.3f}% after detrending.", "",
                   f"Retained-subspace principal angles range from {min(summary['principal_angles_degrees']):.3f}° "
                   f"to {max(summary['principal_angles_degrees']):.3f}°.", "",
                   f"Matched map correlations span {min(table['spatial_r']):.4f}–{max(table['spatial_r']):.4f}; "
                   f"monthly PC correlations span {min(table['pc_r']):.4f}–{max(table['pc_r']):.4f}. "
                   f"{int(np.sum(rows != cols))} of 12 modes change rank in the optimal matching.", "",
                   f"Baseline EOF1 matches detrended EOF{cols[0]+1}: spatial r = {table['spatial_r'][0]:.4f}, "
                   f"PC r = {table['pc_r'][0]:.4f}. Its variance share changes from "
                   f"{table['baseline_variance_percent'][0]:.3f}% to {table['detrended_variance_percent'][0]:.3f}% "
                   f"({table['difference_percentage_points'][0]:+.3f} percentage points), while its "
                   f"absolute eigenvalue changes by {table['eigenvalue_change_percent'][0]:+.3f}%.", "",
                   f"[Variance comparison]({hemisphere}_variance_comparison.png) · "
                   f"[PC overlays]({hemisphere}_pc_comparison.png) · [Full numerical table]({hemisphere}_matching.csv)", "",
                   "| Baseline EOF | Detrended EOF | Spatial r | PC r | Baseline variance % | Detrended variance % | Δ percentage points | Eigenvalue change % |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for match in summary["matches"]:
            report.append(f"| [{match['baseline_mode']}]({hemisphere}_baseline_{match['baseline_mode']}_detrended_{match['detrended_mode']}.png) "
                          f"| {match['detrended_mode']} | {match['spatial_r']:.4f} | {match['pc_r']:.4f} "
                          f"| {match['baseline_variance_percent']:.3f} | {match['detrended_variance_percent']:.3f} "
                          f"| {match['difference_percentage_points']:+.3f} | {match['eigenvalue_change_percent']:+.3f} |")
        report += [""]
    report += ["## Interpretation limits", "",
               "Variance percentages have different denominators: use eigenvalues and total variance "
               "to assess absolute changes. Matching allows mode reordering and sign flips. Small "
               "subspace angles with weaker individual matches can reflect rotations among similar "
               "modes. Detrending removes both forced and internal linear changes; this comparison "
               "does not identify their causes or establish statistical significance.", "",
               "Saved detrended PCs were checked for zero linear slopes and zero calendar-month means. "
               "Source fingerprints, time coordinates, spatial coordinates, mode counts and trend "
               "settings were checked before comparison."]
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    path = output / "detrending_comparison.md"
    path.write_text("\n".join(report) + "\n")
    print(json.dumps(summaries, indent=2, allow_nan=False), flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("diagnostics/mslp_eof_12"))
    parser.add_argument("--detrended-dir", type=Path, default=Path("diagnostics/mslp_eof_12_detrended"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/mslp_eof_12_detrended/comparisons"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    compare(args.baseline_dir, args.detrended_dir, args.output_dir, args.dpi)
