"""Native-grid ERA5 MSLP EOF analysis, independent of the ocean/globe pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys

import numpy as np
import xarray as xr

# Support the documented direct invocation without requiring PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mslp_eof import (compare_periods, month_strings, open_mslp, select_domain,
                           solve_eof, validate_grid, validate_months)
from app.mslp_eof_diagnostics import (export_analysis, export_comparison,
                                      interpretation_scope, write_report)
from app.mslp_indices import compare_indices

LOG = logging.getLogger("mslp_eof")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="NetCDF paths or quoted glob patterns")
    parser.add_argument("--variable", default="msl")
    parser.add_argument("--start-years", nargs="+", type=int, default=[1940, 1979])
    parser.add_argument("--hemispheres", nargs="+", choices=["north", "south"], default=["north", "south"])
    parser.add_argument("--end-year", type=int, default=2026, help="Upper limit; final available month is detected")
    parser.add_argument("--n-modes", type=int, default=6)
    parser.add_argument("--block-latitudes", type=int, default=16)
    parser.add_argument("--max-lag", type=int, default=24)
    parser.add_argument("--detrend", action="store_true", help="Optional sensitivity only; primary default is off")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/mslp_eof"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-maps", action="store_true", help="Explicitly omit polar maps")
    parser.add_argument("--no-coastlines", action="store_true", help="Avoid Natural Earth downloads")
    parser.add_argument("--no-running-mean", action="store_true")
    parser.add_argument("--inspect-only", action="store_true", help="Print metadata without computing EOFs")
    parser.add_argument("--render-only", action="store_true", help="Regenerate diagnostics from compatible saved EOF NetCDFs")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse compatible saved EOFs and compute missing analyses")
    parser.add_argument("--index-data-dir", type=Path,
                        help="Offline AO/SAM tables and provenance.json; defaults to OUTPUT/reference_indices if present")
    parser.add_argument("--allow-other-resolution", action="store_true", help="Synthetic validation only; retains input grid")
    return parser.parse_args(argv)


def run(args) -> Path | None:
    field, metadata = open_mslp(args.input, args.variable, args.block_latitudes)
    try:
        LOG.info("Input inspection:\n%s", json.dumps(metadata, indent=2, default=str))
        if args.inspect_only:
            return None
        validate_grid(field, resolution=None if args.allow_other_resolution else 0.25)
        selected = field.sel(time=slice(None, f"{args.end_year}-12-31"))
        validate_months(selected.time)
        ending = month_strings(selected.time)[-1]
        if len(set(args.start_years)) != len(args.start_years):
            raise ValueError("Duplicate start years")
        # Validate all requested periods before producing any primary outputs.
        for start in args.start_years:
            period = selected.sel(time=slice(f"{start}-01-01", None))
            validate_months(period.time)
            if month_strings(period.time)[0] != f"{start}-01":
                raise ValueError(f"Missing January {start}; input starts {metadata['first_month']}. "
                                 "Supply the missing ERA5 record; periods are never silently shortened.")
        for hemisphere in args.hemispheres:
            domain = select_domain(selected, hemisphere)
            expected = (20, 90) if hemisphere == "north" else (-90, -20)
            if (float(domain.latitude.min()), float(domain.latitude.max())) != expected:
                raise ValueError(f"Input must include the complete {hemisphere} domain {expected}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata.update({"settings": vars(args), "analysis_last_month": ending,
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(),
                         "versions": {name: importlib.metadata.version(name) for name in
                                      ["numpy", "scipy", "xarray", "dask", "h5netcdf", "matplotlib", "cartopy"]},
                         "native_resolution_primary": not args.allow_other_resolution,
                         "command": sys.argv})
        reports = args.output_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        if (args.render_only or args.reuse_existing) and (reports / "metadata.json").exists():
            previous = json.loads((reports / "metadata.json").read_text())
            metadata["analysis_command"] = previous.get("analysis_command", previous["command"])
            metadata["analysis_created_utc"] = previous.get("analysis_created_utc", previous["created_utc"])
            metadata["analysis_settings"] = previous.get("analysis_settings", previous["settings"])
        metadata["operation"] = "render_only" if args.render_only else "analysis"
        metadata["analyses"] = []
        results, comparisons = {}, {}
        for hemisphere in args.hemispheres:
            domain = select_domain(selected, hemisphere)
            for year in sorted(args.start_years):
                LOG.info("Starting %s %d–%s", hemisphere, year, ending)
                folder = "northern_hemisphere" if hemisphere == "north" else "southern_hemisphere"
                directory = args.output_dir / folder / f"{year}_{ending[:4]}"
                period = domain.sel(time=slice(f"{year}-01-01", None))
                reuse = args.render_only or (args.reuse_existing and (directory / "eof.nc").exists())
                if reuse:
                    with xr.open_dataset(directory / "eof.nc", engine="h5netcdf") as stored:
                        ds = stored.load()
                    if not (ds.sizes["mode"] == args.n_modes and
                            ds.sizes["lag"] == min(args.max_lag+1, period.sizes["time"]) and
                            ds.attrs["detrended"] == int(args.detrend) and
                            ds.attrs.get("source_variable") == args.variable and
                            ds.attrs.get("source_files") == json.dumps(metadata["files"]) and
                            all(np.array_equal(ds[d].values, period[d].values)
                                for d in ("time", "latitude", "longitude"))):
                        raise ValueError(f"Saved EOF settings/input mismatch: {directory}; recompute first")
                    LOG.info("Rendering saved EOFs: %s", directory)
                else:
                    ds = solve_eof(period, args.n_modes, args.block_latitudes, args.detrend, args.max_lag)
                    ds.attrs["block_latitudes"] = args.block_latitudes
                    ds.attrs["analysis_command"] = json.dumps(sys.argv)
                ds.attrs.update({"hemisphere": hemisphere, "source_variable": args.variable,
                                 "source_files": json.dumps(metadata["files"]),
                                 "native_resolution_primary": int(not args.allow_other_resolution)})
                results[hemisphere, year] = ds
                metadata["analyses"].append({"hemisphere": hemisphere, "start_year": year,
                                             "last_month": ending, "reused": bool(reuse),
                                             "block_latitudes": ds.attrs.get("block_latitudes",
                                                 metadata.get("analysis_settings", {}).get("block_latitudes")),
                                             "analysis_command": ds.attrs.get("analysis_command",
                                                 metadata.get("analysis_command", sys.argv))})
                export_analysis(ds, directory, hemisphere, args.dpi, not args.no_maps,
                                not args.no_coastlines, not args.no_running_mean)
            if len(args.start_years) == 2:
                first, second = sorted(args.start_years)
                comparison = compare_periods(results[hemisphere, first], results[hemisphere, second])
                comparisons[hemisphere] = comparison
                export_comparison(results[hemisphere, first], results[hemisphere, second], comparison,
                                  args.output_dir / "comparisons", hemisphere, args.dpi,
                                  not args.no_maps, not args.no_coastlines)
        index_directory = args.index_data_dir or args.output_dir / "reference_indices"
        validation = None
        if args.index_data_dir is not None or index_directory.exists():
            LOG.info("Comparing PCs with cached indices: %s", index_directory)
            validation = compare_indices(results, index_directory)
        analyses = []
        for (hemisphere, year), ds in sorted(results.items()):
            analyses.append({"hemisphere": hemisphere, "basis_start": year,
                             "first_month": ds.attrs["first_month"], "last_month": ds.attrs["last_month"],
                             "n_months": ds.sizes["time"], "n_modes": ds.sizes["mode"],
                             "eof1_variance_percent": float(ds.explained_variance_percent.values[0]),
                             "retained_variance_percent": float(ds.cumulative_variance_percent.values[-1])})
        matches = {h: {"full_modes": ds.full_mode_number.values.tolist(),
                        "matched_modern_modes": ds.matched_modern_mode.values.tolist(),
                        "spatial_correlations": ds.aligned_spatial_correlation.values.tolist(),
                        "pc_correlations": ds.common_pc_correlation.values.tolist(),
                        "principal_angles_degrees": ds.principal_angle_degrees.values.tolist()}
                   for h, ds in comparisons.items()}
        summary = {"scope": interpretation_scope(results, metadata), "analyses": analyses,
                   "period_comparisons": matches, "index_validation": validation}
        (reports / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

        # Summary 1: primary EOF results and historical-period stability.
        table = "\n".join(
            f"| {r['hemisphere']} | {r['first_month']}–{r['last_month']} | {r['n_months']} | "
            f"{r['n_modes']} | {r['eof1_variance_percent']:.3f} | {r['retained_variance_percent']:.3f} |"
            for r in analyses)
        stability = []
        for hemisphere, match in matches.items():
            pairs = ", ".join(f"{a}↔{b}: {r:.3f}" for a, b, r in zip(
                match["full_modes"], match["matched_modern_modes"], match["spatial_correlations"]))
            angles = match["principal_angles_degrees"]
            stability.append(f"- **{hemisphere.capitalize()}** matched spatial correlations "
                             f"(earlier EOF ↔ later EOF): {pairs}. Retained-subspace angles span "
                             f"{min(angles):.2f}°–{max(angles):.2f}°.")
        eof_summary = f"""Native-grid covariance EOFs were calculated independently by hemisphere and period,
with period-specific monthly climatologies, sqrt(cos(latitude)) weighting and
the configured trend treatment. Weighted EOF vectors and physical hPa-per-PC-SD
regression maps are saved separately.

| Hemisphere | Period | Monthly samples | Retained modes | EOF1 variance (%) | Retained variance (%) |
|---|---|---:|---:|---:|---:|
{table}

{chr(10).join(stability) if stability else 'Historical-period comparisons were not requested.'}

Mode matching allows rank changes and arbitrary signs. Lower individual correlations
alongside small subspace angles can indicate rotations; large subspace angles
also indicate differences in the retained subspace. These descriptive comparisons
do not establish statistical significance or a physical cause. No F′ estimation
is included.
"""

        # Summary 2: established-index validation and physical sign conventions.
        if validation is None:
            index_summary = ("No reference-index comparison was calculated in this run. Supply "
                             "--index-data-dir, or place official tables and provenance.json in "
                             "the output directory's reference_indices folder. No earlier index "
                             "correlations are substituted for the current EOF basis.")
        else:
            rows = validation["results"]
            table = "\n".join(
                f"| {r['index']} | {r['basis_start']} | {r['start']}–{r['end']} | {r['n']} | "
                f"{r['pc1_multiplier']:+d} | {r['raw_pc1_r']:.3f} | {r['aligned_pc1_r']:.3f} | "
                f"{r['detrended_r']:.3f} |" for r in rows)
            maximum_change = max((abs(r["aligned_pc1_r"]-r["detrended_r"]) for r in rows), default=0)
            other_modes = [abs(value) for r in rows for value in r["all_mode_raw_r"][1:]]
            higher_modes = (f"The largest absolute reference correlation among retained PCs "
                            f"2 onward is {max(other_modes):.3f}." if other_modes else
                            "Only PC1 was retained; higher-mode reference correlations are unavailable.")
            early = "\n".join(
                f"- {r['index']}, {r['basis_start']} EOF basis, "
                f"{r['pre_1979']['start']}–{r['pre_1979']['end']}: "
                f"r = {r['pre_1979']['aligned_pc1_r']:.3f} "
                f"({r['pre_1979']['n']} months)." for r in rows if "pre_1979" in r)
            sources = "\n".join(
                f"- [{name} data]({source['url']}), retrieved {source['retrieved_utc']}. "
                f"SHA256: `{source['sha256']}`." for name, source in validation["provenance"].items())
            formula = (f"Reconstructing Marshall SAM as normalised P40 minus normalised P65, "
                       f"using monthly 1971–2000 means and population SDs, correlates "
                       f"{validation['sam_formula_correlation']:.6f} with the published index. "
                       "This directly checks the pressure-difference sign."
                       if "sam_formula_correlation" in validation else "Station-band pressure sign reconstruction was not available.")
            index_summary = f"""Comparison uses calendar-matched, **unsmoothed monthly** values. Each index
has a shared comparison window across the supplied EOF bases, starting no earlier
than January 1979 and ending at the last jointly available month. Baseline and
standardisation differences are retained; Pearson correlation does not require
equal amplitude scales. Missing or repeated reference months cause an error.

Conventional positive AO/SAM corresponds to lower polar pressure relative to
midlatitudes. The alignment multiplier is chosen from each PC1 regression map's
65–90° cap minus 35–55° band contrast, **not** by maximising correlation. A −1
multiplier means the saved PC sign is opposite to the conventional index.
The original NetCDF signs are preserved. Convention-aligned exported products
would need the PC, weighted EOF and regression map flipped together.

| Reference | EOF basis start | Comparison months | N | PC1 multiplier | Saved PC1 r | Aligned PC1 r | Detrended paired-series r |
|---|---:|---|---:|---:|---:|---:|---:|
{table}

The largest absolute change after separately removing linear trends from the
paired time series is {maximum_change:.6f}. This sensitivity does not refit the
EOFs or change primary results. {higher_modes}

Available pre-1979 checks:

{early or 'No pre-1979 index overlap is available for these bases.'}

{formula}

CPC AO uses 1000-hPa height and CPC AAO uses 700-hPa height, with EOF loading
patterns and index scaling based on 1979–2000. Marshall SAM uses station-based
normalised pressure differences near 40°S and 65°S, with 1971–2000 normalisation.
These differ from the present hemispheric MSLP EOFs, so exact equality is not
expected. High correlations support mode identification, not causal attribution
or independent out-of-sample validation. No correlation significance test is
claimed for autocorrelated monthly series.

The [BAS provider notes](https://legacy.bas.ac.uk/met/gjma/sam.html) identify
ECMWF Operational Analysis infilling of some missing station pressures in
2025/26. Station observations also overlap with reanalysis inputs; these are
independent index definitions, not wholly independent observations. The summary
JSON retains seasonal correlations and comparisons ending in 2024 to inspect
robustness to recent values. The earliest ERA5 years cannot be checked against
indices that begin only in 1950 (AO) or 1957 (Marshall SAM).

Definitions: [CPC methodology](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/history/method.shtml),
[AO sign](https://pmel.noaa.gov/arctic-zone/detect/climate-ao.shtml),
[Marshall SAM](https://legacy.bas.ac.uk/met/gjma/sam.html).

Reference-data provenance (checksums verified before calculation):

{sources}
"""
        summaries = [("Summary of EOF analyses", eof_summary),
                     ("AO/SAM index validation and sign conventions", index_summary)]
        report = write_report(results, comparisons, metadata, reports, summaries=summaries)
        (reports / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        print(json.dumps(summary, indent=2, allow_nan=False))
        LOG.info("Completed. Report: %s", report)
        return report
    finally:
        field.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except (ValueError, OSError) as exc:
        LOG.error("Analysis stopped: %s", exc)
        sys.exit(1)
