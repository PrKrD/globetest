"""Regress native ERA5 surface fields on the saved undetrended 12-PC bases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.mslp_surface_regression import (COMPARE_TOL, ORTHOGONAL_TOL, Design, SPECS, SurfaceInputs,
    common_months, load_mask, load_pc, month_ids, month_name, process_variable)
from app.mslp_surface_regression_diagnostics import (make_figures, numerical_summary,
    turbulent_consistency, write_json, write_report, write_rows)

LOG = logging.getLogger("mslp_surface_regression")
BACKEND = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", help="Combined/separate/time-split NetCDF paths or quoted globs")
    p.add_argument("--mask", type=Path, help="ERA5 lsm on the exact native target grid")
    p.add_argument("--nh-pcs", type=Path, default=BACKEND/"diagnostics/mslp_eof_12/northern_hemisphere/1979_2026/eof.nc")
    p.add_argument("--sh-pcs", type=Path, default=BACKEND/"diagnostics/mslp_eof_12/southern_hemisphere/1979_2026/eof.nc")
    p.add_argument("--map-variable", action="append", default=[], metavar="KIND=NAME")
    p.add_argument("--output-dir", type=Path, default=BACKEND/"diagnostics/mslp_surface_regression")
    p.add_argument("--tile-latitudes", type=int, default=121)
    p.add_argument("--tile-longitudes", type=int, default=240)
    p.add_argument("--max-tiles", type=int, help="Pilot only: first N spatial tiles; separate output directory required")
    p.add_argument("--turbulent", action="store_true", help="Also analyse latent+sensible exchange and check linearity")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--no-coastlines", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--allow-other-resolution", action="store_true", help="Synthetic validation only")
    p.add_argument("--sensitivity", action="store_true", help="Approved annual and seasonal experiments; separate outputs")
    p.add_argument("--baseline-dir", type=Path, default=BACKEND/"diagnostics/mslp_surface_regression")
    args = p.parse_args(argv)
    if args.sensitivity and args.output_dir == BACKEND/"diagnostics/mslp_surface_regression":
        args.output_dir = BACKEND/"diagnostics/mslp_surface_regression_sensitivity"
    if min(args.tile_latitudes, args.tile_longitudes, args.dpi) < 1 or (args.max_tiles is not None and args.max_tiles < 1):
        p.error("Tile sizes, dpi and pilot tile count must be positive")
    if not args.render_only and (not args.input or not args.mask):
        p.error("--input and --mask are required for inspection or analysis")
    return args


def run(args):
    if args.sensitivity:
        from app.mslp_surface_sensitivity import run_sensitivity
        return run_sensitivity(args, LOG.info)
    out = args.output_dir.resolve()
    if args.render_only:
        render_started = time.monotonic()
        metadata = json.loads((out/"reports/metadata.json").read_text())
        if metadata.get("pilot"):
            raise ValueError("Pilot results cannot be rendered as a global scientific analysis")
        kinds = metadata["variables"]
        summary = numerical_summary(out, kinds, metadata["pc_diagnostics"])
        linearity = turbulent_consistency(out) if "turbulent" in kinds else []
        if not args.no_maps:
            make_figures(out, kinds, metadata["pc_diagnostics"], summary, args.dpi,
                         not args.no_coastlines, LOG.info)
        metadata["rendering"] = {"command": sys.argv, "completed_utc": datetime.now(timezone.utc).isoformat(),
                                 "elapsed_seconds": time.monotonic()-render_started, "dpi": args.dpi,
                                 "coastlines": not args.no_coastlines, "figures": len(list(out.rglob("*.png")))}
        write_json(out/"reports/metadata.json", metadata)
        write_report(out, metadata, summary, linearity)
        return summary
    if out.exists() and any(out.iterdir()) and not args.inspect_only:
        raise FileExistsError(f"Use a new empty output directory; refusing to overwrite {out}")
    mappings = {}
    for item in args.map_variable:
        key, value = item.split("=", 1)
        if key not in SPECS:
            raise ValueError(f"Unknown target {key}")
        mappings[key] = value
    inputs = SurfaceInputs(args.input, mappings, native=not args.allow_other_resolution)
    try:
        pcs = {"north": load_pc(args.nh_pcs, "north"), "south": load_pc(args.sh_pcs, "south")}
        common = common_months([month_ids(f.time) for f in inputs.fields.values()]+[p[0] for p in pcs.values()])
        fields = {}
        for kind, field in inputs.fields.items():
            ids = month_ids(field.time)
            fields[kind] = field.isel(time=np.flatnonzero(np.isin(ids, common)))
            if not np.array_equal(month_ids(fields[kind].time), common):
                raise ValueError("Internal time alignment failed")
        designs = {h: Design(x[np.isin(ids, common)]) for h, (ids, x, _) in pcs.items()}
        diagnostics = {h: d.diagnostics() for h, d in designs.items()}
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "python": platform.python_version(), "versions": {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "xarray", "h5netcdf", "h5py", "matplotlib", "cartopy")},
            "surface_inputs": inputs.metadata, "pc_inputs": {h: p[2] for h, p in pcs.items()},
            "pc_diagnostics": diagnostics, "first_month": month_name(common[0]),
            "last_month": month_name(common[-1]), "n_samples": len(common),
            "year_month_ids": common.tolist(), "variables": list(SPECS)+(["turbulent"] if args.turbulent else []),
            "regression_sample_matches_training": {h: bool(np.array_equal(common, p[0])) for h, p in pcs.items()},
            "detrended": False, "target_climatology": "separate calendar-month mean over common regression period",
            "native_resolution_primary": not args.allow_other_resolution, "pilot": args.max_tiles is not None,
            "tile_shape": [args.tile_latitudes, args.tile_longitudes],
            "orthogonality_tolerance": ORTHOGONAL_TOL,
            "coefficient_tolerance": {"rtol": COMPARE_TOL, "atol": "1e-10 times area-RMS target SD; floor SD=1e-12"},
            "scope": "turbulent surface heat and stress associations; no radiation, SST feedback fitting or complete F prime"}
        LOG.info("Common months %s–%s, T=%d", metadata["first_month"], metadata["last_month"], len(common))
        for h, diag in diagnostics.items():
            LOG.info("%s: rank=%d, condition=%.12g, max off-diagonal correlation=%.3e", h,
                     diag["rank"], diag["condition_number"], diag["maximum_offdiagonal_correlation"])
        if args.inspect_only:
            print(json.dumps(metadata, indent=2, default=str))
            return metadata
        LOG.info("Checking fixed ERA5 land–sea mask across all supplied records")
        mask, mask_metadata = load_mask(args.mask, next(iter(fields.values())), args.tile_latitudes)
        metadata["mask"] = mask_metadata
        write_json(out/"reports/metadata.json", metadata)
        for h, design in designs.items():
            xr.Dataset({"standardized_pc": (("time", "mode"), design.original),
                        "correlation": (("mode", "other_mode"), design.correlation),
                        "centred_gram": (("mode", "other_mode"), design.gram)},
                       coords={"time": np.array([month_name(i) for i in common], dtype="datetime64[ns]"),
                               "mode": np.arange(1, 13), "other_mode": np.arange(1, 13)},
                       attrs={"scaling": "original EOF-training sample SD; no restandardisation",
                              "input": str(args.nh_pcs if h == "north" else args.sh_pcs)}).to_netcdf(
                                  out/"reports"/f"{h}_pc_diagnostics.nc", engine="h5netcdf")
            write_rows(out/"reports"/f"{h}_pc_correlation.csv", [
                {"mode": i+1, **{f"PC{j+1}": float(design.correlation[i, j]) for j in range(12)}} for i in range(12)])
        metadata["processing"] = {}
        started = time.monotonic()
        for kind in metadata["variables"]:
            selected = [fields["latent"], fields["sensible"]] if kind == "turbulent" else [fields[kind]]
            metadata["processing"][kind] = process_variable(kind, selected, mask, designs, out,
                {"first_month": metadata["first_month"], "last_month": metadata["last_month"],
                 "n_samples": len(common), "input_metadata": "../../reports/metadata.json",
                 "predictor_diagnostics": diagnostics, "native_resolution_primary": int(not args.allow_other_resolution)},
                (args.tile_latitudes, args.tile_longitudes), args.max_tiles, LOG.info)
            LOG.info("Completed %s: %s", kind, metadata["processing"][kind])
            write_json(out/"reports/metadata.json", metadata)
        metadata["analysis_elapsed_seconds"] = time.monotonic()-started
        # macOS ru_maxrss is bytes; Linux is KiB.
        metadata["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        write_json(out/"reports/metadata.json", metadata)
        if args.max_tiles:
            LOG.info("Pilot complete; numerical outputs explicitly marked pilot, no global scientific claims")
            return metadata
        summary = numerical_summary(out, metadata["variables"], diagnostics)
        linearity = turbulent_consistency(out) if args.turbulent else []
        if not args.no_maps:
            make_figures(out, metadata["variables"], diagnostics, summary, args.dpi,
                         not args.no_coastlines, LOG.info)
        write_report(out, metadata, summary, linearity)
        LOG.info("Report: %s", out/"reports/surface_regression_report.md")
        expected = all(d["orthogonal"] for d in diagnostics.values())
        if expected and not (summary["all_coefficient_agreement_tests_pass"] and summary["all_reconstruction_tests_pass"]):
            raise ValueError("Orthogonal-sample numerical equivalence failed; inspect diagnostics")
        if linearity and not all(r["agreement_pass"] for r in linearity):
            raise ValueError("Turbulent coefficient linearity failed")
        return summary
    finally:
        inputs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
