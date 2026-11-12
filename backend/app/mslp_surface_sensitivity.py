"""Fixed-model annual evaluation and seasonal OLS on the saved MSLP basis.

Targets are streamed on the baseline grid; no time-space cubes are exported.
All reductions and regressions use float64. Baseline products are read-only.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import xarray as xr

from app.mslp_eof import monthly_anomalies
from app.mslp_surface_regression import (
    COUNTS, SPECS, Design, SurfaceInputs, TileWriter, cell_areas, expand,
    fingerprint, load_mask, load_pc, month_ids, month_name,
)
from app.mslp_surface_regression_diagnostics import REGIONS, write_json, write_rows

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
           "JJA": (6, 7, 8), "SON": (9, 10, 11)}
HEMISPHERES = {"north": "northern_hemisphere", "south": "southern_hemisphere"}
MODELS = ("year_round", "seasonal")
METRICS = ("r_squared", "rmse", "observed_variance", "reconstructed_variance",
           "residual_variance", "bias", "correlation", "sd_ratio")
EVALUATIONS = ("monthly_full", "monthly_complete", *SEASONS)


def annual_groups(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select complete calendar years, rejecting duplicate or unsorted months."""
    ids = np.asarray(ids, dtype=int)
    if ids.ndim != 1 or np.any(np.diff(ids) <= 0):
        raise ValueError("Months must be unique and ordered")
    years, groups = [], []
    for year in np.unique(ids//12):
        ix = np.flatnonzero(ids//12 == year)
        if len(ix) == 12 and np.array_equal(ids[ix] % 12, np.arange(12)):
            years.append(year)
            groups.append(ix)
    if len(years) < 2:
        raise ValueError("Require at least two complete calendar years")
    return np.array(years), np.stack(groups)


def annual_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Equal-month means: the exact same operator serves observations/predictions."""
    return np.asarray(values, dtype=np.float64)[groups].mean(axis=1)


def target_statistics(y: np.ndarray):
    """Cache target-only reductions shared by every competing reconstruction."""
    if not np.isfinite(y).all():
        raise ValueError("Evaluation must use identical finite samples")
    centred = y-y.mean(0)
    sst = np.einsum("ij,ij->j", centred, centred)
    threshold = (100*np.finfo(float).eps*np.maximum(np.max(np.abs(y), axis=0), 1e-12))**2*len(y)
    return centred, sst, sst > threshold


def evaluate(y: np.ndarray, predicted: np.ndarray, observed_statistics=None) -> dict[str, np.ndarray]:
    """Prediction-based skill, without modifying the prediction mean.

    Constant targets retain finite errors/variances, but undefined R², r and
    amplitude ratio. Residual variance is centred; RMSE includes bias.
    """
    y, predicted = np.asarray(y, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if y.shape != predicted.shape or y.ndim != 2 or len(y) < 2:
        raise ValueError("Require matching N-by-P arrays with N >= 2")
    if not np.isfinite(predicted).all():
        raise ValueError("Evaluation must use identical finite samples")
    n = len(y)
    yc, sst, varying = target_statistics(y) if observed_statistics is None else observed_statistics
    pc = predicted-predicted.mean(0)
    residual = predicted-y
    ssp = np.einsum("ij,ij->j", pc, pc)
    sse = np.einsum("ij,ij->j", residual, residual)
    r2, corr, ratio = (np.full(y.shape[1], np.nan) for _ in range(3))
    np.divide(sse, sst, out=r2, where=varying)
    r2 = 1-r2
    np.divide(np.einsum("ij,ij->j", yc, pc), np.sqrt(sst*ssp), out=corr,
              where=varying & (ssp > 0))
    np.divide(ssp, sst, out=ratio, where=varying)
    centred_residual = residual-residual.mean(0)
    return {"r_squared": r2, "rmse": np.sqrt(sse/n),
            "observed_variance": sst/(n-1), "reconstructed_variance": ssp/(n-1),
            "residual_variance": np.einsum("ij,ij->j", centred_residual, centred_residual)/(n-1),
            "bias": residual.mean(0), "correlation": corr, "sd_ratio": np.sqrt(ratio),
            "sse": sse, "sst": sst, "constant_target": ~varying}


def seasonal_designs(pc: np.ndarray, months: np.ndarray):
    designs, diagnostics = {}, {}
    for season, members in SEASONS.items():
        ix = np.flatnonzero(np.isin(months, members))
        design = Design(pc[ix])
        augmented = np.column_stack((np.ones(len(ix)), pc[ix]))
        diag = design.diagnostics()
        diag.update({"samples": len(ix), "standard_deviations": pc[ix].std(0, ddof=1).tolist(),
                     "augmented_rank": int(np.linalg.matrix_rank(augmented)),
                     "augmented_condition_number": float(np.linalg.cond(augmented)),
                     "nested": {str(k): {"rank": int(np.linalg.matrix_rank(design.x[:, :k])),
                                          "condition_number": float(np.linalg.cond(design.x[:, :k])),
                                          "coefficients_per_season": k+1,
                                          "coefficients_all_seasons": 4*(k+1)} for k in COUNTS}})
        designs[season] = (ix, design)
        diagnostics[season] = diag
    return designs, diagnostics


class RegionalAccumulator:
    """Sum cell statistics, retaining regional-mean series separately from skill."""
    def __init__(self):
        self.data = {}

    def add(self, key, y, prediction, metrics, weights, regional_series=None):
        eligible = np.isfinite(metrics["r_squared"]) & (weights > 0)
        supported = weights > 0
        if not supported.any():
            return
        w = weights[eligible]
        all_w = weights[supported]
        if key not in self.data:
            self.data[key] = {"weight": 0., "cells": 0, "n": len(y),
                              "all_weight": 0., "all_sse": 0.,
                              **{m: 0. for m in (*METRICS, "sse", "sst")},
                              "correlation_weight": 0., "target_series": np.zeros(len(y)),
                              "prediction_series": np.zeros(len(y))}
        row = self.data[key]
        row["weight"] += w.sum()
        row["all_weight"] += all_w.sum()
        row["all_sse"] += np.sum(all_w*metrics["sse"][supported])
        row["cells"] += len(w)
        for metric in (*METRICS, "sse", "sst"):
            use = eligible if metric in ("r_squared", "sse", "sst") else supported
            metric_w = weights[use]
            a = metrics[metric][use]
            valid = np.isfinite(a)
            row[metric] += np.sum(metric_w[valid]*a[valid])
            if metric == "correlation":
                row["correlation_weight"] += metric_w[valid].sum()
        # RMSE must be pooled through SSE, not averaged across cells.
        if regional_series is None:
            regional_series = y @ weights, prediction @ weights
        row["target_series"] += regional_series[0]
        row["prediction_series"] += regional_series[1]

    def finish(self, ids: np.ndarray, years: np.ndarray):
        rows, series = [], []
        for key, a in self.data.items():
            hemi, kind, model, k, evaluation, region, policy = key
            weight, n, all_weight = a["weight"], a["n"], a["all_weight"]
            observed_var = a["observed_variance"]/all_weight
            predicted_var = a["reconstructed_variance"]/all_weight
            row = {"hemisphere": hemi, "variable": kind, "model": model,
                   "retained_modes": k, "evaluation": evaluation, "region": region,
                   "mask_policy": policy, "n_samples": n, "valid_cells": a["cells"],
                   "valid_area_m2": float(weight),
                   "rmse_and_series_area_m2": float(all_weight),
                   "total_variance_fraction_explained": float(1-a["all_sse"]/(a["observed_variance"]*(n-1))) if a["observed_variance"] > 0 else np.nan,
                   "mean_local_r_squared": float(a["r_squared"]/weight) if weight else np.nan,
                   "area_weighted_rmse": float(np.sqrt(a["all_sse"]/(n*all_weight))),
                   "observed_variance": float(observed_var),
                   "reconstructed_variance": float(predicted_var),
                   "sd_ratio": float(np.sqrt(predicted_var/observed_var)) if observed_var > 0 else np.nan,
                   "residual_variance": float(a["residual_variance"]/all_weight),
                   "bias": float(a["bias"]/all_weight),
                   "mean_local_correlation": float(a["correlation"]/a["correlation_weight"]) if a["correlation_weight"] else np.nan}
            target, prediction = a["target_series"]/all_weight, a["prediction_series"]/all_weight
            sm = evaluate(target[:, None], prediction[:, None])
            row.update({"regional_mean_r_squared": float(sm["r_squared"][0]),
                        "regional_mean_correlation": float(sm["correlation"][0]),
                        "regional_mean_sd_ratio": float(sm["sd_ratio"][0])})
            rows.append(row)
            if evaluation == "annual":
                labels = years
            elif evaluation == "monthly_complete":
                labels = ids[np.isin(ids//12, years)]
            elif evaluation in SEASONS:
                labels = ids[np.isin(ids % 12+1, SEASONS[evaluation])]
            else:
                labels = ids
            for label, actual, fitted in zip(labels, target, prediction):
                series.append({**{name: row[name] for name in ("hemisphere", "variable", "model", "retained_modes", "evaluation", "region", "mask_policy")},
                               "date": str(label) if evaluation == "annual" else month_name(label),
                               "observed": float(actual), "reconstructed": float(fitted)})
        return rows, series


def _metric_schema(prefix, units, all_metrics=True):
    squared = f"({units})2"
    physical = {"rmse": units, "bias": units, "observed_variance": squared,
                "reconstructed_variance": squared, "residual_variance": squared}
    return {m: (prefix, physical.get(m, "1")) for m in (METRICS if all_metrics else METRICS[:2])}


def time_subset(values, indices):
    """Use a view for contiguous evaluations rather than copying a whole cube."""
    if len(indices) and indices[-1]-indices[0]+1 == len(indices):
        return values[indices[0]:indices[-1]+1]
    return values[indices]


def _baseline_check(metrics, joint, valid, ys, xs, k, maxima):
    """Scale-aware checks against saved fields, raising before interpretation."""
    tests = {"r_squared": joint.nested_r_squared.sel(retained_modes=k),
             "rmse": joint.nested_rmse.sel(retained_modes=k)}
    if k == 12:
        tests.update({"observed_variance": joint.target_variance,
                      "reconstructed_variance": joint.reconstruction_variance,
                      "residual_variance": joint.residual_sample_variance})
    for name, field in tests.items():
        reference = np.asarray(field.isel(latitude=ys, longitude=xs), dtype=float)[valid]
        actual = metrics[name]
        if not np.array_equal(np.isfinite(reference), np.isfinite(actual)):
            raise ValueError(f"Baseline finite-mask mismatch: {name}, K={k}")
        finite = np.isfinite(reference)
        if not finite.any():
            continue
        delta = float(np.max(np.abs(actual[finite]-reference[finite])))
        maxima[name] = max(maxima.get(name, 0.), delta)
        scale = 1. if name == "r_squared" else max(float(np.max(np.abs(reference[finite]))), 1e-12)
        if not np.allclose(actual[finite], reference[finite], rtol=1e-10, atol=1e-10*scale):
            raise ValueError(f"Baseline consistency failed: {name}, K={k}, max error={delta}")


def process_sensitivity(kind, fields, mask, pcs, designs, baseline, out, ids,
                        tile_shape=(121, 240), max_tiles=None, progress=print):
    ref = fields[0]
    lat, lon = ref.latitude.values, ref.longitude.values
    years, groups = annual_groups(ids)
    complete = np.flatnonzero(np.isin(ids//12, years))
    subsets = {"monthly_full": np.arange(len(ids)), "monthly_complete": complete,
               **{s: np.flatnonzero(np.isin(ids % 12+1, mm)) for s, mm in SEASONS.items()}}
    units = SPECS[kind][-1] if kind in SPECS else "W m-2"
    acc, checks = RegionalAccumulator(), {h: {} for h in HEMISPHERES}
    pattern_rows = []
    writers, baseline_files = {}, {}
    success = False
    attrs = {"target": kind, "first_month": month_name(ids[0]), "last_month": month_name(ids[-1]),
             "n_samples": len(ids), "annual_years": years.tolist(), "models": MODELS,
             "evaluations": EVALUATIONS, "seasons": list(SEASONS), "numerical_dtype": "float64",
             "annual_weighting": "equal monthly weights; complete calendar years only",
             "climatology": "saved baseline monthly climatology; unchanged in all evaluations",
             "heat_positive": "downward", "stress_positive": "eastward/northward",
             "missing_policy": "fixed full-record baseline complete-cell mask",
             "sea_ice": "retained; exchange over ice is not necessarily open-water ocean heating",
             "pilot": int(max_tiles is not None), "input_metadata": "../../reports/metadata.json"}
    started = time.monotonic()
    pattern_acc = {}
    # Regional weights are calculated once, with exact clipping of boundary cells.
    areas = {r: cell_areas(lat, lon, bounds) for r, bounds in REGIONS.items()}
    with ExitStack() as stack:
        try:
            for hemi, directory in HEMISPHERES.items():
                joint = stack.enter_context(xr.open_dataset(baseline/directory/"joint"/f"{kind}.nc", cache=False))
                if not np.array_equal(joint.latitude, lat) or not np.array_equal(joint.longitude, lon):
                    raise ValueError("Baseline target grid differs")
                baseline_files[hemi] = joint
                local_attrs = {**attrs, "hemisphere": hemi}
                coeff = {}
                dims = {"season": 4, "model": 2, "evaluation": len(EVALUATIONS)}
                for k in COUNTS:
                    dims[f"pc_{k}"] = k
                    coeff[f"coefficient_{k}"] = (("season", f"pc_{k}"), units+" per original PC SD")
                    coeff[f"intercept_{k}"] = (("season",), units)
                specs = {
                    "coefficients": (out/"seasonal_regressions"/hemi/f"{kind}_coefficients.nc", coeff),
                    "monthly": (out/"seasonal_regressions"/hemi/f"{kind}_diagnostics.nc",
                                _metric_schema(("model", "evaluation", "retained_modes"), units, False)),
                    "monthly_detail": (out/"seasonal_regressions"/hemi/f"{kind}_12pc_detail.nc",
                                       {**_metric_schema(("model", "evaluation"), units),
                                        "constant_target": (("model", "evaluation"), "1")}),
                    "annual": (out/"annual_variability"/hemi/f"{kind}_diagnostics.nc",
                               {**_metric_schema(("model", "retained_modes"), units),
                                "constant_target": (("model", "retained_modes"), "1")}),
                }
                for label, (path, schema) in specs.items():
                    writers[hemi, label] = TileWriter(path, lat, lon, schema, local_attrs, tile_shape, dims)
                    f = writers[hemi, label].file
                    f.attrs["evaluation_sample_counts"] = json.dumps({name: len(ix) for name, ix in subsets.items()})
                    f.attrs["annual_sample_count"] = len(years)
                    for dim, size in dims.items():
                        coordinate = f.create_variable(dim, (dim,), dtype="i4")
                        coordinate[:] = np.arange(1, size+1) if dim.startswith("pc_") else np.arange(size)
                        if dim in ("season", "model", "evaluation"):
                            labels = {"season": list(SEASONS), "model": MODELS, "evaluation": EVALUATIONS}[dim]
                            coordinate.attrs["flag_values"] = np.arange(size, dtype=np.int32)
                            coordinate.attrs["flag_meanings"] = " ".join(labels)
            tiles = 0
            for j in range(0, len(lat), tile_shape[0]):
                for i in range(0, len(lon), tile_shape[1]):
                    ys, xs = slice(j, min(j+tile_shape[0], len(lat))), slice(i, min(i+tile_shape[1], len(lon)))
                    raw = None
                    for field in fields:
                        a = np.asarray(field.isel(latitude=ys, longitude=xs).transpose("time", "latitude", "longitude"), dtype=float)
                        a *= float(field.attrs["rate_factor"])
                        raw = a if raw is None else raw+a
                    land = mask[ys, xs]
                    valid = (land < .5) & np.isfinite(raw).all(axis=0)
                    nh = baseline_files["north"]
                    expected = np.asarray(nh.valid_sample_count.isel(latitude=ys, longitude=xs)) == len(ids)
                    if not np.array_equal(valid, expected):
                        raise ValueError("Current complete-cell mask differs from saved baseline")
                    if valid.any():
                        saved_clim = np.asarray(nh.monthly_climatology.isel(latitude=ys, longitude=xs))[:, valid]
                        y = raw[:, valid]-saved_clim[ids % 12]
                        computed, clim = monthly_anomalies(raw[:, valid], ids % 12+1)
                        if not np.allclose(clim, saved_clim, rtol=1e-12, atol=1e-12):
                            raise ValueError("Saved climatology differs from input")
                        np.testing.assert_allclose(y, computed, rtol=1e-12, atol=1e-12)
                        del computed, raw
                        ya = annual_mean(y, groups)
                        regional_weights = [(region, policy, (areas[region][ys, xs]*ocean)[valid])
                                            for region in REGIONS for policy, ocean in
                                            (("primary", land < .5), ("pure_ocean", land <= .05))]
                        weight_matrix = np.column_stack([item[2] for item in regional_weights])
                        observed_subsets = {label: time_subset(y, ix) for label, ix in subsets.items()}
                        observed_stats = {label: target_statistics(a) for label, a in observed_subsets.items()}
                        observed_series = {label: a @ weight_matrix for label, a in observed_subsets.items()}
                        annual_stats, annual_series = target_statistics(ya), ya @ weight_matrix
                        for hemi in HEMISPHERES:
                            joint = baseline_files[hemi]
                            if not np.array_equal(np.asarray(joint.valid_sample_count.isel(latitude=ys, longitude=xs)) == len(ids), valid):
                                raise ValueError("Hemisphere baseline masks differ")
                            beta = np.asarray(joint.coefficient.isel(latitude=ys, longitude=xs))[:, valid]
                            intercept = np.asarray(joint.intercept.isel(latitude=ys, longitude=xs))[valid]
                            fits = {s: dict((k, (b, a)) for k, b, a in d.joint_fits(y[ix]))
                                    for s, (ix, d) in designs[hemi].items()}
                            for ki, k in enumerate(COUNTS):
                                coefficients = np.stack([fits[s][k][0] for s in SEASONS])
                                intercepts = np.stack([fits[s][k][1] for s in SEASONS])
                                writers[hemi, "coefficients"].write(f"coefficient_{k}", expand(coefficients, valid), ys, xs)
                                writers[hemi, "coefficients"].write(f"intercept_{k}", expand(intercepts, valid), ys, xs)
                                # Selected full-model coefficient-pattern sums are tiny.
                                if k == 12:
                                    selected = (1, 2) if hemi == "north" else (1, 11)
                                    for mode in selected:
                                        patterns = np.vstack((beta[mode-1], coefficients[:, mode-1]))
                                        for region in REGIONS:
                                            for policy, ocean in (("primary", land < .5), ("pure_ocean", land <= .05)):
                                                w = (areas[region][ys, xs]*ocean)[valid]
                                                key = (hemi, mode, region, policy)
                                                sums = pattern_acc.setdefault(key, {"w": 0., "sum": np.zeros(5), "cross": np.zeros((5, 5))})
                                                sums["w"] += w.sum()
                                                sums["sum"] += patterns @ w
                                                sums["cross"] += (patterns*w) @ patterns.T
                                for mi, model in enumerate(MODELS):
                                    if model == "year_round":
                                        prediction = intercept+pcs[hemi][:, :k] @ beta[:k]
                                    else:
                                        prediction = np.empty_like(y)
                                        for season, (ix, _) in designs[hemi].items():
                                            b, a = fits[season][k]
                                            prediction[ix] = a+pcs[hemi][ix, :k] @ b
                                    monthly_maps = {m: np.full((len(EVALUATIONS), *valid.shape), np.nan) for m in METRICS[:2]}
                                    detail = {m: np.full((len(EVALUATIONS), *valid.shape), np.nan) for m in (*METRICS, "constant_target")} if k == 12 else {}
                                    for ei, (label, ix) in enumerate(subsets.items()):
                                        observed_subset = observed_subsets[label]
                                        predicted_subset = time_subset(prediction, ix)
                                        metrics = evaluate(observed_subset, predicted_subset, observed_stats[label])
                                        predicted_series = predicted_subset @ weight_matrix
                                        if model == "year_round" and label == "monthly_full":
                                            _baseline_check(metrics, joint, valid, ys, xs, k, checks[hemi])
                                        for m in monthly_maps:
                                            monthly_maps[m][ei] = expand(metrics[m], valid)
                                        for m in detail:
                                            detail[m][ei] = expand(metrics[m], valid)
                                        for wi, (region, policy, weights) in enumerate(regional_weights):
                                            acc.add((hemi, kind, model, k, label, region, policy), observed_subset, predicted_subset,
                                                    metrics, weights, (observed_series[label][:, wi], predicted_series[:, wi]))
                                    for m, a in monthly_maps.items():
                                        writers[hemi, "monthly"].file.variables[m][mi, :, ki, ys, xs] = a
                                    for m, a in detail.items():
                                        writers[hemi, "monthly_detail"].file.variables[m][mi, :, ys, xs] = a
                                    pa = annual_mean(prediction, groups)
                                    am = evaluate(ya, pa, annual_stats)
                                    predicted_annual_series = pa @ weight_matrix
                                    for m in (*METRICS, "constant_target"):
                                        writers[hemi, "annual"].file.variables[m][mi, ki, ys, xs] = expand(am[m], valid)
                                    for wi, (region, policy, weights) in enumerate(regional_weights):
                                        acc.add((hemi, kind, model, k, "annual", region, policy), ya, pa, am, weights,
                                                (annual_series[:, wi], predicted_annual_series[:, wi]))
                    tiles += 1
                    progress(f"{kind}: tile {tiles}, rows {j}:{ys.stop}, columns {i}:{xs.stop}, complete ocean={valid.sum()}")
                    if max_tiles and tiles >= max_tiles:
                        break
                if max_tiles and tiles >= max_tiles:
                    break
            success = True
        finally:
            for writer in writers.values():
                writer.close(success)
    for (hemi, mode, region, policy), sums in pattern_acc.items():
        if sums["w"] <= 0:
            continue
        means = sums["sum"]/sums["w"]
        cov = sums["cross"]/sums["w"]-np.outer(means, means)
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        denominator = np.outer(sd, sd)
        corr = np.full((5, 5), np.nan)
        np.divide(cov, denominator, out=corr, where=denominator > 0)
        labels = ("year_round", *SEASONS)
        for a, first in enumerate(labels):
            for b, second in enumerate(labels):
                pattern_rows.append({"hemisphere": hemi, "variable": kind, "mode": mode,
                                     "pattern": first, "comparison_pattern": second,
                                     "region": region, "mask_policy": policy,
                                     "weighted_pattern_correlation": float(corr[a, b]),
                                     "rms_coefficient": float(np.sqrt(max(sums["cross"][a, a]/sums["w"], 0)))})
    rows, series = acc.finish(ids, years)
    return rows, series, pattern_rows, {"tiles": tiles, "baseline_max_errors": checks,
                                       "elapsed_seconds": time.monotonic()-started}


def validate_baseline(metadata, ids, pc_metadata, surface_metadata, baseline):
    """Bind saved maps to the exact PC values, source files and common sample."""
    if metadata.get("pilot") or metadata.get("detrended"):
        raise ValueError("Require completed undetrended baseline")
    if metadata["year_month_ids"] != ids.tolist():
        raise ValueError("Baseline common months differ")
    for hemi in HEMISPHERES:
        if pc_metadata[hemi]["pc_sha256"] != metadata["pc_inputs"][hemi]["pc_sha256"]:
            raise ValueError("PC values differ from saved baseline")
    for kind in SPECS:
        old = metadata["surface_inputs"][kind]["sources"]
        new = surface_metadata[kind]["sources"]
        if [(v["path"], v["bytes"], v["mtime_ns"], v["variable"], v["factor"]) for v in old] != [
            (v["path"], v["bytes"], v["mtime_ns"], v["variable"], v["factor"]) for v in new]:
            raise ValueError("Surface provenance differs from saved baseline")
    if not (baseline/"reports/summary.json").is_file():
        raise ValueError("Missing baseline summary")


def differences(rows):
    lookup = {(r["hemisphere"], r["variable"], r["retained_modes"], r["evaluation"], r["region"], r["mask_policy"], r["model"]): r for r in rows}
    result = []
    for r in rows:
        if r["model"] != "seasonal":
            continue
        key = (r["hemisphere"], r["variable"], r["retained_modes"], r["evaluation"], r["region"], r["mask_policy"])
        b = lookup[(*key, "year_round")]
        result.append({**{k: r[k] for k in ("hemisphere", "variable", "retained_modes", "evaluation", "region", "mask_policy", "n_samples")},
                       "year_round_r_squared": b["total_variance_fraction_explained"],
                       "seasonal_r_squared": r["total_variance_fraction_explained"],
                       "r_squared_change_percentage_points": 100*(r["total_variance_fraction_explained"]-b["total_variance_fraction_explained"]),
                       "rmse_change_seasonal_minus_year_round": r["area_weighted_rmse"]-b["area_weighted_rmse"],
                       "residual_variance_change": r["residual_variance"]-b["residual_variance"]})
    return result


def run_sensitivity(args, progress=print):
    out, baseline = args.output_dir.resolve(), args.baseline_dir.resolve()
    if out == baseline or baseline in out.parents or out in baseline.parents:
        raise ValueError("Sensitivity outputs must be separate from baseline")
    if args.render_only:
        from app.mslp_surface_sensitivity_diagnostics import render
        return render(out, baseline, args.dpi, not args.no_coastlines, not args.no_maps, progress)
    if out.exists() and any(out.iterdir()) and not args.inspect_only:
        raise FileExistsError(f"Refusing to overwrite {out}")
    old_metadata = json.loads((baseline/"reports/metadata.json").read_text())
    mappings = dict(item.split("=", 1) for item in args.map_variable)
    if set(mappings)-set(SPECS):
        raise ValueError("Unknown surface variable mapping")
    inputs = SurfaceInputs(args.input, mappings, native=not args.allow_other_resolution)
    try:
        loaded = {h: load_pc(path, h) for h, path in (("north", args.nh_pcs), ("south", args.sh_pcs))}
        ids = loaded["north"][0]
        for field in inputs.fields.values():
            if not np.array_equal(month_ids(field.time), ids):
                raise ValueError("Sensitivity requires the exact baseline common months")
        if not np.array_equal(loaded["south"][0], ids):
            raise ValueError("Hemisphere months differ")
        pc_metadata = {h: item[2] for h, item in loaded.items()}
        validate_baseline(old_metadata, ids, pc_metadata, inputs.metadata, baseline)
        years, groups = annual_groups(ids)
        pcs = {h: item[1] for h, item in loaded.items()}
        designs, diagnostics = {}, {}
        for h in HEMISPHERES:
            designs[h], diagnostics[h] = seasonal_designs(pcs[h], ids % 12+1)
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
                    "baseline_directory": str(baseline), "baseline_metadata": fingerprint(baseline/"reports/metadata.json"),
                    "surface_inputs": inputs.metadata, "pc_inputs": pc_metadata,
                    "first_month": month_name(ids[0]), "last_month": month_name(ids[-1]), "n_samples": len(ids),
                    "annual_years": years.tolist(), "annual_samples": len(years),
                    "complete_year_months": int(groups.size), "seasonal_pc_diagnostics": diagnostics,
                    "counts": COUNTS, "models": MODELS, "evaluations": EVALUATIONS,
                    "annual_weighting": "equal monthly weights", "detrended": False,
                    "pc_scaling": "original full-period sample SD; seasonal centring only for intercept handling",
                    "climatology": "saved 574-month baseline calendar-month climatologies, never refitted",
                    "coefficient_counts_per_cell": {str(k): {"year_round": k+1, "seasonal": 4*(k+1)} for k in COUNTS},
                    "numerical_dtype": "float64", "python": sys.version,
                    "baseline_tolerance": {"rtol": 1e-10, "atol": "1e-10 times field scale; dimensionless R² scale=1"},
                    "tile_shape": [args.tile_latitudes, args.tile_longitudes], "pilot": args.max_tiles is not None,
                    "thread_settings": {name: os.environ.get(name, "library default") for name in
                                        ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
                    "native_resolution_primary": not args.allow_other_resolution,
                    "variables": [*SPECS, "turbulent"],
                    "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "xarray", "h5netcdf", "h5py", "matplotlib", "cartopy")}}
        metadata["baseline_products"] = {h: {kind: fingerprint(baseline/directory/"joint"/f"{kind}.nc")
                                            for kind in metadata["variables"]}
                                         for h, directory in HEMISPHERES.items()}
        if args.inspect_only:
            progress(json.dumps(metadata, indent=2, default=str))
            return metadata
        mask, mask_metadata = load_mask(args.mask, inputs.fields["latent"], args.tile_latitudes)
        if any(mask_metadata[k] != old_metadata["mask"][k] for k in ("path", "bytes", "mtime_ns")):
            raise ValueError("Mask provenance differs from saved baseline")
        metadata["mask"] = mask_metadata
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/"reports/metadata.json", metadata)
        write_json(out/"reports/seasonal_pc_diagnostics.json", diagnostics)
        correlation_rows = []
        for h in HEMISPHERES:
            for s in SEASONS:
                for n, values in enumerate(diagnostics[h][s]["correlation"]):
                    correlation_rows.append({"hemisphere": h, "season": s, "mode": n+1,
                                             **{f"PC{j+1}": v for j, v in enumerate(values)}})
        write_rows(out/"seasonal_regressions/pc_correlations.csv", correlation_rows)
        xr.Dataset({"correlation": (("hemisphere", "season", "mode", "other_mode"),
                    np.array([[diagnostics[h][s]["correlation"] for s in SEASONS] for h in HEMISPHERES])),
                    "pc_mean": (("hemisphere", "season", "mode"), np.array([[diagnostics[h][s]["means"] for s in SEASONS] for h in HEMISPHERES])),
                    "pc_sd": (("hemisphere", "season", "mode"), np.array([[diagnostics[h][s]["standard_deviations"] for s in SEASONS] for h in HEMISPHERES]))},
                   coords={"hemisphere": list(HEMISPHERES), "season": list(SEASONS), "mode": np.arange(1, 13), "other_mode": np.arange(1, 13)}).to_netcdf(out/"seasonal_regressions/pc_diagnostics.nc", engine="h5netcdf")
        all_rows, all_series, all_patterns = [], [], []
        metadata["processing"] = {}
        started = time.monotonic()
        for kind in metadata["variables"]:
            fields = [inputs.fields["latent"], inputs.fields["sensible"]] if kind == "turbulent" else [inputs.fields[kind]]
            rows, series, patterns, details = process_sensitivity(kind, fields, mask, pcs, designs, baseline, out, ids,
                (args.tile_latitudes, args.tile_longitudes), args.max_tiles, progress)
            all_rows.extend(rows)
            all_series.extend(series)
            all_patterns.extend(patterns)
            metadata["processing"][kind] = details
            write_json(out/"reports/metadata.json", metadata)
        write_rows(out/"reports/regional_metrics.csv", all_rows)
        write_rows(out/"reports/regional_timeseries.csv", all_series)
        write_rows(out/"seasonal_regressions/pattern_diagnostics.csv", all_patterns)
        write_rows(out/"annual_variability/regional_variance.csv", [r for r in all_rows if r["model"] == "year_round" and r["evaluation"] in ("annual", "monthly_complete", "monthly_full")])
        write_rows(out/"seasonal_regressions/regional_variance.csv", [r for r in all_rows if r["evaluation"] != "annual"])
        write_rows(out/"annual_seasonal_comparison/regional_variance.csv", [r for r in all_rows if r["evaluation"] in ("annual", "monthly_complete")])
        delta = differences(all_rows)
        write_rows(out/"reports/model_differences.csv", delta)
        if any(r["r_squared_change_percentage_points"] < -1e-7 for r in delta if r["evaluation"] in ("monthly_full", *SEASONS)):
            raise ValueError("Seasonal nested-model SSE invariant failed")
        metadata["analysis_elapsed_seconds"] = time.monotonic()-started
        metadata["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)
        metadata["completed"] = True
        metadata["numerical_output_bytes"] = sum(p.stat().st_size for p in out.rglob("*.nc"))
        write_json(out/"reports/metadata.json", metadata)
        if not args.max_tiles:
            from app.mslp_surface_sensitivity_diagnostics import render
            render(out, baseline, args.dpi, not args.no_coastlines, not args.no_maps, progress)
        return metadata
    finally:
        inputs.close()
