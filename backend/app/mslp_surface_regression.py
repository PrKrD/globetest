"""Native-grid, common-sample surface regressions on saved monthly MSLP PCs.

No EOF fitting, reorthogonalisation, target regridding or regularisation occurs.
All numerical work is float64; spatial weights enter summaries, never OLS.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import glob
import hashlib
import json
from pathlib import Path

import h5netcdf
import numpy as np
from scipy.linalg import solve_triangular
import xarray as xr

from app.mslp_eof import monthly_anomalies, validate_grid, validate_months

SPECS = {
    "latent": (235034, 147, ("avg_slhtf", "mslhf"), "slhf", "W m-2"),
    "sensible": (235033, 146, ("avg_ishf", "msshf"), "sshf", "W m-2"),
    "eastward_stress": (235041, 180, ("avg_iews", "metss"), "ewss", "N m-2"),
    "northward_stress": (235042, 181, ("avg_inss", "mntss"), "nsss", "N m-2"),
}
COUNTS = (1, 3, 6, 12)
ORTHOGONAL_TOL = 1e-12
COMPARE_TOL = 1e-10


def fingerprint(path: str | Path) -> dict:
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def month_ids(time: xr.DataArray) -> np.ndarray:
    validate_months(time)
    return time.dt.year.values.astype(int)*12 + time.dt.month.values.astype(int)-1


def month_name(value: int) -> str:
    y, m = divmod(int(value), 12)
    return f"{y:04d}-{m+1:02d}"


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for target, aliases in {"time": ("valid_time",), "latitude": ("lat",),
                            "longitude": ("lon",)}.items():
        if target not in ds.dims:
            candidates = [n for n in aliases if n in ds.dims]
            if len(candidates) == 1:
                rename[candidates[0]] = target
    ds = ds.rename(rename)
    if "longitude" in ds.coords:
        lon = np.mod(ds.longitude.values, 360)
        if not np.array_equal(lon, ds.longitude.values):
            ds = ds.assign_coords(longitude=lon).sortby("longitude")
    return ds


def rate_factor(field: xr.DataArray, kind: str, dataset_attrs: dict) -> float:
    """Validate identity AND units; never infer interval from month length."""
    rate_id, accumulated_id, aliases, accumulated_name, _ = SPECS[kind]
    name = str(field.name)
    param = field.attrs.get("GRIB_paramId")
    if param is not None:
        param = int(param)
        if param not in (rate_id, accumulated_id):
            raise ValueError(f"{kind}: wrong GRIB parameter {param} ({name})")
        accumulated = param == accumulated_id
    else:
        if name not in (*aliases, accumulated_name):
            raise ValueError(f"{kind}: cannot establish parameter identity for {name}")
        accumulated = name == accumulated_name
    units = str(field.attrs.get("units", "")).lower().replace("**", "").replace("^", "").replace(" ", "")
    rate_units = {"wm-2", "w/m2"} if kind in ("latent", "sensible") else {"nm-2", "n/m2", "pa"}
    acc_units = {"jm-2", "j/m2"} if kind in ("latent", "sensible") else {"nm-2s", "nsm-2", "n/m2s"}
    if not accumulated and units in rate_units:
        return 1.0
    if accumulated and units in acc_units:
        history = str(dataset_attrs.get("history", "")).lower()
        stream = str(field.attrs.get("GRIB_stream", dataset_attrs.get("GRIB_stream", ""))).lower()
        if stream in ("moda", "edmo") or any(f'"{s}"' in history for s in ("moda", "edmo")):
            return 1/86400
        raise ValueError("Accumulated flux needs verified moda/edmo daily-mean provenance")
    raise ValueError(f"{kind}: units {field.attrs.get('units')!r} inconsistent with parameter {param}")


class SurfaceInputs:
    """Own lazy source files; support combined files, separate files and time splits."""
    def __init__(self, paths: list[str], mappings: dict[str, str] | None = None,
                 native: bool = True):
        self.stack = ExitStack()
        self.fields = {}
        self.metadata = {}
        try:
            matched = sorted({str(Path(p).resolve()) for pattern in paths for p in glob.glob(pattern)})
            if not matched:
                raise ValueError("No surface inputs matched")
            datasets = [canonical(self.stack.enter_context(xr.open_dataset(p, cache=False))) for p in matched]
            for kind, (_, _, aliases, accumulated_name, _) in SPECS.items():
                candidates = []
                details = []
                for path, ds in zip(matched, datasets):
                    names = [mappings[kind]] if mappings and kind in mappings else [*aliases, accumulated_name]
                    found = [n for n in names if n in ds.data_vars]
                    if len(found) > 1:
                        raise ValueError(f"Ambiguous {kind} in {path}; provide explicit variable mapping")
                    if not found:
                        continue
                    field = ds[found[0]]
                    factor = rate_factor(field, kind, ds.attrs)
                    for dim in set(field.dims)-{"time", "latitude", "longitude", "expver"}:
                        if field.sizes[dim] != 1:
                            raise ValueError(f"Unsupported dimension {dim}")
                        field = field.isel({dim: 0}, drop=True)
                    if not {"time", "latitude", "longitude"}.issubset(field.dims):
                        raise ValueError(f"Missing surface dimensions: {field.dims}")
                    # Conversion is delayed to each tile, not materialised globally.
                    field.attrs = {**field.attrs, "rate_factor": factor}
                    candidates.append(field)
                    details.append({**fingerprint(path), "variable": found[0], "factor": factor,
                                    "attributes": dict(ds[found[0]].attrs), "dataset_attributes": dict(ds.attrs),
                                    "chunks": ds[found[0]].encoding.get("chunksizes")})
                if not candidates:
                    raise ValueError(f"No input for {kind}")
                factors = {v.attrs["rate_factor"] for v in candidates}
                if len(factors) != 1:
                    raise ValueError("Mixed accumulated and rate files for one variable are not supported")
                field = candidates[0] if len(candidates) == 1 else xr.concat(
                    candidates, dim="time", join="exact", coords="minimal", compat="equals")
                field = field.sortby("time")
                ids = month_ids(field.time)
                validate_grid(field.sortby("latitude"), resolution=.25 if native else None)
                if native and set([float(field.latitude.min()), float(field.latitude.max())]) != {-90., 90.}:
                    raise ValueError("Require complete global latitude coverage")
                self.fields[kind] = field
                self.metadata[kind] = {"sources": details, "first_month": month_name(ids[0]),
                                       "last_month": month_name(ids[-1]), "samples": len(ids)}
                if "expver" in field.coords:
                    self.metadata[kind]["expver_labels"] = np.unique(field.expver.values).tolist()
                if "expver" in field.dims:
                    raise ValueError("Select/merge expver dimension explicitly before surface regression; no averaging")
            ref = next(iter(self.fields.values()))
            for field in self.fields.values():
                for dim in ("latitude", "longitude"):
                    if not np.array_equal(ref[dim], field[dim]):
                        raise ValueError("Surface grids differ; no interpolation allowed")
        except Exception:
            self.close()
            raise

    def close(self):
        self.stack.close()


def load_pc(path: str | Path, hemisphere: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with xr.open_dataset(path) as ds:
        if ds.sizes.get("mode") != 12 or not np.array_equal(ds.mode.values, np.arange(1, 13)):
            raise ValueError("Require exactly modes 1–12")
        if int(ds.attrs.get("detrended", -1)) != 0:
            raise ValueError("Require explicitly undetrended EOFs")
        if ds.attrs.get("hemisphere") != hemisphere:
            raise ValueError("Wrong hemisphere EOF input")
        ids = month_ids(ds.time)
        if month_name(ids[0]) != "1979-01" or ids[-1] > 2026*12+11:
            raise ValueError("Require 1979–2026 EOF training basis")
        if ds.attrs.get("native_resolution_primary") != 1:
            raise ValueError("Require native-resolution primary EOFs")
        pc = np.asarray(ds.standardized_pc.transpose("time", "mode"), dtype=np.float64)
        if not np.isfinite(pc).all():
            raise ValueError("Nonfinite PCs")
        if not np.allclose(pc.mean(0), 0, atol=1e-10) or not np.allclose(pc.var(0, ddof=1), 1, atol=1e-10, rtol=0):
            raise ValueError("Saved PCs are not zero-mean unit sample-variance series")
        metadata = {**fingerprint(path), "attributes": dict(ds.attrs),
                    "training_first_month": month_name(ids[0]), "training_last_month": month_name(ids[-1]),
                    "pc_sha256": hashlib.sha256(pc.tobytes()).hexdigest()}
    return ids, pc, metadata


def common_months(series: list[np.ndarray]) -> np.ndarray:
    start, end = max(v[0] for v in series), min(v[-1] for v in series)
    common = np.arange(start, end+1)
    if len(common) < 24:
        raise ValueError("Common regression interval must contain at least 24 months")
    for ids in series:
        if not np.all(np.diff(ids) == 1) or not np.isin(common, ids).all():
            raise ValueError("Missing or duplicate months; no filling or selective dropping")
    return common


def load_mask(path: str | Path, reference: xr.DataArray, tile_lat: int = 121) -> tuple[np.ndarray, dict]:
    """Verify a repeated monthly mask is static, using bounded latitude slabs."""
    with xr.open_dataset(path, cache=False) as raw:
        ds = canonical(raw)
        if "lsm" not in ds:
            raise ValueError("Mask must contain lsm")
        field = ds.lsm
        for dim in set(field.dims)-{"time", "latitude", "longitude"}:
            if field.sizes[dim] != 1:
                raise ValueError(f"Unsupported mask dimension {dim}")
            field = field.isel({dim: 0}, drop=True)
        if int(field.attrs.get("GRIB_paramId", 172)) != 172:
            raise ValueError("Mask has the wrong parameter ID")
        for dim in ("latitude", "longitude"):
            if not np.array_equal(np.sort(field[dim].values), np.sort(reference[dim].values)):
                raise ValueError("Mask must have the identical target grid, not a finer/subset grid")
        field = field.sel(latitude=reference.latitude, longitude=reference.longitude)
        for dim in ("latitude", "longitude"):
            if not np.array_equal(field[dim], reference[dim]):
                raise ValueError("Mask grid does not match target grid exactly")
        mask = np.empty((reference.sizes["latitude"], reference.sizes["longitude"]), dtype=float)
        max_change = 0.
        tolerance = 2*float(np.finfo(np.float32).eps)
        for j in range(0, len(mask), tile_lat):
            for i in range(0, mask.shape[1], 240):
                tile = field.isel(latitude=slice(j, j+tile_lat), longitude=slice(i, i+240))
                if "time" in tile.dims:
                    values = np.asarray(tile.transpose("time", "latitude", "longitude"), dtype=float)
                    baseline = values[0]
                    max_change = max(max_change, float(np.max(np.abs(values-baseline))))
                    stable = np.all((values < .5) == (baseline < .5)) and np.all((values <= .05) == (baseline <= .05))
                else:
                    values = baseline = np.asarray(tile.transpose("latitude", "longitude"), dtype=float)
                    stable = True
                if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
                    raise ValueError("Land fraction must be finite and within [0, 1]")
                if max_change > tolerance or not stable:
                    raise ValueError("Land–sea mask changes over time; fixed-mask policy requires review")
                mask[j:j+baseline.shape[0], i:i+baseline.shape[1]] = baseline
        return mask, {**fingerprint(path), "attributes": dict(field.attrs),
                      "maximum_temporal_change": max_change, "absolute_static_tolerance": tolerance,
                      "classification_invariance": "both thresholds verified across every month",
                      "fixed_fraction": "first supplied mask record; only float32 encoding-level changes permitted",
                      "ocean_rule": "lsm < 0.5",
                      "pure_ocean_rule": "lsm <= 0.05", "sea_ice": "retained; no SST mask"}


def cell_areas(latitude: np.ndarray, longitude: np.ndarray,
               bounds: tuple[float, float] = (-90, 90)) -> np.ndarray:
    lat = np.asarray(latitude, dtype=float)
    half = abs(float(lat[1]-lat[0]))/2
    lower = np.maximum(np.maximum(lat-half, -90), bounds[0])
    upper = np.minimum(np.minimum(lat+half, 90), bounds[1])
    strip = np.maximum(np.sin(np.deg2rad(upper))-np.sin(np.deg2rad(lower)), 0)
    return (6371000.**2 * np.deg2rad(abs(longitude[1]-longitude[0])) *
            strip[:, None] * np.ones((1, len(longitude))))


@dataclass
class Design:
    original: np.ndarray

    def __post_init__(self):
        self.original = np.asarray(self.original, dtype=np.float64)
        if self.original.ndim != 2 or self.original.shape[1] != 12 or not np.isfinite(self.original).all():
            raise ValueError("Require finite T-by-12 predictors")
        self.mean = self.original.mean(0)
        self.x = self.original-self.mean
        self.gram = self.x.T @ self.x
        self.ss = np.diag(self.gram)
        if np.linalg.matrix_rank(self.x) != 12:
            raise ValueError("Rank-deficient PC design")
        self.correlation = self.gram / np.sqrt(self.ss[:, None]*self.ss[None, :])
        self.factors = {k: np.linalg.qr(self.x[:, :k], mode="reduced") for k in COUNTS}

    def diagnostics(self) -> dict:
        off = self.correlation-np.eye(12)
        singular = np.linalg.svd(self.x, compute_uv=False)
        return {"means": self.mean.tolist(), "sample_variance": (self.ss/(len(self.x)-1)).tolist(),
                "correlation": self.correlation.tolist(), "gram": self.gram.tolist(),
                "maximum_offdiagonal_correlation": float(np.max(np.abs(off))),
                "orthogonal": bool(np.max(np.abs(off)) <= ORTHOGONAL_TOL),
                "rank": 12, "condition_number": float(singular[0]/singular[-1]),
                "singular_values": singular.tolist()}

    def joint_fits(self, y: np.ndarray):
        """Yield independently fitted nested joint models in original PC units.

        Shared QR factors also support seasonal subsets; no individual fits or
        predictor rescaling are needed. Intercepts apply to uncentred PCs.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or len(y) != len(self.x) or not np.isfinite(y).all():
            raise ValueError("Require the same finite T-by-P target")
        mean = y.mean(0)
        centred = y-mean
        for k in COUNTS:
            q, r = self.factors[k]
            coefficient = solve_triangular(r, q.T @ centred)
            yield k, coefficient, mean-self.mean[:k] @ coefficient

    def fit(self, y: np.ndarray) -> tuple[dict, dict, dict]:
        """Fit finite, deseasonalized T-by-P cells; reuse factors across tiles."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or len(y) != len(self.x) or not np.isfinite(y).all():
            raise ValueError("Both methods require the same finite T-by-P target")
        t = len(y)
        mean = y.mean(0)
        yc = y-mean
        sst = np.sum(yc*yc, axis=0)
        # Near-constant threshold is tied to original anomaly amplitude and precision.
        threshold = (100*np.finfo(float).eps*np.maximum(np.max(np.abs(y), axis=0), 1e-12))**2*t
        varying = sst > threshold
        cross = self.x.T @ yc
        bi = cross/self.ss[:, None]
        corr = np.full_like(bi, np.nan)
        np.divide(cross, np.sqrt(self.ss[:, None]*sst), out=corr, where=varying[None, :])
        individual = {"coefficient": bi, "intercept": mean-self.mean[:, None]*bi,
                      "correlation": corr, "r_squared": corr*corr}
        nested_r2, nested_rmse = [], []
        for k in COUNTS:
            q, r = self.factors[k]
            b = solve_triangular(r, q.T @ yc)
            fitted = self.x[:, :k] @ b
            residual = yc-fitted
            sse = np.sum(residual*residual, axis=0)
            r2 = np.full_like(sst, np.nan)
            np.divide(sse, sst, out=r2, where=varying)
            nested_r2.append(1-r2)
            nested_rmse.append(np.sqrt(sse/t))
        bj = b
        summed = self.x @ bi
        reconstruction_error = np.max(np.abs(fitted-summed), axis=0)
        # Identity remains valid for correlated predictors and diagnoses their effect.
        algebra_error = np.max(np.abs(bi-(self.gram @ bj)/self.ss[:, None]), axis=0)
        joint = {"coefficient": bj, "intercept": mean-self.mean @ bj,
                 "r_squared": nested_r2[-1], "rmse": nested_rmse[-1],
                 "residual_mean_square": sse/t, "residual_sample_variance": sse/(t-1),
                 "reconstruction_variance": np.sum(fitted*fitted, axis=0)/(t-1),
                 "target_variance": sst/(t-1), "target_mean": mean,
                 "nested_r_squared": np.stack(nested_r2), "nested_rmse": np.stack(nested_rmse),
                 "constant_target": (~varying).astype(float)}
        comparison = {"coefficient_difference": bj-bi,
                      "reconstruction_max_absolute_difference": reconstruction_error,
                      "r_squared_sum_difference": nested_r2[-1]-np.sum(corr*corr, axis=0),
                      "correlated_design_identity_error": algebra_error}
        return individual, joint, comparison


class TileWriter:
    """Compressed float64 NetCDF written incrementally, published on completion."""
    def __init__(self, path: Path, lat: np.ndarray, lon: np.ndarray, schema: dict,
                 attrs: dict, tile_shape: tuple[int, int],
                 extra_dimensions: dict[str, int] | None = None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_suffix(".partial.nc")
        if path.exists() or self.temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or its partial output")
        self.file = h5netcdf.File(self.temporary, "w")
        f = self.file
        f.dimensions = {"latitude": len(lat), "longitude": len(lon), "mode": 12,
                        "retained_modes": 4, "month": 12}
        for name, size in (extra_dimensions or {}).items():
            if name in f.dimensions:
                raise ValueError(f"Duplicate writer dimension {name}")
            f.dimensions[name] = size
        for name, dim, values, units in (("latitude", "latitude", lat, "degrees_north"),
                                        ("longitude", "longitude", lon, "degrees_east"),
                                        ("mode", "mode", np.arange(1, 13), "1"),
                                        ("retained_modes", "retained_modes", COUNTS, "1"),
                                        ("month", "month", np.arange(1, 13), "1")):
            v = f.create_variable(name, (dim,), dtype="f8" if name in ("latitude", "longitude") else "i4")
            v[:] = values
            v.attrs["units"] = units
        for name, (prefix, units) in schema.items():
            dims = tuple(prefix)+("latitude", "longitude")
            chunks = tuple(1 for _ in prefix)+(min(len(lat), tile_shape[0]), min(len(lon), tile_shape[1]))
            v = f.create_variable(name, dims, dtype="f8", fillvalue=np.nan,
                                  chunks=chunks, compression="gzip", compression_opts=4, shuffle=True)
            v.attrs["units"] = units
        f.attrs.update({k: v if isinstance(v, (str, int, float)) else json.dumps(v, default=str)
                        for k, v in attrs.items()})

    def write(self, name: str, a: np.ndarray, ys: slice, xs: slice):
        self.file.variables[name][(..., ys, xs)] = a

    def close(self, success: bool = False):
        self.file.close()
        if success:
            self.temporary.rename(self.path)


def expand(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[:-1]+valid.shape, np.nan)
    out[..., valid] = values
    return out


def schemas(units: str) -> tuple[dict, dict, dict]:
    slope_units = units+" per EOF-training PC SD"
    individual = {"coefficient": (("mode",), slope_units), "intercept": (("mode",), units),
                  "correlation": (("mode",), "1"), "r_squared": (("mode",), "1")}
    joint = {"coefficient": (("mode",), slope_units), "intercept": ((), units),
             "r_squared": ((), "1"), "rmse": ((), units), "residual_mean_square": ((), f"({units})2"),
             "residual_sample_variance": ((), f"({units})2"), "reconstruction_variance": ((), f"({units})2"),
             "target_variance": ((), f"({units})2"), "target_mean": ((), units),
             "nested_r_squared": (("retained_modes",), "1"), "nested_rmse": (("retained_modes",), units),
             "constant_target": ((), "1"), "valid_sample_count": ((), "1"), "finite_sample_count": ((), "1"),
             "ocean_mask": ((), "1"), "land_fraction": ((), "1"), "monthly_climatology": (("month",), units)}
    comparison = {"coefficient_difference": (("mode",), slope_units),
                  "reconstruction_max_absolute_difference": ((), units),
                  "r_squared_sum_difference": ((), "1"), "correlated_design_identity_error": ((), slope_units)}
    return individual, joint, comparison


def process_variable(kind: str, fields: list[xr.DataArray], mask: np.ndarray,
                     designs: dict[str, Design], out: Path, attrs: dict,
                     tile_shape: tuple[int, int] = (121, 240), max_tiles: int | None = None,
                     progress=print) -> dict:
    """One target at a time, both hemisphere designs per loaded tile."""
    ref = fields[0]
    lat, lon = ref.latitude.values, ref.longitude.values
    units = SPECS[kind][-1] if kind in SPECS else "W m-2"
    writers = {}
    completed = False
    import time
    started = time.monotonic()
    tile_count = 0
    valid_cells = incomplete_cells = 0
    raw_min, raw_max = np.inf, -np.inf
    try:
        for hemi in designs:
            directory = out/("northern_hemisphere" if hemi == "north" else "southern_hemisphere")
            for method, schema in zip(("individual", "joint", "comparison"), schemas(units)):
                writers[hemi, method] = TileWriter(directory/method/f"{kind}.nc", lat, lon, schema,
                    {**attrs, "hemisphere": hemi, "target": kind, "method": method,
                     "heat_positive": "downward", "stress_positive": "eastward/northward",
                     "pc_scaling": "original EOF-training sample SD; regression-sample centring only",
                     "missing_policy": "fixed common months; exclude temporally incomplete cells",
                     "sea_ice": "retained; these are surface fluxes at ocean locations, including ice",
                     "numerical_dtype": "float64", "pilot": int(max_tiles is not None)}, tile_shape)
        stop = False
        for j in range(0, len(lat), tile_shape[0]):
            for i in range(0, len(lon), tile_shape[1]):
                ys, xs = slice(j, min(j+tile_shape[0], len(lat))), slice(i, min(i+tile_shape[1], len(lon)))
                values = None
                finite_count = None
                for field in fields:
                    a = np.asarray(field.isel(latitude=ys, longitude=xs).transpose("time", "latitude", "longitude"), dtype=float)
                    a *= float(field.attrs["rate_factor"])
                    finite = np.isfinite(a)
                    if finite.any():
                        raw_min, raw_max = min(raw_min, float(a[finite].min())), max(raw_max, float(a[finite].max()))
                    finite_count = finite.astype(np.int16) if finite_count is None else finite_count*finite
                    values = a if values is None else values+a
                finite_count = finite_count.sum(axis=0)
                land = mask[ys, xs]
                ocean = land < .5
                valid = ocean & (finite_count == len(ref.time))
                valid_cells += int(valid.sum())
                incomplete_cells += int((ocean & ~valid).sum())
                monthly = ref.time.dt.month.values
                if valid.any():
                    anomaly, climatology = monthly_anomalies(values[:, valid], monthly)
                    for hemi, design in designs.items():
                        outputs = design.fit(anomaly)
                        for method, output in zip(("individual", "joint", "comparison"), outputs):
                            for name, v in output.items():
                                writers[hemi, method].write(name, expand(v, valid), ys, xs)
                        writers[hemi, "joint"].write("monthly_climatology", expand(climatology, valid), ys, xs)
                for hemi in designs:
                    w = writers[hemi, "joint"]
                    for name, v in {"ocean_mask": ocean.astype(float), "land_fraction": land,
                                    "finite_sample_count": finite_count,
                                    "valid_sample_count": np.where(valid, len(ref.time), 0)}.items():
                        w.write(name, v, ys, xs)
                tile_count += 1
                progress(f"{kind}: tile {tile_count}, rows {j}:{ys.stop}, columns {i}:{xs.stop}, valid={int(valid.sum())}")
                if max_tiles is not None and tile_count >= max_tiles:
                    stop = True
                    break
            if stop:
                break
        completed = True
    finally:
        for writer in writers.values():
            writer.close(completed)
    return {"tiles": tile_count, "valid_ocean_cells": valid_cells, "incomplete_ocean_cells": incomplete_cells,
            "source_component_minimum": float(raw_min), "source_component_maximum": float(raw_max),
            "elapsed_seconds": time.monotonic()-started}
