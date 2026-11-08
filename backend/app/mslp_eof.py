"""Exact covariance EOFs of monthly MSLP, streamed in native-grid latitude slabs.

Only the temporal Gram matrix is dense. No spatial covariance matrix, spatial
standardisation, interpolation, masking of land, or primary detrending is used.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

LOG = logging.getLogger(__name__)
DETRENDING_METHOD = "joint monthly climatology and linear trend"


def open_mslp(inputs: list[str], variable: str = "msl", block_latitudes: int = 16) -> tuple[xr.DataArray, dict]:
    """Open single/multiple files lazily; preserve the source grid and attributes.

    Caller must close the returned DataArray. expver merging is deferred to each
    slab, where overlapping finite values must agree (within 0.001 hPa).
    """
    paths = sorted({str(Path(p).resolve()) for pattern in inputs
                    for p in glob.glob(pattern)})
    if not paths:
        raise ValueError("No input NetCDF files matched")
    opened = []
    try:
        for path in paths:
            ds = xr.open_dataset(path, chunks=None, cache=False)
            rename = {}
            for target, aliases in {"time": ("valid_time",), "latitude": ("lat",),
                                    "longitude": ("lon",)}.items():
                if target not in ds.dims:
                    found = [a for a in aliases if a in ds.dims]
                    if len(found) == 1:
                        rename[found[0]] = target
            ds = ds.rename(rename)
            # Match computational slabs; never inherit enormous encoded chunks.
            ds = ds.chunk({"latitude": block_latitudes, "time": -1})
            opened.append(ds)
            if variable not in ds:
                raise ValueError(f"{path}: no {variable!r}; available: {list(ds.data_vars)}")
        arrays = [pressure_hpa(ds[variable]) for ds in opened]
        # join='exact' prevents silent regridding/alignment of spatial grids.
        field = (arrays[0] if len(arrays) == 1 else
                 xr.concat(arrays, dim="time", join="exact", coords="minimal",
                           compat="equals", combine_attrs="drop_conflicts"))
        required = {"time", "latitude", "longitude"}
        if not required.issubset(field.dims):
            raise ValueError(f"Expected time/latitude/longitude dimensions, got {field.dims}")
        extra = set(field.dims) - required - {"expver"}
        for dim in extra:
            if field.sizes[dim] != 1:
                raise ValueError(f"Unsupported nonsingleton dimension {dim}")
            field = field.isel({dim: 0}, drop=True)
        field = field.sortby("time").sortby("latitude").sortby("longitude")
        validate_months(field.time)
        validate_grid(field)
        metadata = {"files": [{"path": p, "bytes": Path(p).stat().st_size,
                                "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths],
                    "variable": variable, "source_dimensions": [dict(d.sizes) for d in opened],
                    "source_units": [str(d[variable].attrs.get("units")) for d in opened],
                    "source_attributes": [dict(d.attrs) for d in opened],
                    "first_month": month_strings(field.time)[0],
                    "last_month": month_strings(field.time)[-1],
                    "expver": np.unique(field.expver.values).tolist() if "expver" in field.coords else None,
                    "expver_dimensions": list(field.expver.dims) if "expver" in field.coords else [],
                    "grid_spacing_deg": float(np.diff(field.latitude.values)[0])}
        field.set_close(lambda: [d.close() for d in opened])
        return field, metadata
    except Exception:
        for ds in opened:
            ds.close()
        raise


def pressure_hpa(field: xr.DataArray) -> xr.DataArray:
    units = str(field.attrs.get("units", "")).strip().lower()
    if units in {"pa", "pascal", "pascals"}:
        result = field.astype(np.float64) / 100.0
    elif units in {"hpa", "hectopascal", "hectopascals", "mbar", "millibar"}:
        result = field.copy(deep=False)
    else:
        raise ValueError(f"Unsupported or absent MSLP units: {units!r}")
    result.attrs = {**field.attrs, "units": "hPa"}
    return result


def month_strings(time: xr.DataArray) -> list[str]:
    return [f"{int(y):04d}-{int(m):02d}" for y, m in
            zip(time.dt.year.values, time.dt.month.values)]


def validate_months(time: xr.DataArray) -> None:
    try:
        years, months = time.dt.year.values, time.dt.month.values
        ids = years.astype(int) * 12 + months.astype(int) - 1
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError("Time must contain decoded, finite calendar dates") from exc
    if len(ids) < 24 or not np.all(np.diff(ids) == 1):
        raise ValueError("Require at least 24 consecutive monthly samples; missing, repeated, "
                         "or irregular months are not filled or dropped")
    if set(months.tolist()) != set(range(1, 13)):
        raise ValueError("All twelve calendar months are required")


def validate_grid(field: xr.DataArray, resolution: float | None = None) -> None:
    for name in ("latitude", "longitude"):
        declared_units = str(field[name].attrs.get("units", "")).lower()
        accepted = ({"degrees_north", "degree_north", "degrees_n", "degree_n", "degrees"}
                    if name == "latitude" else
                    {"degrees_east", "degree_east", "degrees_e", "degree_e", "degrees"})
        if declared_units and declared_units not in accepted:
            raise ValueError(f"Unsupported {name} units: {declared_units!r}; require degrees")
        c = np.asarray(field[name].values, dtype=float)
        if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)):
            raise ValueError(f"Invalid {name} coordinate")
        delta = np.diff(c)
        if not np.all(delta > 0) or not np.allclose(delta, delta[0], atol=1e-7):
            raise ValueError(f"Require a regular, unique {name} grid")
        if resolution is not None and not np.allclose(delta, resolution, atol=1e-7):
            raise ValueError(f"Primary analysis requires native {resolution} degree {name}")
    lon = field.longitude.values
    if not np.isclose(lon[-1] - lon[0] + np.diff(lon)[0], 360):
        raise ValueError("Longitude must cover exactly one globe without a duplicate endpoint")
    if field.latitude.min() < -90 or field.latitude.max() > 90:
        raise ValueError("Latitude outside [-90, 90]")


def select_domain(field: xr.DataArray, hemisphere: str) -> xr.DataArray:
    if hemisphere not in {"north", "south"}:
        raise ValueError("Hemisphere must be north or south")
    bounds = (20, 90) if hemisphere == "north" else (-90, -20)
    return field.where((field.latitude >= bounds[0]) &
                       (field.latitude <= bounds[1]), drop=True).sortby("latitude")


def latitude_weights(latitude: np.ndarray) -> np.ndarray:
    cosine = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))
    # Exact pole has zero measure, rather than a floating point 6e-17 weight.
    cosine[np.abs(cosine) < 1e-14] = 0.0
    return np.sqrt(np.clip(cosine, 0, 1))


def monthly_anomalies(values: np.ndarray, months: np.ndarray,
                      detrend: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Independent period climatology; finite, float64 input required everywhere."""
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Missing/nonfinite MSLP: no imputation or changing spatial mask allowed")
    climatology = np.stack([x[months == m].mean(axis=0) for m in range(1, 13)])
    anomaly = x - climatology[months - 1]
    if detrend:
        # Residualise time against the same monthly indicators before fitting.
        # Sequential deseasonalising then ordinary detrending reintroduces a
        # seasonal cycle because raw time has different means in each month.
        t = np.arange(len(x), dtype=float)
        monthly_time = np.array([t[months == m].mean() for m in range(1, 13)])
        t -= monthly_time[months - 1]
        slope = np.tensordot(t, anomaly, axes=(0, 0)) / np.dot(t, t)
        anomaly -= t.reshape((-1,) + (1,) * (x.ndim - 1)) * slope
    return anomaly, climatology


def _slab(field: xr.DataArray, section: slice) -> np.ndarray:
    block = field.isel(latitude=section).load()
    if "expver" in block.dims:
        versions = np.asarray(block.transpose("expver", "time", "latitude", "longitude"),
                              dtype=np.float64)
        merged = versions[0].copy()
        for version in versions[1:]:
            overlap = np.isfinite(merged) & np.isfinite(version)
            if not np.allclose(merged[overlap], version[overlap], atol=0.001, rtol=0):
                raise ValueError("Conflicting overlapping expver values; select a product explicitly")
            merged = np.where(np.isfinite(merged), merged, version)
        return merged
    return np.asarray(block.transpose("time", "latitude", "longitude"), dtype=np.float64)


def autocorrelation(pc: np.ndarray, max_lag: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Biased sample ACF and first sampled 1/e crossing, in whole months.

    This is a descriptive monthly crossing, not an AR(1) fit or submonthly time.
    No crossing within the requested window is recorded as NaN.
    """
    x = np.asarray(pc) - np.mean(pc, axis=0)
    max_lag = min(max_lag, len(x) - 1)
    denominator = np.sum(x * x, axis=0)
    acf = np.stack([np.sum(x[:len(x)-lag] * x[lag:], axis=0) / denominator
                    for lag in range(max_lag + 1)])
    crossing = np.full(x.shape[1], np.nan)
    for k in range(x.shape[1]):
        found = np.flatnonzero(acf[1:, k] <= np.exp(-1))
        if len(found):
            crossing[k] = found[0] + 1
    return acf, crossing


def solve_eof(field: xr.DataArray, n_modes: int = 12, block_latitudes: int = 16,
              detrend: bool = False, max_lag: int = 24) -> xr.Dataset:
    """Two exact passes: sum Xw Xw.T, then recover EOFs and regressions.

    PC = U sqrt((T-1) lambda); weighted EOF = V.T; standardized PC = U sqrt(T-1).
    Regression = A.T standardized_PC/(T-1), in hPa per unit sample SD PC.
    Total variance is trace(Gram)/(T-1), including all omitted modes.
    """
    validate_months(field.time)
    if block_latitudes < 1 or max_lag < 24:
        raise ValueError("block_latitudes must be positive and max_lag at least 24")
    nt, ny, nx = (field.sizes[d] for d in ("time", "latitude", "longitude"))
    if not 1 <= n_modes < min(nt, ny * nx):
        raise ValueError("n_modes must be positive and below both sample and cell counts")
    months = field.time.dt.month.values
    weights = latitude_weights(field.latitude.values)
    slabs = [slice(i, min(i + block_latitudes, ny)) for i in range(0, ny, block_latitudes)]
    # Conservative live-array estimate; backend decompression/BLAS caches are additional.
    memory = 8 * (5 * nt * nt + 6 * nt * min(ny, block_latitudes) * nx
                  + (2 * n_modes + 12) * ny * nx)
    LOG.info("T=%d, cells=%d; estimated working arrays %.2f GiB (plus IO/BLAS caches)",
             nt, ny * nx, memory / 2**30)
    gram = np.zeros((nt, nt), dtype=np.float64)
    for i, section in enumerate(slabs):
        anomaly, _ = monthly_anomalies(_slab(field, section), months, detrend)
        x = (anomaly * weights[section][None, :, None]).reshape(nt, -1)
        gram += x @ x.T
        LOG.info("Gram pass %d/%d latitude slabs", i+1, len(slabs))
    total = float(np.trace(gram) / (nt-1))
    # One extra eigenvalue permits testing separation of the last retained mode.
    count = min(n_modes + 1, nt)
    eigen, u = eigh(gram / (nt-1), subset_by_index=(nt-count, nt-1), driver="evr")
    eigen, u = eigen[::-1], u[:, ::-1]
    if eigen[n_modes-1] <= max(total * 1e-12, 0):
        raise ValueError("Requested modes exceed numerical rank of the anomaly field")
    eigen, neighbour = eigen[:n_modes], eigen[1:n_modes+1]
    u = u[:, :n_modes]
    standardized = u * np.sqrt(nt-1)
    pc = standardized * np.sqrt(eigen)
    eof = np.empty((n_modes, ny, nx), dtype=float)
    regression = np.empty_like(eof)
    climatology = np.empty((12, ny, nx), dtype=float)
    for i, section in enumerate(slabs):
        anomaly, climatology[:, section, :] = monthly_anomalies(_slab(field, section), months, detrend)
        flat = anomaly.reshape(nt, -1)
        reg = (standardized.T @ flat / (nt-1)).reshape(n_modes, -1, nx)
        regression[:, section, :] = reg
        eof[:, section, :] = reg * weights[section][None, :, None] / np.sqrt(eigen)[:, None, None]
        LOG.info("Pattern pass %d/%d latitude slabs", i+1, len(slabs))
    # Reproducible sign convention: largest absolute regression loading is positive.
    signs = np.array([1 if r.flat[np.argmax(np.abs(r))] >= 0 else -1 for r in regression])
    eof *= signs[:, None, None]
    regression *= signs[:, None, None]
    pc *= signs
    standardized *= signs
    acf, crossing = autocorrelation(standardized, max_lag)
    # Heuristic positive-sequence integral correlation time, without an AR1 assumption.
    neff = np.empty(n_modes)
    for k in range(n_modes):
        positive = acf[1:, k]
        stop = np.flatnonzero(positive <= 0)
        positive = positive[:stop[0]] if len(stop) else positive
        neff[k] = np.clip(nt / (1 + 2 * positive.sum()), 2, nt)
    error = eigen * np.sqrt(2 / neff)
    # Neighbour's effective N is unknown at the boundary: use this mode's N,
    # flag this explicitly as a heuristic, and retain the extra eigenvalue.
    separated = eigen - error > neighbour * (1 + np.sqrt(2 / neff))
    result = xr.Dataset(
        data_vars={
            "weighted_eof": (("mode", "latitude", "longitude"), eof, {"units": "1"}),
            "regression_map": (("mode", "latitude", "longitude"), regression,
                               {"units": "hPa", "description": "hPa per one sample-SD PC anomaly"}),
            "monthly_climatology": (("month", "latitude", "longitude"), climatology, {"units": "hPa"}),
            "pc": (("time", "mode"), pc, {"units": "hPa"}),
            "standardized_pc": (("time", "mode"), standardized, {"units": "1", "ddof": 1}),
            "latitude_weight": ("latitude", weights),
            "eigenvalue": ("mode", eigen, {"units": "hPa2"}),
            "explained_variance_percent": ("mode", 100 * eigen / total),
            "cumulative_variance_percent": ("mode", 100 * np.cumsum(eigen) / total),
            "acf": (("lag", "mode"), acf),
            "lag_one_acf": ("mode", acf[1]),
            "first_efold_crossing_months": ("mode", crossing),
            "effective_sample_size_heuristic": ("mode", neff),
            "north_error_iid": ("mode", eigen * np.sqrt(2 / nt)),
            "north_error_adjusted_heuristic": ("mode", error),
            "next_eigenvalue": ("mode", neighbour),
            "separated_from_next_heuristic": ("mode", separated.astype("i1")),
        },
        coords={"time": field.time, "mode": np.arange(1, n_modes+1),
                "latitude": field.latitude, "longitude": field.longitude,
                "month": np.arange(1, 13), "lag": np.arange(len(acf))},
        attrs={"solver": "exact spatially streamed temporal Gram eigendecomposition",
               "first_month": month_strings(field.time)[0], "last_month": month_strings(field.time)[-1],
               "n_samples": nt, "detrended": int(detrend), "total_weighted_variance_hpa2": total,
               "detrending_method": DETRENDING_METHOD if detrend else "none",
               "estimated_working_array_bytes": memory,
               "anomalies": "independent monthly climatology for this analysis period",
               "weighting": "sqrt(cos(latitude)); covariance, no cell standardisation",
               "missing_policy": "fail on any missing cell or missing/repeated month",
               "expver_policy": "combine complementary versions; fail if finite overlap differs >0.001 hPa",
               "north_diagnostic": "heuristic; positive-sequence ACF truncated at configured lag; "
                                   "neighbour error uses current mode effective N; no significance claim"})
    return result


def spatial_correlation(a: np.ndarray, b: np.ndarray, latitude: np.ndarray) -> float:
    """Area-weighted, spatial-mean-centred Pearson correlation."""
    weights = np.broadcast_to(latitude_weights(latitude)[:, None] ** 2, a.shape)
    valid = np.isfinite(a) & np.isfinite(b) & (weights > 0)
    w, x, y = weights[valid], a[valid], b[valid]
    if not len(w):
        return float("nan")
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w*x*x) * np.sum(w*y*y))
    return float(np.sum(w*x*y) / denominator) if denominator > 0 else float("nan")


def compare_periods(full: xr.Dataset, modern: xr.Dataset) -> xr.Dataset:
    """Hungarian matching maximises absolute spatial correlations, then aligns signs.

    Bases remain unmodified. Comparison PCs use each period's own sample SD.
    Singular values of weighted EOF overlap measure retained-subspace stability.
    """
    if not (np.array_equal(full.latitude, modern.latitude) and
            np.array_equal(full.longitude, modern.longitude)):
        raise ValueError("Period comparison requires identical native grids")
    a, b = full.regression_map.values, modern.regression_map.values
    correlation = np.array([[spatial_correlation(x, y, full.latitude.values) for y in b] for x in a])
    if not np.all(np.isfinite(correlation)):
        raise ValueError("Undefined spatial correlations; cannot match constant patterns")
    rows, cols = linear_sum_assignment(-np.abs(correlation))
    signs = np.where(correlation[rows, cols] >= 0, 1, -1)
    full_pc, modern_pc = xr.align(full.standardized_pc, modern.standardized_pc, join="inner")
    temporal = np.array([np.corrcoef(full_pc.values[:, i], modern_pc.values[:, j]*s)[0, 1]
                         for i, j, s in zip(rows, cols, signs)])
    overlap = full.weighted_eof.values.reshape(len(a), -1) @ modern.weighted_eof.values.reshape(len(b), -1).T
    singular = np.linalg.svd(overlap, compute_uv=False)
    return xr.Dataset({
        "spatial_correlation_matrix": (("full_mode", "modern_mode"), correlation),
        "matched_modern_mode": ("match", cols+1), "full_mode_number": ("match", rows+1),
        "modern_sign": ("match", signs),
        "aligned_spatial_correlation": ("match", np.abs(correlation[rows, cols])),
        "common_pc_correlation": ("match", temporal),
        "full_variance_percent": ("match", full.explained_variance_percent.values[rows]),
        "modern_variance_percent": ("match", modern.explained_variance_percent.values[cols]),
        "variance_difference_percentage_points": ("match", modern.explained_variance_percent.values[cols]
                                                  - full.explained_variance_percent.values[rows]),
        "subspace_cosine": ("angle", np.clip(singular, 0, 1)),
        "principal_angle_degrees": ("angle", np.rad2deg(np.arccos(np.clip(singular, 0, 1)))),
    }, coords={"full_mode": full.mode.values, "modern_mode": modern.mode.values,
               "match": np.arange(1, len(rows)+1), "angle": np.arange(1, len(singular)+1)},
        attrs={"matching": "maximum total absolute area-weighted spatial Pearson correlation",
               "pc_caveat": "separate climatologies, EOF bases and sample-SD normalisations; "
                            "common-interval correlations are descriptive, not identical-mode projections",
               "common_first_month": month_strings(full_pc.time)[0],
               "common_last_month": month_strings(full_pc.time)[-1]})
