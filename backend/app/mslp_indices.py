"""Offline comparison of MSLP PCs with provenance-checked AO/SAM indices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import detrend
from scipy.stats import spearmanr
import xarray as xr

HEMISPHERES = {"AO": "north", "SAM": "south", "AAO": "south"}


def read_monthly_index(text: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read CPC year/month/value or BAS year/Jan/.../Dec, including partial years."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit() or not 1900 <= int(parts[0]) <= 2100:
            continue
        year = int(parts[0])
        if name in {"AO", "AAO"}:
            if len(parts) != 3:
                raise ValueError(f"Invalid {name} year/month/value row")
            months = [(int(parts[1]), float(parts[2]))]
        else:
            if not 2 <= len(parts) <= 13:
                raise ValueError(f"Invalid {name} January–December row")
            months = enumerate(map(float, parts[1:]), start=1)
        for month, value in months:
            valid_value = (0 < value < 2000 if name in {"P40", "P65"} else abs(value) < 100)
            if name not in {"P40", "P65"} and np.any(
                    np.isclose(abs(value), [99.9, 99.99], rtol=0, atol=1e-8)):
                valid_value = False
            if not 1 <= month <= 12 or not np.isfinite(value) or not valid_value:
                raise ValueError(f"Invalid or missing {name} value in {year}-{month}")
            key = year * 12 + month - 1
            if key in values:
                raise ValueError(f"Repeated {name} month")
            values[key] = value
    ids = np.array(sorted(values), dtype=int)
    if len(ids) < 24 or not np.all(np.diff(ids) == 1):
        raise ValueError(f"Require continuous monthly {name} data with at least 24 samples")
    return ids, np.array([values[key] for key in ids], dtype=float)


def _month(key: int) -> str:
    year, month = divmod(int(key), 12)
    return f"{year:04d}-{month+1:02d}"


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("Correlation requires at least three varying paired samples")
    return float(np.corrcoef(a, b)[0, 1])


def _pc1_multiplier(ds: xr.Dataset) -> int:
    """Choose physical sign from pressure contrast, without maximising correlation."""
    lat = np.abs(ds.latitude.values)
    weights = np.clip(np.cos(np.deg2rad(lat)), 0, 1)
    zonal = ds.regression_map.values[0].mean(axis=1)
    bands = []
    for low, high in ((65, 90), (35, 55)):
        selected = (lat >= low) & (lat <= high) & (weights > 1e-14)
        if not np.any(selected):
            raise ValueError("Cannot orient PC1 without polar and midlatitude bands")
        bands.append(float(np.average(zonal[selected], weights=weights[selected])))
    contrast = bands[0] - bands[1]
    if not np.isfinite(contrast) or abs(contrast) < 1e-12:
        raise ValueError("Cannot orient PC1: no polar–midlatitude pressure contrast")
    return -1 if contrast > 0 else 1


def align_pc1_convention(ds: xr.Dataset) -> xr.Dataset:
    """Orient saved PC1 like positive AO/SAM, preserving the reconstructed field.

    This is a prescribed pressure-contrast convention, not an index fit or an
    identification of higher modes. Reapplying it to saved results is idempotent.
    """
    multiplier = _pc1_multiplier(ds)
    # A shallow copy keeps the large climatology and unsigned diagnostics shared;
    # replace signed arrays rather than changing the caller's underlying data.
    result = ds.copy(deep=False)
    signs = xr.DataArray([multiplier] + [1] * (ds.sizes["mode"] - 1),
                         dims="mode", coords={"mode": ds.mode})
    for name in ("pc", "standardized_pc", "weighted_eof", "regression_map"):
        result[name] = ds[name] * signs
        result[name].attrs = ds[name].attrs.copy()
    result.attrs["pc1_sign_convention"] = (
        "positive AO/SAM: area-weighted 65–90 degree cap minus 35–55 degree "
        "midlatitude regression loading is negative; no reference-index fitting")
    result.attrs["pc1_sign_multiplier_from_initial"] = (
        int(ds.attrs.get("pc1_sign_multiplier_from_initial", 1)) * multiplier)
    result.attrs["higher_mode_sign_convention"] = "largest absolute regression loading positive"
    return result


def _paired_summary(ids: np.ndarray, pc: np.ndarray, index: np.ndarray, multiplier: int) -> dict:
    x = multiplier * pc[:, 0]
    r = _correlation(x, index)
    return {"start": _month(ids[0]), "end": _month(ids[-1]), "n": len(ids),
            "raw_pc1_r": _correlation(pc[:, 0], index), "aligned_pc1_r": r,
            "r_squared": r*r, "detrended_r": _correlation(detrend(x), detrend(index)),
            "spearman": float(spearmanr(x, index).statistic),
            "all_mode_raw_r": [_correlation(pc[:, k], index) for k in range(pc.shape[1])]}


def compare_indices(results: dict, directory: Path) -> dict:
    """Recalculate all comparisons from local official tables, never cached correlations.

    provenance.json maps source names to URL, retrieval time and SHA256. No
    download, smoothing, gap filling, EOF refitting or saved sign change occurs.
    Primary correlations share a 1979-or-later window across the supplied bases.
    """
    provenance = json.loads((directory / "provenance.json").read_text())
    sources = {}
    for name, source in provenance.items():
        if name not in {*HEMISPHERES, "P40", "P65"}:
            raise ValueError(f"Unknown reference index {name!r}")
        raw = (directory / f"{name}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"{name} checksum mismatch; verify reference-data provenance")
        if not source.get("url") or not source.get("retrieved_utc"):
            raise ValueError(f"Missing {name} provenance")
        sources[name] = read_monthly_index(raw.decode("utf-8"), name)
    if not any(name in sources for name in HEMISPHERES):
        raise ValueError("No AO, SAM or AAO reference data")
    rows = []
    for name, hemisphere in HEMISPHERES.items():
        if name not in sources:
            continue
        reference_ids, reference = sources[name]
        bases = [(year, ds) for (h, year), ds in sorted(results.items()) if h == hemisphere]
        common_start = max([1979*12, int(reference_ids[0])] +
                           [int(ds.time.dt.year.values[0])*12 + int(ds.time.dt.month.values[0])-1
                            for _, ds in bases])
        common_end = min([int(reference_ids[-1])] +
                         [int(ds.time.dt.year.values[-1])*12 + int(ds.time.dt.month.values[-1])-1
                          for _, ds in bases])
        for year, ds in bases:
            pc_ids = ds.time.dt.year.values.astype(int)*12 + ds.time.dt.month.values.astype(int)-1
            ids, pi, ri = np.intersect1d(pc_ids, reference_ids, return_indices=True)
            mask = (ids >= common_start) & (ids <= common_end)
            if mask.sum() < 24:
                raise ValueError(f"Insufficient common monthly {name} overlap")
            pc, target = ds.standardized_pc.values[pi], reference[ri]
            multiplier = _pc1_multiplier(ds)
            row = {"index": name, "hemisphere": hemisphere, "basis_start": year,
                   "pc1_multiplier": multiplier,
                   **_paired_summary(ids[mask], pc[mask], target[mask], multiplier)}
            extended = _paired_summary(ids, pc, target, multiplier)
            row["extended_full_overlap"] = {k: extended[k] for k in ("start", "end", "n", "aligned_pc1_r")}
            early = ids < 1979*12
            if early.sum() >= 24:
                early_summary = _paired_summary(ids[early], pc[early], target[early], multiplier)
                row["pre_1979"] = {k: early_summary[k] for k in ("start", "end", "n", "aligned_pc1_r")}
            months = ids[mask] % 12 + 1
            row["seasonal_monthly_r"] = {}
            for season, selected_months in (("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]),
                                             ("JJA", [6, 7, 8]), ("SON", [9, 10, 11])):
                selected = np.isin(months, selected_months)
                if selected.sum() >= 3:
                    row["seasonal_monthly_r"][season] = _correlation(
                        multiplier*pc[mask][selected, 0], target[mask][selected])
            historical = mask & (ids < 2025*12)
            if historical.sum() >= 24:
                row["through_2024_r"] = _correlation(multiplier*pc[historical, 0], target[historical])
            rows.append(row)
    if not rows:
        raise ValueError("No reference indices correspond to the requested hemispheres")
    summary = {"provenance": provenance, "results": rows,
               "common_start_year": 1979,
               "alignment": "positive index = lower polar pressure relative to midlatitudes; "
                            "PC1 multiplier chosen from regression-map pressure contrast",
               "method": "calendar-month intersection; unsmoothed monthly Pearson correlations; "
                         "separate linear detrending of paired series as sensitivity; no EOF refitting"}
    if {"P40", "P65", "SAM"}.issubset(sources):
        reconstructed = []
        common = sources["SAM"][0]
        for name in ("P40", "P65"):
            ids, values = sources[name]
            baseline = (ids >= 1971*12) & (ids < 2001*12)
            normalized = np.empty_like(values)
            for month in range(12):
                selected = ids % 12 == month
                base = values[selected & baseline]
                if len(base) != 30 or np.std(base) == 0:
                    raise ValueError("SAM sign check requires complete varying 1971–2000 station pressures")
                normalized[selected] = (values[selected] - base.mean()) / base.std(ddof=0)
            common = np.intersect1d(common, ids)
            reconstructed.append((ids, normalized))
        p40 = reconstructed[0][1][np.searchsorted(reconstructed[0][0], common)]
        p65 = reconstructed[1][1][np.searchsorted(reconstructed[1][0], common)]
        observed = sources["SAM"][1][np.searchsorted(sources["SAM"][0], common)]
        summary["sam_formula_correlation"] = _correlation(p40-p65, observed)
    return summary
