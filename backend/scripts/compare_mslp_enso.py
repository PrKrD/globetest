"""Exploratory ENSO associations of saved 1979–2026 MSLP PCs 2–12."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/mslp_enso"
SOURCES = {
    "MEIv2": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "Nino34": "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data",
}
FILES = {"MEIv2": "meiv2.data", "Nino34": "nino34.long.anom.data"}


def read_index(path: Path) -> pd.Series:
    values = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        if not 1850 <= year <= 2100:
            continue
        for month, token in enumerate(parts[1:], 1):
            value = float(token)
            key = pd.Period(year=year, month=month, freq="M")
            if key in values:
                raise ValueError("Duplicate index month")
            values[key] = value if np.isfinite(value) and abs(value) < 90 else np.nan
    result = pd.Series(values, dtype=float).sort_index()
    if result.empty or not np.all(np.diff(result.index.asi8) == 1):
        raise ValueError("Index must contain continuous monthly slots")
    return result


def pair(pc: pd.Series, index: pd.Series, lag: int = 0) -> pd.DataFrame:
    """Positive lag means index precedes PC; no wrapping or interpolation."""
    shifted = index.copy()
    shifted.index = shifted.index + lag
    return pd.concat({"pc": pc, "index": shifted}, axis=1).dropna()


def stats(paired: pd.DataFrame) -> dict:
    x, y = paired.pc.to_numpy(), paired["index"].to_numpy()
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("Too few varying paired values")
    r = float(np.corrcoef(x, y)[0, 1])
    return {"r": r, "r_squared": r*r, "n": len(x),
            "paired_linear_detrended_r": float(np.corrcoef(detrend(x), detrend(y))[0, 1])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    indices = {name: read_index(OUT / "reference_indices" / filename)
               for name, filename in FILES.items()}
    provenance = {name: {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                         "sha256": hashlib.sha256((OUT / "reference_indices" / FILES[name]).read_bytes()).hexdigest()}
                  for name, url in SOURCES.items()}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows, lags, seasons = [], [], []
    report = ["# ENSO associations of MSLP PCs 2–12", "",
              "Exploratory correlations of the saved 1979–2026 EOF bases, calculated separately "
              "for both hemispheres and baseline/linear-detrended runs. EOFs are not recomputed or "
              "rotated to maximise ENSO agreement. Original PC signs are preserved.", "",
              "MEI.v2 is NOAA's leading multivariate tropical-Pacific EOF PC. Its DJ, JF, …, ND "
              "values are assigned to the ending month (DJ to January) and compared with matching "
              "trailing two-month PC means. January 1979 is excluded because December 1978 PCs "
              "are unavailable. Niño 3.4 (HadISST) is a separate SST-only cross-check using unsmoothed "
              "monthly PCs. Each index uses the same available window across all four EOF analyses. "
              "Missing-value sentinels are excluded; no extrapolation is used.", "",
              "The primary ranking uses simultaneous all-month correlation with MEI.v2. The "
              "reported r² is shared variance of that PC at the stated averaging scale, not total "
              "hemispheric pressure variance or a causal attribution. Negative correlations indicate "
              "the opposite saved PC sign during El Niño; EOF signs are arbitrary.", ""]
    for hemisphere, folder in (("north", "northern_hemisphere"), ("south", "southern_hemisphere")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
        for treatment, run in (("baseline", "mslp_eof_12"), ("detrended", "mslp_eof_12_detrended")):
            path = ROOT / "diagnostics" / run / folder / "1979_2026/standardized_pcs.csv"
            pcs = pd.read_csv(path, index_col=0)
            pcs.index = pd.to_datetime(pcs.index).to_period("M")
            if not np.all(np.diff(pcs.index.asi8) == 1):
                raise ValueError("PCs have missing months")
            (OUT / f"{hemisphere}_{treatment}_pc_source.json").write_text(json.dumps({
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
            for mode in range(2, 13):
                pc = pcs[f"EOF_{mode}"]
                for name, index in indices.items():
                    target = pc.rolling(2, min_periods=2).mean() if name == "MEIv2" else pc
                    paired = pair(target, index)
                    key = {"hemisphere": hemisphere, "treatment": treatment, "mode": mode, "index": name}
                    rows.append({**key, "first": str(paired.index[0]), "last": str(paired.index[-1]), **stats(paired)})
                    paired.to_csv(OUT / f"{hemisphere}_{treatment}_mode{mode}_{name}_pairs.csv")
                    # Fixed window across lags; only the index month changes.
                    window = paired.index[6:]
                    for lag in range(7):
                        lagged = pair(target, index, lag).reindex(window).dropna()
                        lags.append({**key, "enso_leads_months": lag, **stats(lagged)})
                # Complete 3-month meteorological season means, winter labelled by Jan/Feb year.
                paired = pair(pc, indices["Nino34"])
                month = paired.index.month
                season = np.array(["DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5)
                                   else "JJA" if m in (6, 7, 8) else "SON" for m in month])
                year = paired.index.year + (month == 12)
                for label in ("DJF", "MAM", "JJA", "SON"):
                    grouped = paired.loc[season == label].groupby(year[season == label])
                    mean = grouped.mean().loc[grouped.size() == 3]
                    seasons.append({"hemisphere": hemisphere, "treatment": treatment, "mode": mode,
                                    "season": label, "index": "Nino34", **stats(mean)})
            selected = pd.DataFrame(rows)
            selected = selected[(selected.hemisphere == hemisphere) & (selected.treatment == treatment)]
            matrix = selected.pivot(index="mode", columns="index", values="r").reindex(columns=["MEIv2", "Nino34"])
            ax = axes[0 if treatment == "baseline" else 1]
            im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-.65, vmax=.65, aspect="auto")
            ax.set(xticks=[0, 1], xticklabels=["MEI.v2 / 2-month PC", "Niño 3.4 / monthly PC"],
                   yticks=np.arange(11), yticklabels=np.arange(2, 13), ylabel="PC mode", title=treatment.capitalize())
            for i in range(11):
                for j in range(2):
                    ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
            ranked = selected[selected["index"] == "MEIv2"].copy()
            ranked = ranked.loc[ranked.r.abs().sort_values(ascending=False).index]
            report += [f"## {hemisphere.capitalize()}: {treatment}", "",
                       f"MEI.v2: {ranked.iloc[0]['first']}–{ranked.iloc[0]['last']}, n={ranked.iloc[0]['n']} overlapping two-month periods.", "",
                       "| Mode (ranked by absolute MEI correlation) | MEI.v2 r | MEI.v2 r² (%) | Niño 3.4 r | MEI r after paired linear detrending |",
                       "|---:|---:|---:|---:|---:|"]
            for _, row in ranked.iterrows():
                nino = selected[(selected["index"] == "Nino34") & (selected["mode"] == row["mode"])].iloc[0]
                report.append(f"| {row['mode']} | {row['r']:+.3f} | {100*row['r_squared']:.1f} "
                              f"| {nino['r']:+.3f} | {row['paired_linear_detrended_r']:+.3f} |")
            report += [""]
        fig.suptitle(f"{hemisphere.capitalize()}: ENSO correlations of PCs 2–12")
        fig.colorbar(im, ax=list(axes), label="Pearson r", shrink=.8)
        fig.savefig(OUT / f"{hemisphere}_enso_correlations.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "correlations.csv", index=False)
    pd.DataFrame(lags).to_csv(OUT / "lag_correlations.csv", index=False)
    seasonal = pd.DataFrame(seasons)
    seasonal.to_csv(OUT / "seasonal_correlations.csv", index=False)
    report += ["## Seasonal structure", "",
               "Complete seasonal means reveal associations that can be diluted in the all-month "
               "correlations. Baseline Niño 3.4 correlations for selected candidates are below. "
               "DJF is December–February and SON is September–November; seasons are named by "
               "calendar months, not hemisphere-specific season names.", "",
               "| Hemisphere | PC | DJF r | MAM r | JJA r | SON r |",
               "|---|---:|---:|---:|---:|---:|"]
    for hemisphere, modes in (("north", [2, 8, 11]), ("south", [2, 3, 4, 7])):
        for mode in modes:
            selected = seasonal[(seasonal.hemisphere == hemisphere) &
                                (seasonal.treatment == "baseline") & (seasonal["mode"] == mode)].set_index("season")
            report.append(f"| {hemisphere} | {mode} | " + " | ".join(
                f"{selected.loc[season, 'r']:+.3f}" for season in ("DJF", "MAM", "JJA", "SON")) + " |")
    report += ["", "Northern PC2 is the strongest DJF candidate, with additional DJF associations "
               "in PCs 8 and 11. Southern PC2 is strongest in JJA and SON. These seasonal means "
               "are more heavily averaged than the monthly comparison and have fewer samples, "
               "so their correlation sizes should not be compared as if they were the same test.", "",
               "## Secondary checks and limits", "",
               "[Lag correlations](lag_correlations.csv) examine ENSO leading PCs by 0–6 months on "
               "a fixed common response window for each index. [Seasonal correlations](seasonal_correlations.csv) "
               "use complete seasonal means of monthly Niño 3.4 and PCs, with December assigned to the "
               "following DJF year. They have roughly 47 seasonal samples, not hundreds of independent months.", "",
               "These are exploratory associations across many modes, seasons and lags. No significance "
               "claims or multiple-comparison-adjusted p-values are made. ENSO persistence and overlapping "
               "two-month means reduce independent information. MEI includes SLP and shares the 20–30° "
               "latitude band with these EOF domains; it is not a fully independent remote-teleconnection "
               "validation. Niño 3.4 provides an SST-only cross-check. A pattern can contain ENSO-related "
               "variability without being a unique ENSO mode or entirely forced by ENSO.", "",
               "## Sources", "",
               "- [NOAA MEI.v2 methods and data](https://psl.noaa.gov/enso/mei/)",
               "- [NOAA Niño 3.4 HadISST methods and data](https://psl.noaa.gov/data/timeseries/month/Nino34/)",
               "- Download hashes and timestamps: [provenance](provenance.json)"]
    (OUT / "enso_comparison.md").write_text("\n".join(report) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
