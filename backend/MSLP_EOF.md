# Native-grid ERA5 monthly MSLP EOF analysis

This independent backend workflow analyses **20–90°N and 90–20°S separately**,
over all longitudes, including land. Primary defaults are January 1940 and
January 1979 through the same latest month no later than December 2026. It
retains all calendar months and the original **0.25° grid**, without regridding,
spatial smoothing, cell standardisation or detrending. It does not estimate
stochastic surface heat-flux forcing.

## Installation and execution

From `backend`, using the project's Python environment:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/analyse_mslp_eof.py \
  --input data/2374cdda86a09cf96f159388ce2d78e1.nc \
  --start-years 1940 1979 --hemispheres north south --n-modes 12 \
  --output-dir diagnostics/mslp_eof
python3 -m pytest -q
```

The entry point works without a `PYTHONPATH` override. Supply any local path,
multiple paths, or a quoted glob: `--input '/external/era5/mslp_*.nc'`.
Dependencies are in [requirements.txt](requirements.txt): NumPy, SciPy, xarray,
Dask, h5netcdf/h5py, Matplotlib and Cartopy. HDF5 NetCDF4 uses h5netcdf/h5py;
NetCDF3 can use xarray's SciPy backend. No NetCDF C compiler is required.

Inspect metadata first with `--inspect-only`. The local input inspected on
6 November 2026 contains 1,042 monthly samples, January 1940–October 2026,
721 latitudes and 1,440 longitudes, pressure in Pa, and an `expver(time)` string
coordinate. This labels each month; it is not a dimension to average.

The loader accepts `time`/`valid_time`, `latitude`/`lat`, `longitude`/`lon`, either
latitude ordering and either usual longitude convention. Coordinates must form
the same regular global grid in every file; duplicate cyclic endpoints are
rejected. Each hemisphere must include the full boundary range. `--variable`
overrides `msl`. Units must explicitly identify Pa or hPa. If an actual `expver`
dimension exists, complementary finite values are merged per slab. Overlapping
finite values must agree within 0.001 hPa. Other nonsingleton dimensions,
missing cells, missing/duplicate months and irregular sampling cause an error;
none are silently interpolated or removed.

## Method and numerical scaling

Each period subtracts **its own** twelve monthly climatologies in hPa. Anomalies
are multiplied by `sqrt(cos(latitude))`, with zero weight at the exact pole.
The solver uses an exact temporal Gram matrix, float64 covariance accumulation,
and sample eigenvalues with denominator `T−1`. Percentages use **total** weighted
variance, including modes outside the retained set. For unweighted anomalies
`A` and weighted anomalies `X = A W = U S Vᵀ`:

```text
weighted_eof = Vᵀ
pc = U S
standardized_pc = U sqrt(T−1)
regression_map = standardized_pcᵀ A / (T−1)
A_retained = standardized_pc @ regression_map
X_retained = pc @ weighted_eof
```

The physical regression maps have units hPa per one sample-standard-deviation
PC. These differ from the dimensionless orthonormal weighted EOF vectors. They
remain defined at the pole without dividing by zero. Each initial EOF sign
makes the largest absolute regression loading positive. Before export, the
workflow enforces the AO/SAM convention on **PC1 only**: positive PC1 has lower
area-weighted 65–90° polar-cap pressure relative to the 35–55° midlatitude band.
PC1, standardised PC1, weighted EOF1 and regression map 1 are flipped together,
preserving reconstruction and all variance/persistence statistics. An absent or
numerically zero pressure contrast stops the run. This prescribed rule does not
fit reference correlations or identify higher modes as AO/SAM. PCs 2–12 keep the
largest-loading-positive convention. The applied PC1 multiplier is saved in
NetCDF attributes and the summary JSON; rerendering is idempotent.

The default is **12 modes**. Retaining more modes increases cumulative captured
variance without changing leading eigenvalues or their percentages of total
variance. Existing six-mode outputs must be recomputed, not rendered with the
new default. Pass `--n-modes 6` explicitly to render a previous six-mode basis;
the PC1 sign guardrail will still apply.

Primary analysis keeps
trends; use `--detrend` with a **separate output directory** for sensitivity.
This fits monthly indicators and a linear trend jointly, keeping residual
calendar-month means zero even with a partial final year. Previous detrended
caches using sequential seasonal removal and detrending must be recomputed;
their missing method attribute prevents reuse.
Independent climatologies mean that period comparisons measure variability
about different seasonal baselines, not mean-pressure change.

Only the `T × T` covariance and a spatial slab are dense. Two passes accumulate
covariance and recover patterns; no spatial covariance or full anomaly cube is
held in RAM. `--block-latitudes 16` is the default. Smaller slabs lower memory
but increase compressed-file reads. For 1,042 months, 281 × 1,440 cells and twelve
modes, the live-array estimate is **1.22 GiB**, plus IO, Dask and BLAS caches and
completed outputs. A full float64 hemisphere would be about 3.1 GiB; a spatial
covariance exceeds 1 TiB. Actual chunking/compression affects peak memory and
runtime. Logs show memory estimates and progress through both passes. The
solver is exact and uses no random seed. Tests compare it against dense SVD
and known low-rank modes.

On the inspected 16 GiB workstation, `--block-latitudes 64` substantially reduced
repeated decompression of this file's wide encoded spatial chunks. Its working
array estimates with twelve modes are 4.44 GiB (full record) and 2.49 GiB (modern period), plus
caches. Use this setting when memory permits; the conservative default remains
16 rows. Both settings preserve the identical covariance definition and grid.

## Outputs and interpretation

Under `diagnostics/mslp_eof/` (ignored by Git):

- `northern_hemisphere/{1940,1979}_2026/` and
  `southern_hemisphere/{1940,1979}_2026/`: `eof.nc`, variance/persistence and
  standardised-PC CSVs, polar regression maps, PCs with optional 12-month means,
  scree/cumulative-variance plots and ACF plots.
- `comparisons/`: correlation matrices, Hungarian assignment and sign alignment,
  variance and common-PC statistics, paired maps with common colour limits,
  variance/PC comparison figures, and retained-subspace principal angles.
- `reports/mslp_eof_report.md` and `reports/metadata.json`: computed results,
  methods, cautions, actual ending month, input provenance, package versions,
  settings and command.

Directory names use the actual ending **year**; metadata and titles also record
the ending **month**. NetCDF stores climatologies, native coordinates, weighted
EOFs, unweighted regressions, raw/standardised PCs, eigenvalues, variance, ACF
and uncertainty diagnostics. Source `expver(time)` labels survive when present.
New local NetCDF inputs under `data/` are ignored; existing tracked ocean
products remain tracked.

Matching maximises total absolute **area-weighted, spatial-mean-centred Pearson
correlation** among retained regression maps. Comparison maps and PCs use the
same sign flips; saved bases retain their exported signs (AO/SAM convention for
PC1, largest-loading-positive for higher modes). PC comparisons
use each basis's own climatology and full-period sample SD. These are descriptive
common-interval correlations, not controlled projections onto a single basis.
Modes outside the retained set cannot be matched. Small subspace principal
angles with weaker individual correlations can indicate rotations among
degenerate EOFs. Nested-period comparisons do not establish significance.

North et al. (1982) errors use both IID sample counts and a **heuristic** effective
sample size from the positive ACF sequence, stopping at the first nonpositive
lag or requested maximum (default 24 months). The adjacent-mode overlap flag
uses the current mode's effective N for its neighbour too. It is not a formal
significance test. Interpret alongside the spectrum, ACF and subspace angles.
The first 1/e ACF crossing is sampled in whole months, not an AR1 fit; monthly
means cannot establish a ten-day atmospheric decorrelation time. Mode identities
require map inspection and independent indices, following Thompson and Wallace
(2000); rank alone does not identify AO, NAO or SAM.

## Options and validation

Use `--help` for all options. Additional controls are `--max-lag` (at least 24),
`--end-year`, `--dpi` (300 by default), `--no-running-mean`, `--no-coastlines`
and `--no-maps`. Cartopy may download Natural Earth 110m coastlines on first use;
`--no-coastlines` avoids that download while retaining polar maps and graticules.
More than two start years are allowed for individual analyses; automatic
matching is generated when exactly two are requested. `--allow-other-resolution`
is an explicit synthetic-validation option; it marks outputs as non-primary
and preserves the supplied grid.

After an analysis, repeat its command with `--render-only` to regenerate
figures, tables, comparisons and the report from the saved EOF NetCDFs. Input
file fingerprints, coordinate arrays, mode count, detrending and ACF lag count
must match. Metadata preserves the original analysis command and records the
render invocation. No covariance calculation is repeated.
Use `--reuse-existing` to resume a partial run: compatible saved EOFs are reused,
missing analyses are computed, and all diagnostics are regenerated. Mismatched
saved settings cause an error rather than silent reuse. Latitude slab size may
change on resume because it changes performance, not the covariance definition;
per-analysis settings and commands are recorded in metadata.

When a run includes manually inspected physical interpretation, its notes and
scope JSON live in `reports/`. The scope binds the notes to input fingerprints,
periods, detrending, eigenvalues and hashes of the computed regression maps.
Reports include these notes only when that scope matches, so a new input or
changed EOF basis cannot silently inherit the earlier interpretation.

## AO/SAM validation and retained summaries

The end of [analyse_mslp_eof.py](scripts/analyse_mslp_eof.py) contains the two
calculated Markdown summaries: EOF results/period stability, and established
AO/SAM index validation with sign conventions. They are appended to the report
on every run, and their numerical inputs are saved in `reports/summary.json`
and printed to stdout, following the subduction diagnostic's report/JSON pattern.
No historical result values are hard-coded into future analyses.

Index validation runs offline when `OUTPUT/reference_indices/` exists, or when
`--index-data-dir PATH` is supplied. For example, regenerate existing products:

```bash
python3 scripts/analyse_mslp_eof.py \
  --input data/2374cdda86a09cf96f159388ce2d78e1.nc \
  --output-dir diagnostics/mslp_eof --n-modes 6 --render-only \
  --index-data-dir diagnostics/mslp_eof/reference_indices
```

This example explicitly selects the previous six-mode basis. A twelve-mode run
needs recomputation first; rerender it with `--n-modes 12` (the new default).

The reference directory contains official text tables named `AO.txt`, `AAO.txt`
and/or `SAM.txt`. Optional `P40.txt` and `P65.txt` allow direct reconstruction of
Marshall SAM's sign from station-band pressures. `provenance.json` maps each
source name to its download URL, UTC retrieval time and exact SHA256, e.g.:

```json
{"AO": {"url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        "retrieved_utc": "<actual download time>", "sha256": "<SHA256 of AO.txt>"}}
```

The calculation verifies these checksums and rejects missing/duplicate reference
months. It never downloads files automatically or uses stored correlations in
place of recalculation. The local reference snapshot retains the November 2026
provider data and original retrieval times. Official sources are
[CPC AO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml),
[CPC AAO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/aao.shtml)
and [Marshall SAM](https://legacy.bas.ac.uk/met/gjma/sam.html). To update them,
replace the tables and record their new retrieval times and hashes together.
Reference data remain ignored by Git with the other diagnostics.

Primary correlations use unsmoothed, calendar-matched monthly values on a common
1979-or-later interval for each index across the supplied EOF bases. Sign
alignment is inferred from the polar–midlatitude regression loading, rather
than maximising correlation. Saved EOF1, regression map 1 and PC1 already follow
that convention, so the validation multiplier should be +1. Higher modes keep
their original signs. The JSON also retains raw higher-mode correlations, Spearman
correlations, paired-series detrending sensitivity, monthly correlations within
seasons, pre-1979 checks and comparisons ending in 2024. These comparisons support
mode identification; they do not establish causation or independent forecasting
skill. With no reference directory, the report explicitly records that the
index comparison was not calculated.

[test_mslp_eof.py](tests/test_mslp_eof.py) covers both hemisphere boundaries,
latitude order, longitude conventions, anomalies, pressure conversion, area
weights, exact dense-reference agreement, known modes, variance, PC scaling,
reconstruction, physical regressions, sign/rank matching, weighted correlations,
missing/incomplete months, multifile loading, expver, ACF, grid validation and
the four-analysis CLI workflow. Existing ocean/backend tests run with it.
