# MSLP principal-component surface regressions

This workflow regresses global native-grid ERA5 turbulent heat flux and stress
anomalies on existing **undetrended, monthly, standardized 12-mode 1979 EOF
bases**, separately for NH and SH. It does not refit EOFs or estimate complete
stochastic forcing F′. Radiation and SST feedback fitting are excluded.

## Annual and seasonal sensitivity experiments

Run the two approved experiments through the existing entry point, from `backend`:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
../.venv/bin/python scripts/analyse_mslp_surface_regression.py \
  --sensitivity --input data/3fa99ca792816ec2a9b65bf822235b4a.nc \
  --mask data/bea33c89038e10d15ddb0fa02b351c62.nc
```

The default sensitivity destination is `diagnostics/mslp_surface_regression_sensitivity/`;
the baseline directory remains read-only. Use `--baseline-dir` to select another
completed baseline. `--inspect-only`, `--max-tiles 1`, `--no-maps`, `--no-coastlines`,
`--dpi`, `--tile-latitudes` and `--tile-longitudes` retain their usual meanings.
Pilots require a separate destination and cannot be rendered as global results.
`--render-only --sensitivity` regenerates the report/figures from saved products.

The annual experiment reuses saved joint coefficients and the intercept, truncating
to 1/3/6/12 modes without refitting. Equal-month calendar means exclude incomplete
years. The original 574-month climatologies are retained even for the matched
564-month/47-year 1979–2025 evaluation. The seasonal experiment fits separate
joint QR regressions for DJF/MAM/JJA/SON and each nested mode count, using monthly
observations and original PC standardization. Its annual comparison averages
month-specific predictions, not seasonal coefficients or seasonal R² values.

Outputs include all seasonal coefficients/intercepts, R²/RMSE maps for every mode
count, fuller monthly diagnostics at 12 PCs, annual diagnostics for every count,
all-region/all-mask CSVs, regional series, PC conditioning/correlation statistics,
selected pattern statistics, PNG figures and a calculated scientific report.
No full-resolution monthly cubes are saved. Undefined constant-target R² is
excluded from R² summaries; finite RMSE and regional means retain constant cells,
matching the baseline. Negative prediction R² is preserved. Model differences
use seasonal minus year-round, with R² expressed in percentage points.

For downstream use, load `coefficient_12(season, pc_12, latitude, longitude)`
and `intercept_12(season, latitude, longitude)` from each target's coefficient
file under `seasonal_regressions/north/` or `seasonal_regressions/south/`.
The mode labels are `pc_12=1..12`; the separate `mode` coordinate is unused by
these coefficient arrays. Seasons are integers `0=DJF`, `1=MAM`, `2=JJA`,
`3=SON`, also documented in the coordinate's `flag_meanings` attribute.
Maps use descending latitude 90 to −90 and ascending longitude 0 to 359.75°E,
both at 0.25° spacing. Coefficients retain the original PC SD units; intercepts
have physical field units and are added once per reconstruction. Rename the
EOF `standardized_pc` dimension from `mode` to `pc_12` before xarray multiplication
to avoid broadcasting across independent mode axes. Annual seasonal-model
evaluation averages monthly reconstructions; there is no separately fitted
annual coefficient set or saved annual-PC series.

Source fingerprints and PC hashes must match the baseline. Before interpretation,
reconstructed monthly maps and regional metrics must reproduce saved diagnostics.
Seasonal turbulent coefficients/intercepts must equal latent plus sensible within
scale-aware float64 tolerances. All output creation refuses existing destinations.
Seasonal in-sample gains reflect a larger model (52 versus 13 coefficients for
12 PCs), not validated predictive skill. No EOF recomputation, annual regression,
cross-validation, extra predictors or frontend changes are involved.

## Run

From `backend`, using the existing project environment:

```sh
python scripts/analyse_mslp_surface_regression.py \
  --input data/3fa99ca792816ec2a9b65bf822235b4a.nc \
  --mask data/bea33c89038e10d15ddb0fa02b351c62.nc \
  --turbulent --output-dir diagnostics/mslp_surface_regression
```

The default PC inputs are the modern-period files under
`diagnostics/mslp_eof_12/{northern,southern}_hemisphere/1979_2026/eof.nc`.
Override with `--nh-pcs` and `--sh-pcs`. An output directory must be empty;
existing products are never overwritten by a new analysis. Interrupted files
retain `.partial.nc` names. Use another output directory for a new run.

`--input` accepts multiple files or quoted globs. Each file can contain one or
more targets, and files can split the time axis. Duplicate months or ambiguous
variables fail explicitly. Use `--map-variable latent=NAME` (and analogous
`sensible`, `eastward_stress`, `northward_stress` mappings) to disambiguate.
Identity must still be established by GRIB ID or a recognized variable name.

`--inspect-only` validates source metadata, months, grids and PCs without writing
outputs or loading full target fields. `--max-tiles 1 --no-maps` runs a pilot,
which must use a separate directory; pilot results are clearly labelled and
cannot be rendered as a global analysis. `--tile-latitudes` and
`--tile-longitudes` control memory and IO (defaults 121 and 240, matching the
downloaded spatial chunks). `--no-coastlines` avoids coastline downloads.

To regenerate diagnostics and figures from completed numerical outputs:

```sh
python scripts/analyse_mslp_surface_regression.py \
  --output-dir diagnostics/mslp_surface_regression --render-only
```

## Variables, time and signs

| Target | Recognized mean-rate names | GRIB ID | Units |
|---|---|---:|---|
| Latent heat flux | `avg_slhtf`, `mslhf` | 235034 | W m⁻² |
| Sensible heat flux | `avg_ishf`, `msshf` | 235033 | W m⁻² |
| Eastward stress | `avg_iews`, `metss` | 235041 | N m⁻² |
| Northward stress | `avg_inss`, `mntss` | 235042 | N m⁻² |

The `avg_*` stresses are mean-rate parameters, not instantaneous stress IDs
229/230. Do not identify them by a substring alone. The loader checks IDs and
units. Accumulated alternatives `slhf`, `sshf`, `ewss`, `nsss` are accepted only
with compatible units and verified `moda`/`edmo` provenance, and divided by
86400 seconds. Rates are never divided by time. Other accumulation intervals
are rejected for explicit review.

Dates are aligned by unique calendar year/month; the downloaded surface fields
are stamped at 06:00 and PCs at 00:00. A common continuous endpoint intersection
is used across all four targets and both PC sets. Any truncation of EOF training
months is recorded. Gaps and repeated months are rejected, never interpolated.
Each target subtracts its own calendar-month climatology over this interval.
All months receive equal temporal weight. No detrending or filtering occurs.

Heat is positive downward: positive anomalies indicate increased downward
exchange or reduced ocean heat loss; negative latent anomalies indicate
enhanced evaporative loss. Stress is positive eastward/northward. Original PC
signs and original training-period SD units are preserved. Predictors are
centred on the regression sample without restandardising or reorthogonalising.

## Mask

The exact ERA5 grid is required; no target regridding occurs. Ocean means land
fraction <0.5; mixed coastal cells are retained. Regional tables repeat the
calculation for land fraction ≤0.05. No fractional-area rescaling is used to
pretend mixed-cell fluxes are pure-ocean values. The static mask is verified
across its supplied monthly copies. Full-record finite cells are used for both
methods, with accepted and finite sample counts saved. Missing SST is irrelevant
because SST is not an input.

Ice-covered ocean locations are retained so their exclusion does not change
temporal samples. Their fluxes are surface exchange including sea ice, not
necessarily heat entering open water. No ice-concentration separation is claimed.

Repeated masks may differ by at most twice float32 machine epsilon (2.38e-7),
and neither the primary nor pure-ocean classification may change in any month.
The uploaded mask differs by 1.19e-7 at one equatorial coastal cell, with no
classification changes. The first mask record defines fixed fractions, and the
measured change and precision tolerance are retained in provenance.

## Numerical method

For centred predictors `X` and centred anomaly targets `Y`, individual slopes
are `(X.T @ Y) / diag(X.T @ X)` and joint slopes use a reduced QR factorization
and triangular solve. The factors are reused across spatial blocks. Individual
and joint paths are evaluated independently. Intercepts use the original PC
means and are saved. NH/SH predictors are never combined into a 24-PC model.

With diagonal `G = X.T @ X`, individual and joint slopes agree. More generally,
`b_individual = diag(G)^(-1) G b_joint`; this identity is also checked. Centred
joint fits are compared with summed individual contributions, adding the mean
once. Joint R² is compared with summed individual R². These equivalences require
orthogonality and identical samples, masks, scaling and intercept treatment.

Float64 is used for all calculations and saved numerical maps. Orthogonality
requires max off-diagonal correlation ≤1e-12. Coefficient comparisons use
`abs(delta) <= 1e-10 * S + 1e-10 * abs(b_individual)`, where `S` is the global
area-RMS target anomaly SD, floored at 1e-12 physical units. Reconstruction
max errors use `1e-10*S`; R²-additivity uses absolute 1e-10. Relative map errors
use area-RMS coefficient amplitude with a positive scale floor. Constant-target
R²/correlation are missing, with an explicit constant flag.

Nested joint fits use PCs 1, 1–3, 1–6 and 1–12. RMSE is sqrt(SSE/T); descriptive
residual sample variance is SSE/(T−1). Neither is labelled an unbiased estimate
after climatology fitting and serial dependence. No IID significance claims
are made. A full reconstruction cube is not saved.

Spherical cell areas use clipped cell edges, including polar caps. Regional
boundaries clip cell areas rather than double-count boundary cells. The tables
distinguish area-mean local R² from `1 - sum(area*SSE)/sum(area*SST)`. NH/SH
percentages and reconstructions can overlap and must not be added together.

The optional latent+sensible diagnostic uses the intersection of complete cells
and verifies coefficient linearity. Its variance includes latent–sensible
covariance; component R² values are not added. It is turbulent heat flux, not
total net heat flux or complete F′.

## Outputs and validation

Each hemisphere has `individual/`, `joint/` and `comparison/` NetCDFs and PNGs.
Joint NetCDFs include monthly climatologies, masks, sample counts, residual
statistics and nested-model results. Differences, prediction errors and
R²-additivity diagnostics are stored in comparison files. `reports/` contains
actual-sample PC series/Gram/correlation matrices, numerical tables, provenance
JSON and the scientific report. `variance_diagnostics/` contains regional tables,
global R² maps and mode-count comparisons. All are ignored by Git.

Figures retain the complete numerical grid. Only vector arrows are thinned.
Exact-pole arrows are omitted because geographical east/north is ambiguous
there; pole regression coefficients remain in the numerical files. Arrow
longitudes are normalized for Cartopy's geographic vector transformation.
Native cell-edge projection coordinates are cached between map panels. The
dateline cell is drawn as two identical half-cells, without interpolation or
averaging, so projection caching does not alter numerical values or resolution.
Paired individual/joint maps share colour limits; difference maps have separate
numerical scales. All 12 stress components are saved, with vector figures for
modes 1–6. Scientific results are in-sample associations, not causal attribution
or out-of-sample validation. Strict prediction tests would require training-only
EOF bases, climatologies and regression coefficients.

```sh
python -m pytest -q
```

[Synthetic regression tests](tests/test_mslp_surface_regression.py) verify known
coefficients, prediction/R² identities, correlated and truncated predictors,
units/signs, calendar alignment, missing/constant targets, fixed masks, spherical
areas, turbulent linearity, dense references, block independence and CLI IO.
