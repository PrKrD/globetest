# Equatorial geostrophic-subduction diagnostics

## Scope and method

This is a diagnostic of the existing method; it does not alter the globe's velocity or
subduction calculation. The baseline uses a 3.5° β-plane spline fit,
0.01 m smoothing parameter, a 0.25° β-only core, and a
smoothstep transition to ordinary geostrophy at ±3°. Ordinary
geostrophy is shown at every nonzero-latitude row, but quantitative "well-conditioned"
comparison uses |latitude| ≥ 3.5°, one full grid stencil beyond
the blend. The diagnostic decomposition reproduces the production stencil to
4.163e-17 m/s.

## Main findings

- The positive signal is concentrated in the transition rather than the full 10°S–10°N
  band. From 0.25° to 3°, current area-weighted
  mean Sgeo is 9.210e-06 m/s and
  61.1% of valid area is positive.
  Across the full tropical band the mean is
  1.653e-06 m/s and the
  positive-area fraction is 48.1%.
- Ordinary geostrophy over the same nonzero transition rows has mean
  -2.176e-05 m/s. Its spatial correlation
  with the blended result is -0.294;
  ordinary values close to the equator remain numerically ill-conditioned and are a
  comparison, not a preferred estimate.
- The exact discrete product-rule residual associated with the blend-weight gradient has
  transition mean 1.386e-05 m/s, compared
  with 1.337e-05
  m/s from `-H (v_f - v_beta) dw/dy`. Their pointwise correlation is
  0.790. This term is therefore
  large enough to explain the positive transition-band mean, while the remaining terms
  partly offset it.

## Numerical sensitivity

- Across all one-at-a-time parameter variants, the 10°S–10°N mean remains positive,
  spanning 1.062e-06 to 2.440e-06 m/s. The transition-band
  mean spans 6.302e-06 to 1.232e-05 m/s.
- Spatial correlations with the baseline span 0.177 to
  1.000. The blend-width sweep produces the strongest pattern changes;
  the 1.5° blend has baseline correlation
  0.177.
- The unsmoothed spline is materially different. Smoothing values 0.01, 0.03, and 0.1 m
  give the same fitted result for this dataset because the spline residual already falls
  below those thresholds; this should not be read as general smoothing insensitivity.
- The sign of the broad transition-band mean is robust over the tested settings, but many
  local anomalies and their magnitudes are not. See `06_parameter_robustness.png` for the
  per-cell sign frequency and ensemble spread.

## Conservation check

The source grid stops at -66.5° and 77.5° and
its mask combines coastlines with source no-data. Using wet-cell transport to estimate
outward flux across source-mask, derived-missing, and outer-domain faces gives:

- area-integrated Sgeo: 85.86 Sv
- estimated outward boundary flux: -51.75 Sv
- closure residual `integral(Sgeo) + outward flux`:
  34.11 Sv
  (39.7% of the larger budget term)

The current centered/one-sided cell derivative is not a finite-volume flux operator, so
it does not satisfy exact discrete global conservation. Boundary-adjacent cells contribute
34.82 Sv and
interior cells 51.04
Sv. The boundary-flux estimate is approximate because no normal velocity is available on
the dry/no-data side; no equatorial values were zeroed and no anomalies were smoothed.

## Outputs

The numbered PNGs contain velocity components, subduction components, the ordinary
comparison, latitude profiles, parameter sensitivity, sign robustness, and conservation.
The CSV and JSON files retain the plotted statistics for independent inspection.
