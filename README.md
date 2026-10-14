# Ocean Velocity Globe

An interactive Cesium globe combining the Maximenko mean dynamic-topography
product with the de Boyer Montégut et al. (2023) mixed-layer-depth climatology.
The native 0.5° Maximenko grid supplies the displayed land/no-data mask and the
common display grid for both products.

## What the globe shows

- **Mask:** Maximenko `MDOT` fill values are displayed as land/no-data. This is
  data coverage, not a high-resolution coastline.
- **Velocity:** outside ±3°, horizontal MDT gradients use ordinary geostrophy.
  At 0°, a cubic smoothing spline supplies the curvature terms for an
  equatorial β-plane estimate; a cubic smoothstep blends it into ordinary
  geostrophy between 0° and ±3°. Equatorial estimates are identified in the
  hover detail because this balanced estimate does not include wind-driven or
  other ageostrophic flow.
- **Maximum mixed-layer depth (H):** at every source cell, H is the largest
  valid value among the 12 monthly climatological MLD maps. H is calculated on
  the source 1° grid before periodic, mask-aware bilinear interpolation onto
  the Maximenko grid. No additional Gaussian smoothing is applied because the
  source climatology has already been objectively mapped.
- **Geostrophic subduction:** the signed diagnostic is
  `S_geo = -div(H v_geo)`, evaluated with the metric-correct spherical vector
  divergence. Positive values indicate horizontal transport convergence and a
  downward contribution; negative values indicate divergence and an upward
  contribution. Values are in m/s and use the maximum MLD described above.
- **Controls:** subduction, H, and velocity have mutually exclusive switches,
  with velocity on by default. Subduction and velocity each have an independent,
  default-on **Use experimental equator** sub-toggle. Turning one off paints the
  experimental band red without removing its hover values. A product legend
  shows the active scale and its 98th-percentile clipping limit. Outside that
  band, ocean cells where velocity or subduction is unavailable are also red.

Hover details contain independent **Geostrophic velocity**, **Maximum
mixed-layer depth (H)**, and **Geostrophic subduction** sections. H reports its
exact interpolated value and a conservative source-coverage count. If any
contributing source cell lacks a month, the lowest contributor count is shown
as "at least X/12 months" and marked incomplete. Subduction reports its sign
meaning and whether it inherits an experimental equatorial velocity estimate.

For subduction, longitude wraps periodically. Each flux derivative uses a
centered difference when both neighbours are available and a one-sided
difference when only one is available. A cell is unavailable unless both the
east-west and north-south derivatives can be formed; a single directional term
is never displayed as the total divergence.

## Build the dataset

The raw sources must be at:

- `backend/data/Maximenko_dyntopo.nc`
- `backend/data/mld_dr003_ref10m_v2023.nc`

### Local macOS/Homebrew note

This project uses the global Python environment rather than a virtual environment.
On the local macOS/Homebrew machine, the working Homebrew Git executable is:

```text
/opt/homebrew/Cellar/git/2.56.0_1/bin/git
```

Use that path if the macOS Command Line Tools installation reports as missing.

```bash
cd backend
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/build_ocean_maps.py
python3 -m pytest -q
```

This creates `frontend/public/data/maximenko_slope.json`. Export is atomic, so a
running local viewer keeps its prior dataset until a new build completes.

## Data and method references

- **Source MDT:** the bundled `Maximenko_dyntopo.nc` is the Maximenko/Niiler
  1992–2002 mean dynamic ocean topography (MDOT) product. It combines satellite
  altimetry, GRACE geoid information, drifter observations, and wind data; it
  was distributed through the University of Hawaiʻi APDRC Dynamic Ocean
  Topography project. See
  [Maximenko et al. (2009), *Mean Dynamic Topography of the Ocean Derived from
  Satellite and Drifting Buoy Data Using Three Different Techniques*](https://doi.org/10.1175/2009JTECHO672.1).
- **Equatorial calculation:** the β-plane/f-plane transition follows the
  equatorial current-estimation approach of
  [Lagerloef et al. (1999)](https://doi.org/10.1029/1999JC900197). This viewer
  uses a local cubic smoothing spline to obtain the needed first and second
  topography derivatives before applying that transition.
- **Subduction calculation:** on longitude `lambda` and latitude `phi`, the
  implementation evaluates
  `-[d(Hu)/d(lambda) + d(Hv cos(phi))/d(phi)] / [R cos(phi)]`. It is a diagnostic
  combining mean geostrophic velocity with maximum climatological MLD, not a
  seasonally resolved subduction estimate.
- **Source MLD:** the bundled `mld_dr003_ref10m_v2023.nc` is the 2023 monthly
  climatology described by
  [de Boyer Montégut et al., SEANOE](https://doi.org/10.17882/91774). It maps
  profiles collected from 1970–2021 and defines MLD using a potential-density
  threshold of +0.03 kg/m³ relative to the value at 10 m depth. The displayed H
  uses every available month; incomplete high-latitude source coverage remains
  visible in hover details. Bilinear regridding improves display continuity but
  does not create genuine 0.5° observations.

## Equatorial diagnostic status

The equatorial velocity and subduction fields are experimental and have not yet
been independently or physically validated. The current velocity reconstruction
fits cubic smoothing splines to contiguous MDT samples within ±3.5°. At 0° it
uses the equatorial β-plane curvature relations; from the 0.25° β-plane core to
±3° it blends smoothly into ordinary geostrophic velocity. Ordinary geostrophy
is retained outside that transition.

Diagnostics over 10°S–10°N show that the mostly positive subduction signal in
the transition band is strongly influenced by the spatial gradient of this
blending weight. The exact discrete blend-gradient contribution has an
area-weighted transition-band mean of `1.386e-5 m/s`, close to the analytic
product-rule estimate of `1.337e-5 m/s`, while the resulting total mean is
`9.210e-6 m/s`. The broad positive mean persists across the tested fit widths,
spline smoothing parameters, and blend widths, but many local anomalies depend
strongly on those numerical choices. These results must therefore be treated as
diagnostic rather than established physical subduction.

The current centered/one-sided derivative scheme is also not a conservative
finite-volume operator. Its global area integral does not close exactly against
estimated transport through coastline/no-data and outer-domain boundaries; the
diagnostic closure residual is about 40% of the larger budget term. A future
revision will replace this derivative with a conservative finite-volume flux
calculation and then validate the equatorial velocities independently. No
equatorial values have been forced to zero or smoothed away. See the complete
[diagnostic report](backend/diagnostics/equatorial_subduction/REPORT.md) and its
associated plots and CSV summaries.

## Run the globe

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
