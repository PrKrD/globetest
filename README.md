# Ocean Velocity Globe

An interactive Cesium globe combining the Maximenko mean dynamic-topography
product with the de Boyer Montégut et al. (2023) mixed-layer-depth climatology.
The native 0.5° Maximenko grid supplies the common display grid. The de Boyer
Montégut product supplies the displayed land/ocean mask.

## What the globe shows

- **Mask:** the de Boyer Montégut 1° land/ocean mask is remapped to the 0.5°
  display grid by spherical overlap area. It is data-product coverage, not a
  high-resolution coastline. Ocean cells without Maximenko data remain ocean.
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
  source climatology has already been objectively mapped. MLD is displayed
  throughout the remapped de Boyer Montégut ocean wherever source data are
  available, including cells without Maximenko coverage.
- **Geostrophic subduction:** the signed diagnostic is
  `S_geo = -div(H v_geo)`, evaluated with a conservative finite-volume
  spherical divergence. Positive values indicate horizontal transport
  convergence and a downward contribution; negative values indicate divergence
  and an upward contribution. Values are in m/s and use the maximum MLD
  described above.
- **Controls:** subduction, H, and velocity have mutually exclusive switches,
  with H on by default. Subduction and velocity each have an independent,
  default-on **Use experimental equator** sub-toggle. Turning one off paints the
  experimental band red without removing its hover values. A product legend
  shows the active scale and its 98th-percentile clipping limit. Outside that
  band, ocean cells where velocity or subduction is unavailable are also red.
  This includes de Boyer Montégut ocean cells without Maximenko data.

Hover details contain independent **Geostrophic velocity**, **Maximum
mixed-layer depth (H)**, and **Geostrophic subduction** sections. H reports its
exact interpolated value and a conservative source-coverage count. If any
contributing source cell lacks a month, the lowest contributor count is shown
as "at least X/12 months" and marked incomplete. Subduction reports its sign
meaning and whether it inherits an experimental equatorial velocity estimate.

Detailed reconstruction, finite-volume, masking, and boundary methods are
documented in [the backend guide](backend/README.md).

## Build the dataset

The raw sources must be at:

- `backend/data/Maximenko_dyntopo.nc`
- `backend/data/mld_dr003_ref10m_v2023.nc`

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
- **Subduction calculation:** this is a conservative diagnostic combining mean
  geostrophic velocity with maximum climatological MLD, not a seasonally
  resolved subduction estimate. See the [backend method](backend/README.md).
- **Source MLD:** the bundled `mld_dr003_ref10m_v2023.nc` is the 2023 monthly
  climatology described by
  [de Boyer Montégut et al., SEANOE](https://doi.org/10.17882/91774). It maps
  profiles collected from 1970–2021 and defines MLD using a potential-density
  threshold of +0.03 kg/m³ relative to the value at 10 m depth. The displayed H
  uses every available month; incomplete high-latitude source coverage remains
  visible in hover details. Bilinear regridding improves display continuity but
  does not create genuine 0.5° observations.

## Diagnostic status

The equatorial velocity and inherited subduction estimates remain experimental
and require independent physical validation. The diagnostic workflow tests
parameter sensitivity, separates the blend-gradient contribution, verifies
finite-volume conservation, and compares the current operator with the retired
centered/one-sided stencil. Its generated reports, tables, and figures are local
artifacts; reproduction instructions are in [the backend guide](backend/README.md).

## Run the globe

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
