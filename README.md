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
- **Controls:** velocity and H have mutually exclusive switches. Velocity is on
  by default; enabling one product disables the other, while turning both off
  shows the flat Maximenko mask. The default-on **Use experimental equator**
  sub-toggle is enabled only for velocity. A product legend shows the active
  square-root colour scale, capped at its 98th percentile.

Hover details contain independent **Geostrophic velocity** and **Maximum
mixed-layer depth (H)** sections. H reports its exact interpolated value and a
conservative source-coverage count. If any contributing source cell lacks a
month, the lowest contributor count is shown as "at least X/12 months" and
marked incomplete.

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
PYTHONPATH=. python3 scripts/build_dyntopo.py
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
- **Source MLD:** the bundled `mld_dr003_ref10m_v2023.nc` is the 2023 monthly
  climatology described by
  [de Boyer Montégut et al., SEANOE](https://doi.org/10.17882/91774). It maps
  profiles collected from 1970–2021 and defines MLD using a potential-density
  threshold of +0.03 kg/m³ relative to the value at 10 m depth. The displayed H
  uses every available month; incomplete high-latitude source coverage remains
  visible in hover details. Bilinear regridding improves display continuity but
  does not create genuine 0.5° observations.

## Run the globe

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
