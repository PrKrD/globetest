# Ocean Velocity Globe

An interactive Cesium globe built from the Maximenko mean dynamic-topography
product. Its native 0.5° grid supplies both the displayed land/no-data mask and
a derived geostrophic-velocity layer.

## What the globe shows

- **Mask:** Maximenko `MDOT` fill values are displayed as land/no-data. This is
  data coverage, not a high-resolution coastline.
- **Velocity:** outside ±3°, horizontal MDT gradients use ordinary geostrophy.
  At 0°, a cubic smoothing spline supplies the curvature terms for an
  equatorial β-plane estimate; a cubic smoothstep blends it into ordinary
  geostrophy between 0° and ±3°. Equatorial estimates are identified in the
  hover detail because this balanced estimate does not include wind-driven or
  other ageostrophic flow.
- **Controls:** the bottom-left switch enables/disables the velocity colour
  layer. Off shows the flat Maximenko land/sea mask. Its default-on
  **Use experimental equator** sub-toggle shows the β-plane estimates; off
  marks that experimental band red without rebuilding the globe geometry.

Hover details use sectioned groups, beginning with **Geostrophic velocity**.
Additional products such as mixed-layer depth can add their own independent
section without changing the location and mask fields.

## Build the dataset

The raw source must be at `backend/data/Maximenko_dyntopo.nc`.

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
  Satellite and Drifting Buoy Data Using Three Different Techniques*](https://doi.org/10.1175/2009JTECHO672.1)
  and [Maximenko et al. (2008), *Stationary Mesoscale Jet-Like Features in the
  Ocean*](https://doi.org/10.1029/2008GL033267).
- **Equatorial calculation:** the β-plane/f-plane transition follows the
  equatorial current-estimation approach of
  [Lagerloef et al. (1999)](https://doi.org/10.1029/1999JC900197). This viewer
  uses a local cubic smoothing spline to obtain the needed first and second
  topography derivatives before applying that transition.

## Run the globe

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
