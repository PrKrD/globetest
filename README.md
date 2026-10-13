# Ocean Velocity Globe

An interactive Cesium globe built from the Maximenko mean dynamic-topography
product. Its native 0.5° grid supplies both the displayed land/no-data mask and
a derived geostrophic-velocity layer.

## What the globe shows

- **Mask:** Maximenko `MDOT` fill values are displayed as land/no-data. This is
  data coverage, not a high-resolution coastline.
- **Velocity:** horizontal ODT gradients are converted to geostrophic velocity
  outside ±2° latitude. The ±2°–±5° band is coloured red to indicate low
  confidence; ordinary geostrophy is undefined at the equator.
- **Controls:** the bottom-left switch enables/disables the velocity colour
  layer. Off shows the flat Maximenko land/sea mask.

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

## Run the globe

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
