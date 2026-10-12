# Ocean Grid Viewer Prototype

This repository now contains a minimal but structured prototype for a global ocean-grid viewer. The backend generates a configurable global latitude-longitude grid and classifies each cell as land or ocean from Natural Earth land polygons using center-point masking. The frontend uses TypeScript + CesiumJS to render a coarse, interactive overlay that is aligned to real spherical lat/lon cell corners.

For performance, the underlying 0.25° product is exported as compact binary (`mask_u8.bin`) plus compact metadata (`grid_meta.json`) instead of a giant per-cell polygon JSON. This keeps the base architecture compatible with ~1 million cells while avoiding heavyweight browser parsing for the full-resolution product.

For initial rendering, the backend also exports a coarser 2.0° JSON product for Phase 2. This coarse layer is intentionally small enough to render interactively and verify GUI behavior now, while preserving the 0.25° backend as the target architecture for later LOD refinement.

Cesium rendering is batched using primitives/geometry instances rather than one Cesium entity per cell. Land and ocean are rendered in separate batched primitives with distinct colors, and mouse hover uses scene-level picking so interaction scales with batched geometry.

Longitude uses the `[-180, 180]` range and periodic indexing through regular edge spacing. Grid IDs use row-major indexing (`cell_id = lat_index * n_lon + lon_index`) and all bounds/centers are explicitly defined from edge arrays. Validation checks include expected dimensions, spacing continuity, dateline duplication checks, and bounds checks.

Polar handling uses a small rendering simplification in the frontend: polygons touching ±90° are inset to ±89.999° when generating corner vertices to avoid degenerate duplicate-pole vertices in Cesium geometry construction.

The code is split so the masking strategy, export strategy, Cesium layer, picking, and hover-deformation logic are independent modules. This keeps the prototype easy to evolve toward fractional ocean coverage, binary/tiled delivery, and progressive level-of-detail rendering in later phases.

## Repository structure

```text
ocean-grid-viewer/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── grid.py
│   │   ├── mask.py
│   │   └── export.py
│   ├── data/
│   ├── scripts/
│   │   └── build_grid.py
│   ├── tests/
│   │   └── test_grid.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── public/
│   │   └── data/
│   ├── src/
│   │   ├── main.ts
│   │   ├── globe/
│   │   │   ├── GridLayer.ts
│   │   │   ├── GridPicking.ts
│   │   │   └── HoverDeformation.ts
│   │   ├── ui/
│   │   │   └── Tooltip.ts
│   │   └── config.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
└── README.md
```

## Install and run

From repository root:

```bash
cd /home/runner/work/globetest/globetest
```

Backend setup + grid build:

```bash
cd /home/runner/work/globetest/globetest/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/build_grid.py --resolution 0.25 --coarse-resolution 2.0
pytest -q
```

If the environment cannot download Natural Earth directly, set:

```bash
export NATURAL_EARTH_LAND_SHP=/absolute/path/to/ne_50m_land.shp
```

Frontend setup + run:

```bash
cd /home/runner/work/globetest/globetest/frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Scope

This prototype intentionally does **not** include SST, MLD, currents, fluxes, forecasting, or climate-model logic yet.
