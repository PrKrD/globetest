# Backend

Builds a configurable global latitude-longitude grid and center-point land/sea mask.

## Quick start

```bash
cd /home/runner/work/globetest/globetest/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/build_grid.py
```

Outputs:
- `backend/data/mask_u8.bin` (0=ocean, 1=land)
- `backend/data/grid_meta.json`
- `frontend/public/data/coarse_grid_<resolution>.json`
- `frontend/public/data/grid_manifest.json` (points frontend to active coarse file)
