# Backend

Transforms the Maximenko mean dynamic-topography NetCDF product into the globe's
native 0.5° land/sea mask and geostrophic-velocity dataset.

```bash
cd backend
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/build_dyntopo.py
python3 -m pytest -q
```

The raw source is `data/Maximenko_dyntopo.nc`; the generated browser dataset is
`frontend/public/data/maximenko_slope.json`.
