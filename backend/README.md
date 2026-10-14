# Backend

Transforms the Maximenko mean dynamic-topography and de Boyer Montégut et al.
(2023) monthly mixed-layer-depth NetCDF products into the globe's common 0.5°
land/sea mask, geostrophic-velocity, and maximum-MLD dataset.

```bash
cd backend
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/build_dyntopo.py
python3 -m pytest -q
```

The raw sources are `data/Maximenko_dyntopo.nc` and
`data/mld_dr003_ref10m_v2023.nc`; the generated browser dataset is
`../frontend/public/data/maximenko_slope.json` (relative to `backend/`). Use
`--input`, `--mld-input`, or `--output` to override those paths.

For MLD, the build takes the maximum of the available 12 monthly climatologies
on the source 1° grid, then uses periodic, mask-aware bilinear interpolation to
reach Maximenko cell centres. It applies the Maximenko mask last and stores the
minimum valid-month count among contributing source cells. No additional
Gaussian smoothing is used. See the
[source dataset and method description](https://doi.org/10.17882/91774); its MLD
criterion is a +0.03 kg/m³ potential-density change relative to 10 m, based on
profiles spanning 1970–2021.
