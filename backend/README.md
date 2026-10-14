# Backend

Transforms the Maximenko mean dynamic-topography and de Boyer Montégut et al.
(2023) monthly mixed-layer-depth NetCDF products into the globe's common 0.5°
land/sea mask, geostrophic-velocity, and maximum-MLD dataset.
It also calculates the signed geostrophic contribution to subduction,
`S_geo = -div(H v_geo)`, in m/s.

```bash
cd backend
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/build_ocean_maps.py
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

Subduction uses the metric-correct spherical vector divergence:

```text
S_geo = -[d(Hu)/d(lambda) + d(Hv cos(phi))/d(phi)] / [R cos(phi)]
```

Longitude is periodic. Derivatives are centered where both neighbours are
usable and one-sided where only one is usable. The result is unavailable if
either directional derivative cannot be formed, or if H or geostrophic
velocity is unavailable. Experimental equatorial velocities are included and
identified in the browser. The generated JSON stores `has_subduction`,
`subduction_m_per_s`, and the symmetric colour limit
`subduction_abs_p98_m_per_s`.

## Equatorial diagnostics

Run the separate diagnostic workflow without changing the generated globe data:

```bash
cd backend
PYTHONPATH=. python3 scripts/diagnose_equatorial_subduction.py
```

Outputs are written to `diagnostics/equatorial_subduction/`. They include maps
of the eastward and northward velocities, zonal and meridional subduction
terms, the total field, ordinary-geostrophy comparisons, area-weighted latitude
profiles, parameter sweeps, the explicit blend-weight-gradient contribution,
and a global boundary-flux conservation budget. `REPORT.md`, CSV tables, and
`summary.json` contain the corresponding numerical results. Use `--output-dir`
to write an alternative diagnostic directory.
