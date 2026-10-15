# Backend

The backend combines Maximenko mean dynamic topography with the de Boyer
Montégut et al. (2023) mixed-layer-depth climatology on the native 0.5° grid.
It exports geostrophic velocity, maximum MLD, and the signed geostrophic
subduction diagnostic

```text
S_geo = -div(H v_geo)
```

Positive values represent horizontal transport convergence and a downward
contribution. The diagnostic combines mean velocity with maximum climatological
MLD; it is not a seasonally resolved subduction estimate.

## Build and test

```bash
cd backend
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/build_ocean_maps.py
python3 -m pytest -q
```

Inputs default to `data/Maximenko_dyntopo.nc` and
`data/mld_dr003_ref10m_v2023.nc`. The generated browser dataset is
`../frontend/public/data/maximenko_slope.json`. Use `--input`,
`--mld-input`, or `--output` to override these paths. Export is atomic.

## Velocity and MLD reconstruction

Outside ±3°, horizontal MDT gradients use ordinary f-plane geostrophy. At the
equator, cubic smoothing splines fitted to contiguous valid latitude runs
supply beta-plane curvature estimates. A cubic smoothstep blends the beta-plane
solution from the 0.25° core into ordinary geostrophy at ±3°. The shared
`reconstruct_geostrophic_velocity` function is used by both production and
diagnostics.

MLD is the maximum of the available 12 monthly climatologies on the source 1°
grid. Periodic, mask-aware bilinear interpolation maps it to the 0.5° grid over
the remapped de Boyer Montégut ocean mask, including ocean cells without
Maximenko coverage, and renormalizes around invalid source corners. The stored
coverage is the minimum valid-month count among contributing cells. No
additional smoothing is applied.
The source product defines MLD using a +0.03 kg/m³ potential-density change
relative to 10 m and is described at
[SEANOE](https://doi.org/10.17882/91774).

## Conservative finite-volume subduction

Let (T_x = Hu) and (T_y = Hv) be cell-centred depth-integrated transports.
For two wet cells sharing a face, the unique oriented face fluxes are

```text
F_e = 0.5 * (T_x,P + T_x,E) * R * Δφ
F_n = 0.5 * (T_y,P + T_y,N) * R * cos(φ_n) * Δλ
```

with equivalent expressions on the west and south faces. Cell area and
subduction are

```text
A = R² * Δλ * [sin(φ_n) - sin(φ_s)]
S_geo = -(F_e - F_w + F_n - F_s) / A
```

Longitude is periodic, including one shared dateline face. Boundary faces use
three explicit policies:

- Physical coastline: zero normal flux.
- Oceanic missing data: transport from the adjacent wet cell.
- Northern and southern domain edges: transport from the adjacent wet cell.

The physical-ocean mask comes from an exact spherical overlap-area remap of the
1° MLD land/ocean mask. A target cell is ocean when at least half its area
overlaps source ocean; ties remain ocean. This mask also defines displayed land
and ocean. Within it, missing Maximenko velocity and inherited subduction values
remain unavailable and are displayed in red.

Cell tendencies and the categorized boundary budget are derived from the same
face-flux arrays. Shared internal faces therefore cancel pairwise, and the
area-integrated subduction closes against total outward boundary transport to
floating-point precision.

## Diagnostics

Run the diagnostic workflow independently of the browser dataset:

```bash
cd backend
PYTHONPATH=. python3 scripts/diagnose_equatorial_subduction.py
```

Outputs are generated under `diagnostics/equatorial_subduction/` and are not
tracked by Git. They include velocity and subduction components, latitude
profiles, parameter sensitivity and robustness, conservation tables, a
Markdown report, and a finite-volume-minus-legacy comparison. The comparison
uses separate boundary and wet-interior colour scales for the full domain and
10°S–10°N. Use `--output-dir` to choose another destination.
