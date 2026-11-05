"""Combine ERA5 Ekman pumping with geostrophic subduction."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser()
parser.add_argument("--ekman", required=True)
args = parser.parse_args()

seconds_per_year = 365.25 * 24 * 3600

# 1. Load our existing calculations
ekman = np.load(args.ekman)
geo = np.load("holte_geostrophic_comparison.npz")

lat_e = ekman["latitude"]
lon_e = ekman["longitude"]
w_e = ekman["w_Ek"]  # m/s

lat_g = geo["latitude"]
lon_g = geo["longitude"]

S_d = geo["S_deboyer_m_per_s"]
S_h = geo["S_holte_m_per_s"]

# 2. Prepare ERA5 coordinates for interpolation
# ERA5 latitudes descend from north to south.
lat_e = lat_e[::-1]
w_e = w_e[::-1, :]

# Extend longitude periodically across 0/360 degrees
lon_extended = np.concatenate((
    [lon_e[-1] - 360],
    lon_e,
    [lon_e[0] + 360]
))

def extend(arr):
    return np.concatenate(
        (arr[:, -1:], arr, arr[:, :1]), axis=1
    )

# 3. Interpolate Ekman pumping onto geostrophic grid
# Preserve missing-data coverage to avoid filling land gaps.
coverage = np.isfinite(w_e).astype(float)
values = np.where(np.isfinite(w_e), w_e, 0.0)

def interpolate(field):
    interpolator = RegularGridInterpolator(
        (lat_e, lon_extended),
        extend(field),
        bounds_error=False,
        fill_value=np.nan
    )

    lat_grid, lon_grid = np.meshgrid(
        lat_g, np.mod(lon_g, 360),
        indexing="ij"
    )

    points = np.column_stack((
        lat_grid.ravel(),
        lon_grid.ravel()
    ))

    return interpolator(points).reshape(lat_grid.shape)

w_sum = interpolate(values)
w_coverage = interpolate(coverage)

# Only keep cells with effectively complete valid coverage
w_regridded = np.where(
    w_coverage >= 0.999,
    w_sum / np.maximum(w_coverage, 1e-12),
    np.nan
)

# 4. Combine the two contributions
valid = (
    geo["common_valid"].astype(bool)
    & np.isfinite(w_regridded)
    & np.isfinite(S_d)
    & np.isfinite(S_h)
)

Su_d = np.where(valid, S_d - w_regridded, np.nan)
Su_h = np.where(valid, S_h - w_regridded, np.nan)

# 5. Save final results
np.savez_compressed(
    "total_subduction.npz",
    latitude=lat_g,
    longitude=lon_g,
    valid=valid,
    w_Ek_m_per_s=w_regridded,
    S_geo_deboyer_m_per_s=np.where(valid, S_d, np.nan),
    S_geo_holte_m_per_s=np.where(valid, S_h, np.nan),
    Su_deboyer_m_per_s=Su_d,
    Su_holte_m_per_s=Su_h
)

# 6. Print statistics
def rms(x):
    return np.sqrt(np.mean(x**2))

print("Common valid cells:", valid.sum())

for name, field in [
    ("Ekman contribution", -w_regridded),
    ("Total - de Boyer", Su_d),
    ("Total - Holte", Su_h)
]:
    values = field[valid] * seconds_per_year

    print(f"\n{name}")
    print("Mean (m/year):", np.mean(values))
    print("RMS (m/year):", rms(values))
    print("Median (m/year):", np.median(values))

# 7. Plot total subduction using de Boyer
plot = Su_d * seconds_per_year
limit = np.nanpercentile(np.abs(plot), 98)

plt.figure(figsize=(12, 5))
plt.pcolormesh(
    lon_g, lat_g, plot,
    shading="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)
plt.colorbar(label="Total subduction (m/year)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Total Subduction - de Boyer + ERA5")

plt.tight_layout()
plt.savefig("total_subduction.png", dpi=200)
plt.show()

print("\nSaved total_subduction.npz")
print("Saved total_subduction.png")
