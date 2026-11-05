
"""Compare geostrophic subduction using de Boyer and Holte MLD."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from app.config import DATA_DIR
from app.dyntopo import load_maximenko
from app.mld import (
    load_mld,
    maximum_mld,
    regrid_maximum_mld,
    regrid_ocean_mask_by_overlap,
)
from app.velocity import reconstruct_geostrophic_velocity
from app.subduction import geostrophic_subduction_rate

parser = argparse.ArgumentParser()
parser.add_argument("--holte", required=True)
args = parser.parse_args()

# 1. Load last week's data
source = load_maximenko(
    str(DATA_DIR / "Maximenko_dyntopo.nc")
)

mld_source = load_mld(
    str(DATA_DIR / "mld_dr003_ref10m_v2023.nc")
)

# 2. Calculate geostrophic velocity
velocity = reconstruct_geostrophic_velocity(
    source.height_m,
    source.longitude_deg,
    source.latitude_deg
)

# 3. Use the same physical ocean mask as last week
physical_ocean = regrid_ocean_mask_by_overlap(
    mld_source.source_ocean,
    mld_source.longitude_deg,
    mld_source.latitude_deg,
    source.longitude_deg,
    source.latitude_deg,
)

# 4. Regrid de Boyer's H
deboyer_max = maximum_mld(mld_source.monthly_m)

deboyer = regrid_maximum_mld(
    deboyer_max.maximum_m,
    deboyer_max.month_count,
    mld_source.longitude_deg,
    mld_source.latitude_deg,
    source.longitude_deg,
    source.latitude_deg,
    physical_ocean,
)

# 5. Load Holte's corrected H
holte = np.load(args.holte)
print("Available variables:", holte.files)

H_holte = holte["H"]
lat_h = holte["lat"]
lon_h = holte["lon"]

# Only retain grid cells with all 12 months
month_count_h = np.where(
    np.isfinite(H_holte), 12, 0
)

# 6. Regrid Holte onto the same 0.5-degree grid
holte_grid = regrid_maximum_mld(
    H_holte,
    month_count_h,
    lon_h,
    lat_h,
    source.longitude_deg,
    source.latitude_deg,
    physical_ocean,
)

# 7. Use identical valid cells for both calculations
common = (
    physical_ocean
    & velocity.valid
    & deboyer.valid
    & holte_grid.valid
    & np.isfinite(deboyer.maximum_m)
    & np.isfinite(holte_grid.maximum_m)
)

if not np.any(common):
    raise ValueError("No overlapping valid cells; check regridding.")

# 8. Calculate both geostrophic contributions
def calculate_subduction(H):
    return geostrophic_subduction_rate(
        H,
        velocity.east,
        velocity.north,
        source.longitude_deg,
        source.latitude_deg,
        common,
        physical_ocean,
    )

S_deboyer = calculate_subduction(deboyer.maximum_m)
S_holte = calculate_subduction(holte_grid.maximum_m)

# 9. Compare the results
valid = common & S_deboyer.valid & S_holte.valid

seconds_per_year = 365.25 * 24 * 3600

# Separate geostrophic subduction contributions
S_x = S_deboyer.zonal_rate_m_per_s * seconds_per_year
S_y = S_deboyer.meridional_rate_m_per_s * seconds_per_year
S_total = S_deboyer.rate_m_per_s * seconds_per_year

# Apply common valid mask
S_x = np.where(valid, S_x, np.nan)
S_y = np.where(valid, S_y, np.nan)
S_total = np.where(valid, S_total, np.nan)

# Shared colour scale
all_values = np.concatenate([
    np.abs(S_x[np.isfinite(S_x)]),
    np.abs(S_y[np.isfinite(S_y)]),
    np.abs(S_total[np.isfinite(S_total)])
])

limit = np.percentile(all_values, 98)

# Plot three maps
fig, axes = plt.subplots(3, 1, figsize=(12, 13))

fields = [S_x, S_y, S_total]
titles = [
    "Zonal (East-West) Contribution",
    "Meridional (North-South) Contribution",
    "Total Geostrophic Subduction"
]

for ax, field, title in zip(axes, fields, titles):

    im = ax.pcolormesh(
        source.longitude_deg,
        source.latitude_deg,
        field,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")

fig.colorbar(
    im,
    ax=axes,
    label="Geostrophic contribution (m/year)",
    shrink=0.8
)

plt.savefig(
    "geostrophic_components.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# Compare magnitudes
for name, field in zip(titles, fields):
    values = field[np.isfinite(field)]

    print(name)
    print("RMS:", np.sqrt(np.mean(values**2)), "m/year")
    print("98th percentile:", np.percentile(np.abs(values), 98))
    print()





from scipy.ndimage import gaussian_filter, minimum_filter

# Smooth MLD while respecting missing-data regions
def smooth_mld(H, mask, sigma):

    if sigma == 0:
        return H.copy()

    # Missing cells contribute no value or weight
    values = np.where(mask, H, 0.0)
    weights = mask.astype(float)

    numerator = gaussian_filter(
        values, sigma=sigma, mode=("nearest", "wrap")
    )
    denominator = gaussian_filter(
        weights, sigma=sigma, mode=("nearest", "wrap")
    )

    result = np.full(H.shape, np.nan)

    np.divide(
        numerator, denominator,
        out=result,
        where=denominator > 1e-10
    )

    return np.where(mask, result, np.nan)


# Use the same interior comparison region
# Exclude cells within 3 grid points of missing-data boundaries
interior = minimum_filter(
    common.astype(np.uint8),
    size=(7, 7),
    mode=("constant", "wrap"),
    cval=0
).astype(bool)

# Focus on latitudes 10-65 degrees in either hemisphere
latitude = source.latitude_deg[:, None]

interior &= (
    (np.abs(latitude) >= 10)
    & (np.abs(latitude) <= 65)
)

print("Interior comparison cells:", interior.sum())

# Compare different smoothing strengths
for sigma in [0, 1, 2]:

    H_d = smooth_mld(
        deboyer.maximum_m, common, sigma
    )

    H_h = smooth_mld(
        holte_grid.maximum_m, common, sigma
    )

    # Recalculate subduction using the existing method
    S_d = calculate_subduction(H_d)
    S_h = calculate_subduction(H_h)

    comparison = (
        interior & S_d.valid & S_h.valid
    )

    old = (
        S_d.rate_m_per_s[comparison]
        * seconds_per_year
    )

    new = (
        S_h.rate_m_per_s[comparison]
        * seconds_per_year
    )

    difference = new - old

    def rms(x):
        return np.sqrt(np.mean(x**2))

    print(f"\n--- Smoothing sigma = {sigma} ---")
    print("Compared cells:", comparison.sum())
    print("de Boyer RMS:", rms(old))
    print("Holte RMS:", rms(new))
    print("Difference RMS:", rms(difference))
    print("Median absolute difference:",
          np.median(np.abs(difference)))











# Compare the zonal and meridional subduction contributions

components = [
    (
        "Zonal (east-west)",
        S_deboyer.zonal_rate_m_per_s,
        S_holte.zonal_rate_m_per_s
    ),
    (
        "Meridional (north-south)",
        S_deboyer.meridional_rate_m_per_s,
        S_holte.meridional_rate_m_per_s
    )
]

for name, old_component, new_component in components:

    old = old_component[valid] * seconds_per_year
    new = new_component[valid] * seconds_per_year

    difference = new - old

    print(f"\n--- {name} ---")

    print("de Boyer RMS:",
          np.sqrt(np.mean(old**2)))

    print("Holte RMS:",
          np.sqrt(np.mean(new**2)))

    print("RMS difference:",
          np.sqrt(np.mean(difference**2)))

    print("Median absolute difference:",
          np.median(np.abs(difference)))






old = S_deboyer.rate_m_per_s * seconds_per_year
new = S_holte.rate_m_per_s * seconds_per_year

difference = np.where(valid, new - old, np.nan)

print("Overlapping valid cells:", np.sum(valid))
print("de Boyer RMS (m/year):",
      np.sqrt(np.mean(old[valid] ** 2)))
print("Holte RMS (m/year):",
      np.sqrt(np.mean(new[valid] ** 2)))
print("RMS difference (m/year):",
      np.sqrt(np.mean(difference[valid] ** 2)))
print("Median absolute difference (m/year):",
      np.median(np.abs(difference[valid])))

# 10. Save the numerical results
np.savez_compressed(
    "holte_geostrophic_comparison.npz",
    latitude=source.latitude_deg,
    longitude=source.longitude_deg,
    H_deboyer=deboyer.maximum_m,
    H_holte=holte_grid.maximum_m,
    S_deboyer_m_per_s=S_deboyer.rate_m_per_s,
    S_holte_m_per_s=S_holte.rate_m_per_s,
    difference_m_per_year=difference,
    common_valid=valid,
)

# 11. Plot the difference map
limit = np.nanpercentile(np.abs(difference), 98)

plt.figure(figsize=(12, 5))
plt.pcolormesh(
    source.longitude_deg,
    source.latitude_deg,
    difference,
    shading="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
)

plt.colorbar(label="Holte - de Boyer (m/year)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Difference in Geostrophic Subduction")

plt.tight_layout()
plt.savefig("holte_geostrophic_difference.png", dpi=200)
plt.show()


# DIAGNOSTIC: MLD gradients and numerical boundaries

for name, H, result in [
    ("de Boyer", deboyer.maximum_m, S_deboyer),
    ("Holte", holte_grid.maximum_m, S_holte),
]:
    print(f"\n--- {name} ---")

    # MLD statistics on the same comparison region
    print("Median H (m):", np.median(H[valid]))
    print("H 98th percentile (m):",
          np.percentile(H[valid], 98))

    # Differences between neighbouring valid grid cells
    for axis, direction in [(0, "north-south"),
                            (1, "east-west")]:

        neighbour_valid = np.roll(valid, -1, axis=axis)
        pairs = valid & neighbour_valid

        # Latitude is not periodic
        if axis == 0:
            pairs[-1, :] = False

        neighbour_H = np.roll(H, -1, axis=axis)
        jumps = np.abs(neighbour_H - H)[pairs]

        print(f"{direction} H jump p95 (m):",
              np.percentile(jumps, 95))

    # Subduction statistics
    rates = np.abs(
        result.rate_m_per_s[valid] * seconds_per_year
    )

    print("Subduction |p98| (m/year):",
          np.percentile(rates, 98))

    # Examine treatment of missing-data boundaries
    budget = result.conservation

    print("Missing-data boundary faces:",
          budget.missing_data_face_count)

    print("Missing-data outward transport (m3/s):",
          budget.missing_data_outward_m3_per_s)

    print("Conservation closure residual (m3/s):",
          budget.closure_residual_m3_per_s)
