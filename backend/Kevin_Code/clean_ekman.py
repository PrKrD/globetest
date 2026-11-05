#


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Load the Ekman pumping result
data = np.load("ekman_pumping.npz")

latitude = data["latitude"]
longitude = data["longitude"]
w = data["w_Ek_m_per_year"]

# Load the land-sea mask
with Dataset("Data/era5_land_sea_mask.nc") as ds:
    lsm = np.ma.filled(ds.variables["lsm"][:], np.nan)

    mask_lat = ds.variables["latitude"][:]
    mask_lon = ds.variables["longitude"][:]

# Remove any singleton time dimensions
lsm = np.squeeze(lsm)

# Check that the grids agree
assert lsm.shape == w.shape
assert np.allclose(latitude, mask_lat)
assert np.allclose(longitude, mask_lon)

# Keep ocean points with less than 5% land
ocean = lsm < 0.05

# Exclude points touching non-ocean cells
safe_ocean = (
    ocean
    & np.roll(ocean, 1, axis=1)
    & np.roll(ocean, -1, axis=1)
)

safe_ocean[1:-1] &= ocean[:-2] & ocean[2:]
safe_ocean[0, :] = False
safe_ocean[-1, :] = False

# Set land and nearby coastal points to NaN
w_clean = np.where(safe_ocean, w, np.nan)

# Calculate colour limits from ocean values only
valid = w_clean[np.isfinite(w_clean)]
limit = np.percentile(np.abs(valid), 95)

# Plot
plt.figure(figsize=(12, 5))

plt.pcolormesh(
    longitude,
    latitude,
    w_clean,
    shading="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(label="Ekman pumping (m/year)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Mean Ocean Ekman Pumping (2005–2020)")

plt.tight_layout()
plt.savefig("ekman_pumping_ocean.png", dpi=200)
plt.show()

# Save the corrected field
np.savez_compressed(
    "ekman_pumping_ocean.npz",
    latitude=latitude,
    longitude=longitude,
    w_Ek=w_clean / (365.25 * 24 * 3600),
    w_Ek_m_per_year=w_clean
)

print("Ocean-only colour limit:", limit)
print("Median absolute pumping:",
      np.median(np.abs(valid)))
print("95th percentile:",
      np.percentile(np.abs(valid), 95))

# Check representative ocean regions
regions = {
    "North Pacific subtropical": (15, 35, 170, 220),
    "North Atlantic subtropical": (15, 35, 290, 340),
    "North Pacific subpolar": (45, 60, 170, 210)
}

for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():

    lat_mask = (latitude >= lat_min) & (latitude <= lat_max)
    lon_mask = (longitude >= lon_min) & (longitude <= lon_max)

    region = w_clean[np.ix_(lat_mask, lon_mask)]

    print(name)
    print("Mean:", np.nanmean(region), "m/year")
    print("Median:", np.nanmedian(region), "m/year")
    print()