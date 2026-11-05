
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Check the new ERA5 product
with Dataset("Data/era5_mean_wind_stress.nc") as ds:
    print("New ERA5 metadata:")
    print(getattr(ds, "history", "History not available"))

# Load our two calculations
old = np.load("ekman_pumping_ocean.npz")
new = np.load("ekman_pumping_new.npz")

latitude = old["latitude"]
longitude = old["longitude"]

# Check coordinates match
assert np.allclose(latitude, new["latitude"])
assert np.allclose(longitude, new["longitude"])

w_old = old["w_Ek_m_per_year"]
w_new = new["w_Ek_m_per_year"]

# Calculate difference
difference = w_new - w_old

# Keep only points where both results are valid
valid = np.isfinite(w_old) & np.isfinite(w_new)
difference = np.where(valid, difference, np.nan)

# Plot difference
limit = np.nanpercentile(np.abs(difference), 98)

plt.figure(figsize=(12, 5))

plt.pcolormesh(
    longitude,
    latitude,
    difference,
    cmap="RdBu_r",
    shading="auto",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(label="New - Old Ekman pumping (m/year)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Difference in Ekman Pumping (2005–2020)")

plt.tight_layout()
plt.savefig("ekman_difference.png", dpi=200)
plt.show()

# Compare three ocean regions
regions = {
    "North Pacific subtropical": (15, 35, 170, 220),
    "North Atlantic subtropical": (15, 35, 290, 340),
    "North Pacific subpolar": (45, 60, 170, 210)
}

for name, (lat1, lat2, lon1, lon2) in regions.items():

    rows = (latitude >= lat1) & (latitude <= lat2)
    cols = (longitude >= lon1) & (longitude <= lon2)

    # Approximate spherical grid-cell area weights
    weights = np.cos(np.deg2rad(latitude[rows]))[:, None]

    def regional_mean(field):
        values = field[np.ix_(rows, cols)]
        good = np.isfinite(values)
        return np.nansum(values * weights) / np.sum(
            np.broadcast_to(weights, values.shape)[good]
        )

    original = regional_mean(w_old)
    updated = regional_mean(w_new)

    print(f"\n{name}")
    print(f"Old: {original:.2f} m/year")
    print(f"New: {updated:.2f} m/year")
    print(f"Difference: {updated - original:.2f} m/year")
