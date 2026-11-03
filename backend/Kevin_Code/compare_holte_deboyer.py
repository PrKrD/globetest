
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# 1. File paths
DEBOYER_FILE = "/Users/zhengkaizhang/Desktop/Uni Notes/Year Notes/Fourth Year Notes/MSci Project/Week 2 Research/mld_dr003_ref10m_v2023.nc"
HOLTE_FILE = "holte_H_corrected.npz"

# 2. Load Holte results
holte = np.load(HOLTE_FILE)

lat_h = holte["latitude"]
lon_h = holte["longitude"]

H_dt = holte["H_dt_complete"]
H_da = holte["H_da_complete"]

# 3. Load de Boyer data
with Dataset(DEBOYER_FILE) as ds:

    # Find geographical coordinates
    def get_coordinate(names):
        for name in names:
            if name in ds.variables:
                return ds.variables[name][:]
        raise KeyError(f"Cannot find coordinate: {names}")

    lat_d = get_coordinate(["lat", "latitude"])
    lon_d = get_coordinate(["lon", "longitude"])

    mld = np.ma.masked_invalid(
        ds.variables["mld_dr003"][:]
    )
    mld = np.ma.masked_less_equal(mld, 0)

# de Boyer: (month, latitude, longitude)
print("de Boyer MLD shape:", mld.shape)

assert mld.shape == (12, len(lat_d), len(lon_d))

valid_months_d = np.ma.count(mld, axis=0)

H_d = np.ma.max(mld, axis=0).filled(np.nan)

# Require 12 valid months
H_d = np.where(valid_months_d == 12, H_d, np.nan)

print("de Boyer maximum H:", np.nanmax(H_d))

# 4. Put both datasets in the same coordinate order
def reorder(H, latitude, longitude):

    lat_order = np.argsort(latitude)
    lon_360 = np.mod(longitude, 360)
    lon_order = np.argsort(lon_360)

    return (
        H[np.ix_(lat_order, lon_order)],
        latitude[lat_order],
        lon_360[lon_order]
    )

H_dt, lat_h, lon_h = reorder(H_dt, lat_h, lon_h)
H_da, _, _ = reorder(
    H_da, holte["latitude"], holte["longitude"]
)
H_d, lat_d, lon_d = reorder(H_d, lat_d, lon_d)

# Check actual geographical alignment
assert H_dt.shape == H_d.shape
assert np.allclose(lat_h, lat_d), "Latitude grids differ"
assert np.allclose(lon_h, lon_d), "Longitude grids differ"

# 5. Compare overlapping valid ocean grid cells
common = (
    np.isfinite(H_dt)
    & np.isfinite(H_da)
    & np.isfinite(H_d)
)

diff_dt = np.where(common, H_dt - H_d, np.nan)
diff_da = np.where(common, H_da - H_d, np.nan)

# Use spherical area weighting for mean differences
weights = np.broadcast_to(
    np.cos(np.deg2rad(lat_h))[:, None],
    H_d.shape
)

print("\nCommon valid grid points:", np.sum(common))

for name, difference in [
    ("Holte DT - de Boyer", diff_dt),
    ("Holte DA - de Boyer", diff_da)
]:
    mean_diff = np.average(
        difference[common], weights=weights[common]
    )

    median_abs = np.median(
        np.abs(difference[common])
    )

    print("\n", name)
    print("Area-weighted mean difference:", mean_diff, "m")
    print("Median absolute difference:", median_abs, "m")

# 6. Plot Holte DT minus de Boyer
limit = np.nanpercentile(np.abs(diff_dt), 98)

plt.figure(figsize=(12, 5))

plt.pcolormesh(
    lon_h, lat_h, diff_dt,
    cmap="RdBu_r",
    shading="auto",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(label="Holte DT - de Boyer (m)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Difference in Maximum Mixed Layer Depth")

plt.tight_layout()
plt.savefig("holte_deboyer_difference.png", dpi=200)
plt.show()
