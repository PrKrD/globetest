
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# 1. Open Holte dataset
with Dataset("Data/Argo_mixedlayers_monthlyclim_04142022.nc") as ds:

    latitude = ds.variables["lat"][:]
    longitude = ds.variables["lon"][:]

    def read_mld(name):
        var = ds.variables[name]

        # Disable automatic masking by valid_min/valid_max
        var.set_auto_mask(False)

        raw = np.asarray(var[:], dtype=float)
        fill = var.getncattr("_FillValue")

        # Exclude only missing and nonphysical values
        valid = (
            np.isfinite(raw)
            & (raw != fill)
            & (raw > 0)
        )

        return np.ma.array(raw, mask=~valid)

    mld_da = read_mld("mld_da_mean")
    mld_dt = read_mld("mld_dt_mean")

# 2. Count valid months at every grid point
valid_months_da = np.ma.count(mld_da, axis=2)
valid_months_dt = np.ma.count(mld_dt, axis=2)

# 3. Maximum of the 12 monthly mean MLDs
H_da = np.ma.max(mld_da, axis=2).filled(np.nan)
H_dt = np.ma.max(mld_dt, axis=2).filled(np.nan)

# 4. Also create stricter versions using all 12 months
H_da_complete = np.where(valid_months_da == 12, H_da, np.nan)
H_dt_complete = np.where(valid_months_dt == 12, H_dt, np.nan)

# 5. Print statistics
for name, H, count in [
    ("Density algorithm", H_da, valid_months_da),
    ("Density threshold", H_dt, valid_months_dt)
]:
    print("\n", name)
    print("Shape:", H.shape)
    print("Maximum H:", np.nanmax(H), "m")
    print("Median H:", np.nanmedian(H), "m")
    print("98th percentile:", np.nanpercentile(H, 98), "m")
    print("Valid grid points:", np.isfinite(H).sum())
    print("12 valid months:", np.sum(count == 12))
    print("1-11 valid months:",
          np.sum((count > 0) & (count < 12)))
    print("No valid months:", np.sum(count == 0))

# 6. Save results
np.savez_compressed(
    "holte_H_corrected.npz",
    latitude=latitude,
    longitude=longitude,
    H_da=H_da,
    H_dt=H_dt,
    H_da_complete=H_da_complete,
    H_dt_complete=H_dt_complete,
    valid_months_da=valid_months_da,
    valid_months_dt=valid_months_dt
)

# 7. Plot corrected density-threshold MLD
plt.figure(figsize=(12, 5))

plt.pcolormesh(
    longitude,
    latitude,
    H_dt,
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=np.nanpercentile(H_dt, 98)
)

plt.colorbar(label="Maximum monthly mean MLD (m)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Holte Maximum Mixed Layer Depth - Corrected")

plt.tight_layout()
plt.savefig("holte_H_corrected.png", dpi=200)
plt.show()

