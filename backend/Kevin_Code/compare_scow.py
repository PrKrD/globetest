
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

# 1. Read ERA5 wind stress
with Dataset("Data/era5_wind_stress.nc") as ds:
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]

    time = ds.variables["valid_time"]
    dates = num2date(time[:], time.units,
                     calendar=getattr(time, "calendar", "standard"))

    # Find all Januaries between 2005 and 2020
    january = [i for i, date in enumerate(dates)
               if date.month == 1]

    # Average January wind stresses
    tau_x = np.ma.filled(
        np.ma.mean(ds.variables["iews"][january, :, :], axis=0),
        np.nan
    )
    tau_y = np.ma.filled(
        np.ma.mean(ds.variables["inss"][january, :, :], axis=0),
        np.nan
    )

print("Number of Januaries:", len(january))

# 2. Spherical coordinates
R = 6.371e6
phi = np.deg2rad(lat)
lam = np.deg2rad(lon)
cos_phi = np.cos(phi)

# 3. Calculate wind-stress curl

# Longitude derivative (periodic)
dlambda = lam[1] - lam[0]

d_tauy_dlambda = (
    np.roll(tau_y, -1, axis=1)
    - np.roll(tau_y, 1, axis=1)
) / (2 * dlambda)

# Latitude derivative
d_tauxcos_dphi = np.gradient(
    tau_x * cos_phi[:, None], phi, axis=0
)

# Spherical wind-stress curl
with np.errstate(divide="ignore", invalid="ignore"):
    curl = (
        d_tauy_dlambda - d_tauxcos_dphi
    ) / (R * cos_phi[:, None])

# 4. Apply ocean mask
with Dataset("Data/era5_land_sea_mask.nc") as ds:
    lsm = np.squeeze(
        np.ma.filled(ds.variables["lsm"][:], np.nan)
    )
    assert np.allclose(lat, ds.variables["latitude"][:])
    assert np.allclose(lon, ds.variables["longitude"][:])

assert lsm.shape == curl.shape

ocean = lsm < 0.05

# Exclude land and adjacent coastal cells
safe = ocean.copy()
safe &= np.roll(ocean, 1, axis=1)
safe &= np.roll(ocean, -1, axis=1)
safe[1:-1] &= ocean[:-2] & ocean[2:]
safe[0] = False
safe[-1] = False

# Match the approximate latitude coverage of SCOW
safe &= (np.abs(lat[:, None]) < 70)

curl = np.where(safe, curl, np.nan)

# 5. Plot in units of 10^-7 N/m^3
curl_scaled = curl / 1e-7

limit = np.nanpercentile(np.abs(curl_scaled), 95)

plt.figure(figsize=(12, 5))

plt.pcolormesh(
    lon, lat, curl_scaled,
    shading="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(label="Wind-stress curl (10^-7 N/m^3)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("ERA5 January Wind-Stress Curl (2005–2020)")

plt.tight_layout()
plt.savefig("era5_january_curl.png", dpi=200)
plt.show()

print("95th percentile absolute curl:", limit, "x 10^-7 N/m^3")
