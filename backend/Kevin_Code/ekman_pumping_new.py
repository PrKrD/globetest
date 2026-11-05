
import numpy as np
from netCDF4 import Dataset

# 1. Load new mean wind stresses
data = np.load("mean_wind_stress_new.npz")

latitude = data["latitude"]
longitude = data["longitude"]
tau_x = data["tau_x"]
tau_y = data["tau_y"]

# 2. Physical constants
R = 6.371e6
Omega = 7.2921e-5
rho = 1025
seconds_per_year = 365.25 * 24 * 3600

# 3. Spherical coordinates
phi = np.deg2rad(latitude)
lam = np.deg2rad(longitude)

cos_phi = np.cos(phi)
f = 2 * Omega * np.sin(phi)

# Exclude equatorial and near-polar regions
valid_lat = (np.abs(latitude) >= 10) & (np.abs(latitude) <= 85)
f = np.where(valid_lat, f, np.nan)

# 4. Calculate horizontal Ekman transport
Mx = tau_y / (rho * f[:, None])
My = -tau_x / (rho * f[:, None])

# 5. Calculate spherical divergence
dlambda = lam[1] - lam[0]

dMx_dlambda = (
    np.roll(Mx, -1, axis=1)
    - np.roll(Mx, 1, axis=1)
) / (2 * dlambda)

dMycos_dphi = np.gradient(
    My * cos_phi[:, None], phi, axis=0
)

with np.errstate(divide="ignore", invalid="ignore"):
    w_new = (
        dMx_dlambda + dMycos_dphi
    ) / (R * cos_phi[:, None])

# 6. Apply the same ocean mask as before
with Dataset("Data/era5_land_sea_mask.nc") as ds:
    lsm = np.squeeze(
        np.ma.filled(ds.variables["lsm"][:], np.nan)
    )

    assert np.allclose(latitude, ds.variables["latitude"][:])
    assert np.allclose(longitude, ds.variables["longitude"][:])

ocean = lsm < 0.05

safe_ocean = (
    ocean
    & np.roll(ocean, 1, axis=1)
    & np.roll(ocean, -1, axis=1)
)

safe_ocean[1:-1] &= ocean[:-2] & ocean[2:]
safe_ocean[0, :] = False
safe_ocean[-1, :] = False

w_new = np.where(safe_ocean, w_new, np.nan)

# 7. Save new Ekman pumping
np.savez_compressed(
    "ekman_pumping_new.npz",
    latitude=latitude,
    longitude=longitude,
    w_Ek=w_new,
    w_Ek_m_per_year=w_new * seconds_per_year
)

# 8. Compare with original Ekman pumping
old = np.load("ekman_pumping_ocean.npz")

assert np.array_equal(latitude, old["latitude"])
assert np.array_equal(longitude, old["longitude"])

w_old = old["w_Ek"]

valid = np.isfinite(w_old) & np.isfinite(w_new)

a = w_old[valid] * seconds_per_year
b = w_new[valid] * seconds_per_year

difference = b - a

print("Number of points compared:", len(a))
print("Original RMS (m/year):", np.sqrt(np.mean(a**2)))
print("New RMS (m/year):", np.sqrt(np.mean(b**2)))

print("RMS difference (m/year):",
      np.sqrt(np.mean(difference**2)))

print("Relative RMS difference (%):",
      100 * np.sqrt(np.mean(difference**2))
      / np.sqrt(np.mean(a**2)))

print("Mean difference (m/year):", np.mean(difference))
print("Calculation complete!")
