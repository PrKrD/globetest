import numpy as np
import matplotlib.pyplot as plt

# 1. Load the averaged wind stresses from last time
data = np.load("mean_wind_stress.npz")

latitude = data["latitude"]
longitude = data["longitude"]
tau_x = data["tau_x"]
tau_y = data["tau_y"]

# 2. Physical constants
R = 6.371e6          # Earth's radius (m)
Omega = 7.2921e-5    # Earth's rotation rate (rad/s)
rho = 1025           # Seawater density (kg/m^3)

# Convert degrees to radians
phi = np.deg2rad(latitude)
lam = np.deg2rad(longitude)

# 3. Coriolis parameter
f = 2 * Omega * np.sin(phi)

# Exclude equatorial and near-polar regions
valid_lat = (np.abs(latitude) >= 10) & (np.abs(latitude) <= 85)
f = np.where(valid_lat, f, np.nan)

# 4. Horizontal Ekman transport (m^2/s)
Mx = tau_y / (rho * f[:, None])
My = -tau_x / (rho * f[:, None])

# 5. Spherical divergence of Ekman transport
cos_phi = np.cos(phi)

# Longitude derivatives are periodic around the globe
dlambda = lam[1] - lam[0]

dMx_dlambda = (
    np.roll(Mx, -1, axis=1) -
    np.roll(Mx, 1, axis=1)
) / (2 * dlambda)

# Latitude derivative, accounting for descending latitudes
My_cos = My * cos_phi[:, None]

dMycos_dphi = np.gradient(
    My_cos, phi, axis=0
)

# 6. Ekman pumping velocity (m/s)
w_Ek = (
    dMx_dlambda + dMycos_dphi
) / (R * cos_phi[:, None])

# 7. Convert from m/s to m/year
seconds_per_year = 365.25 * 24 * 3600
w_Ek_year = w_Ek * seconds_per_year

# Save for combining with the geostrophic term later
np.savez_compressed(
    "ekman_pumping.npz",
    latitude=latitude,
    longitude=longitude,
    w_Ek=w_Ek,
    w_Ek_m_per_year=w_Ek_year
)

# 8. Plot the result
valid = w_Ek_year[np.isfinite(w_Ek_year)]
colour_limit = np.percentile(np.abs(valid), 98)

plt.figure(figsize=(12, 5))

plt.pcolormesh(
    longitude,
    latitude,
    w_Ek_year,
    shading="auto",
    cmap="RdBu_r",
    vmin=-colour_limit,
    vmax=colour_limit
)

plt.colorbar(label="Ekman pumping velocity (m/year)")
plt.xlabel("Longitude (degrees east)")
plt.ylabel("Latitude (degrees north)")
plt.title("Mean Ekman Pumping (2005–2020)")

plt.tight_layout()
plt.savefig("ekman_pumping.png", dpi=200)
plt.show()

print("Ekman pumping array shape:", w_Ek.shape)
print("Minimum (m/year):", np.nanmin(w_Ek_year))
print("Maximum (m/year):", np.nanmax(w_Ek_year))
print("Calculation complete!")
