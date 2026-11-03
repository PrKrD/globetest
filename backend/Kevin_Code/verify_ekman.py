
import numpy as np

# Load the mean wind stresses
data = np.load("mean_wind_stress.npz")

latitude = data["latitude"]
longitude = data["longitude"]
tau_x = data["tau_x"]
tau_y = data["tau_y"]

# Load our previous Ekman pumping calculation
previous = np.load("ekman_pumping_ocean.npz")
w_previous = previous["w_Ek"]  # m/s

# Physical constants
R = 6.371e6
Omega = 7.2921e-5
rho = 1025

# Convert geographical coordinates into radians
phi = np.deg2rad(latitude)
lam = np.deg2rad(longitude)

# Coriolis parameter and beta
f = 2 * Omega * np.sin(phi)
beta = 2 * Omega * np.cos(phi) / R

cos_phi = np.cos(phi)

# Calculate wind-stress curl on a sphere

# Longitude derivative (periodic)
dlambda = lam[1] - lam[0]

d_tauy_dlambda = (
    np.roll(tau_y, -1, axis=1)
    - np.roll(tau_y, 1, axis=1)
) / (2 * dlambda)

# Latitude derivative
tau_x_cos = tau_x * cos_phi[:, None]

d_tauxcos_dphi = np.gradient(
    tau_x_cos, phi, axis=0
)

# Spherical vertical component of wind-stress curl
with np.errstate(divide="ignore", invalid="ignore"):
    curl_tau = (
        d_tauy_dlambda - d_tauxcos_dphi
    ) / (R * cos_phi[:, None])

    # Alternative Ekman pumping calculation
    w_check = (
        curl_tau / (rho * f[:, None])
        + beta[:, None] * tau_x
        / (rho * f[:, None]**2)
    )

# Compare only open-ocean, midlatitude points
valid = (
    np.isfinite(w_previous)
    & np.isfinite(w_check)
    & (np.abs(latitude[:, None]) >= 15)
    & (np.abs(latitude[:, None]) <= 75)
)

# Convert to m/year for easy interpretation
seconds_per_year = 365.25 * 24 * 3600

a = w_previous[valid] * seconds_per_year
b = w_check[valid] * seconds_per_year

difference = a - b

print("Number of points checked:", len(a))

print("Original RMS (m/year):",
      np.sqrt(np.mean(a**2)))

print("RMS difference (m/year):",
      np.sqrt(np.mean(difference**2)))

print("Relative RMS difference (%):",
      100 * np.sqrt(np.mean(difference**2))
      / np.sqrt(np.mean(a**2)))

print("95th percentile absolute difference (m/year):",
      np.percentile(np.abs(difference), 95))
