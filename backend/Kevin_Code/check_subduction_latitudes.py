import numpy as np

data = np.load("holte_geostrophic_comparison.npz")

latitude = data["latitude"]
valid = data["common_valid"].astype(bool)

seconds_per_year = 365.25 * 24 * 3600

old = data["S_deboyer_m_per_s"] * seconds_per_year
new = data["S_holte_m_per_s"] * seconds_per_year

difference = new - old

# Group ocean grid cells by absolute latitude
bands = [
    (0, 10),
    (10, 30),
    (30, 50),
    (50, 70),
    (70, 90)
]

for lower, upper in bands:

    region = (
        valid
        & np.isfinite(difference)
        & (np.abs(latitude[:, None]) >= lower)
        & (np.abs(latitude[:, None]) < upper)
    )

    values = difference[region]

    if len(values) == 0:
        continue

    print(f"\nLatitude: {lower}°–{upper}°")

    print("Valid cells:", len(values))

    print("RMS difference:",
          np.sqrt(np.mean(values**2)), "m/year")

    print("Median absolute difference:",
          np.median(np.abs(values)), "m/year")

    print("98th percentile absolute difference:",
          np.percentile(np.abs(values), 98), "m/year")
