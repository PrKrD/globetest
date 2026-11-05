import numpy as np
from netCDF4 import Dataset, num2date

# Open our ERA5 dataset
with Dataset("Data/era5_wind_stress.nc", "r") as ds:

    latitude = ds.variables["latitude"][:]
    longitude = ds.variables["longitude"][:]

    time = ds.variables["valid_time"]
    dates = num2date(time[:], time.units,
                     calendar=getattr(time, "calendar", "standard"))

    # Get the wind-stress variables
    tau_x = ds.variables["iews"]
    tau_y = ds.variables["inss"]

    # Prepare arrays for adding the monthly values
    sum_x = np.zeros((len(latitude), len(longitude)))
    sum_y = np.zeros((len(latitude), len(longitude)))
    count_x = np.zeros(sum_x.shape, dtype=int)
    count_y = np.zeros(sum_y.shape, dtype=int)

    # Read and accumulate one month at a time
    for t in range(len(time)):
        x = np.ma.filled(tau_x[t, :, :], np.nan)
        y = np.ma.filled(tau_y[t, :, :], np.nan)

        valid_x = np.isfinite(x)
        valid_y = np.isfinite(y)

        sum_x[valid_x] += x[valid_x]
        sum_y[valid_y] += y[valid_y]

        count_x += valid_x
        count_y += valid_y

    # Calculate the mean at each geographical point
    mean_tau_x = np.divide(
        sum_x, count_x,
        out=np.full(sum_x.shape, np.nan),
        where=count_x > 0
    )

    mean_tau_y = np.divide(
        sum_y, count_y,
        out=np.full(sum_y.shape, np.nan),
        where=count_y > 0
    )

print("First month:", dates[0])
print("Last month:", dates[-1])

print("Mean eastward stress shape:", mean_tau_x.shape)
print("Mean northward stress shape:", mean_tau_y.shape)

print("Example mean eastward stress:", mean_tau_x[100, 200])
print("Example mean northward stress:", mean_tau_y[100, 200])

# Save the results for our next calculation
np.savez(
    "mean_wind_stress.npz",
    latitude=latitude,
    longitude=longitude,
    tau_x=mean_tau_x,
    tau_y=mean_tau_y
)
