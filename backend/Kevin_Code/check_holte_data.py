import numpy as np
from netCDF4 import Dataset

with Dataset("Data/Argo_mixedlayers_monthlyclim_04142022.nc") as ds:

    for name in ["mld_da_mean", "mld_dt_mean"]:

        var = ds.variables[name]
        data = np.ma.masked_invalid(var[:])
        data = np.ma.masked_less_equal(data, 0)

        values = data.compressed()

        print("\nVARIABLE:", name)

        # Inspect metadata
        print("Attributes:")
        for attr in var.ncattrs():
            print(attr, ":", var.getncattr(attr))

        # Inspect all valid monthly measurements
        print("Total valid values:", len(values))
        print("Minimum:", np.min(values))
        print("Median:", np.median(values))
        print("95th percentile:", np.percentile(values, 95))
        print("Maximum:", np.max(values))

        print("Values deeper than 200 m:",
              np.sum(values > 200))

        # Check data availability in each month
        print("\nMonthly coverage:")
        for month in range(12):
            monthly = data[:, :, month]
            print(
                f"Month {month+1}:",
                np.ma.count(monthly),
                "valid grid points"
            )
