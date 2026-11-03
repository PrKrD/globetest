import numpy as np
from netCDF4 import Dataset

with Dataset("Data/Argo_mixedlayers_monthlyclim_04142022.nc") as ds:

    for name in ["mld_da_mean", "mld_dt_mean"]:

        var = ds.variables[name]

        # Read with automatic masking
        normal = var[:]
        print("\nVariable:", name)
        print("Automatically masked:",
              np.ma.count_masked(normal))

        # Read without automatic masking
        var.set_auto_mask(False)
        raw = np.asarray(var[:])

        # Only remove the explicit missing-value marker
        fill = var.getncattr("_FillValue")

        valid = (
            np.isfinite(raw)
            & (raw != fill)
            & (raw > 0)
        )

        values = raw[valid]

        print("Valid raw values:", len(values))
        print("Minimum:", np.min(values))
        print("Median:", np.median(values))
        print("Maximum:", np.max(values))
        print("Values deeper than 200 m:",
              np.sum(values > 200))

        # Count locations with complete monthly coverage
        monthly_valid = np.sum(valid, axis=2)

        print("Locations with 12 months:",
              np.sum(monthly_valid == 12))
