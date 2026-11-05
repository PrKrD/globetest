
from netCDF4 import Dataset

with Dataset("Data/era5_mean_wind_stress.nc", "r") as ds:

    print("Dimensions:")
    for name, dim in ds.dimensions.items():
        print(name, len(dim))

    print("\nVariables:")
    for name, var in ds.variables.items():
        print(name, var.shape, getattr(var, "units", ""))
