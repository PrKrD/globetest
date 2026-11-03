from netCDF4 import Dataset

with Dataset("Data/Argo_mixedlayers_monthlyclim_04142022.nc") as ds:

    print("DIMENSIONS:")
    for name, dim in ds.dimensions.items():
        print(name, len(dim))

    print("\nVARIABLES:")
    for name, var in ds.variables.items():
        print(name, var.shape,
              getattr(var, "units", ""))
