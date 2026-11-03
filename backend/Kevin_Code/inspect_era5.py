from netCDF4 import Dataset, num2date

with Dataset("Data/era5_wind_stress.nc", "r") as ds:

    time = ds.variables["valid_time"]
    latitude = ds.variables["latitude"]
    longitude = ds.variables["longitude"]

    tau_x = ds.variables["iews"]
    tau_y = ds.variables["inss"]

    dates = num2date(time[:], time.units,
                     calendar=getattr(time, "calendar", "standard"))

    print("First date:", dates[0])
    print("Last date:", dates[-1])

    print("First latitude:", latitude[0])
    print("First longitude:", longitude[0])

    print("Eastward wind stress units:", tau_x.units)
    print("Northward wind stress units:", tau_y.units)

    print("Example eastward stress:", tau_x[0, 100, 200])
    print("Example northward stress:", tau_y[0, 100, 200])