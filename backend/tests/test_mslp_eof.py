"""Scientific invariants and small dense references for streamed MSLP EOFs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from app.mslp_eof import (autocorrelation, compare_periods, latitude_weights,
                          monthly_anomalies, open_mslp, pressure_hpa, select_domain,
                          solve_eof, spatial_correlation, validate_grid, validate_months)


def pressure_field(n=72, descending=False, negative_lon=False):
    rng = np.random.default_rng(42)
    time = np.arange(np.datetime64("2000-01"), np.datetime64("2000-01") + n).astype("datetime64[ns]")
    lat = np.array([-90., -60., -30., -20., 0., 20., 30., 60., 90.])
    lon = np.arange(0, 360., 45.)
    if negative_lon:
        lon -= 180
    shape = (n, len(lat), len(lon))
    values = (101000 + 500*np.sin(np.arange(n)[:, None, None]*2*np.pi/12)
              + rng.normal(0, 100, shape))
    ds = xr.DataArray(values, dims=("time", "latitude", "longitude"),
                      coords={"time": time, "latitude": lat, "longitude": lon},
                      name="msl", attrs={"units": "Pa"})
    return ds.sortby("latitude", ascending=not descending)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("negative_lon", [False, True])
def test_domains_preserve_coordinates_and_include_boundaries(descending, negative_lon):
    field = pressure_field(descending=descending, negative_lon=negative_lon)
    for hemi, expected in (("north", [20, 30, 60, 90]), ("south", [-90, -60, -30, -20])):
        result = select_domain(field, hemi)
        np.testing.assert_array_equal(result.latitude, expected)
        np.testing.assert_array_equal(result.longitude, field.longitude)


@pytest.mark.parametrize("units,scale", [("Pa", .01), ("hPa", 1), ("mbar", 1)])
def test_pressure_conversion(units, scale):
    field = pressure_field()
    field.attrs["units"] = units
    result = pressure_hpa(field)
    np.testing.assert_allclose(result, field*scale)
    assert result.attrs["units"] == "hPa"


def test_reject_unknown_pressure_units():
    with pytest.raises(ValueError, match="units"):
        pressure_hpa(pressure_field().assign_attrs(units=""))


def test_climatology_and_optional_detrending():
    field = pressure_hpa(pressure_field())
    months = field.time.dt.month.values
    anomalies, climatology = monthly_anomalies(field.values, months)
    np.testing.assert_allclose(anomalies + climatology[months-1], field, atol=1e-12)
    for month in range(1, 13):
        np.testing.assert_allclose(anomalies[months == month].mean(axis=0), 0, atol=1e-12)
    trending = field.values + 2*np.arange(len(field))[:, None, None]
    primary, _ = monthly_anomalies(trending, months)
    detrended, _ = monthly_anomalies(trending, months, detrend=True)
    t = np.arange(len(field)) - (len(field)-1)/2
    assert np.max(np.abs(np.einsum("t,tij->ij", t, primary))) > 1
    np.testing.assert_allclose(np.einsum("t,tij->ij", t, detrended), 0, atol=1e-8)


def test_latitude_area_weights():
    np.testing.assert_allclose(latitude_weights(np.array([-90., -60., 0., 60., 90.])),
                               [0, np.sqrt(.5), 1, np.sqrt(.5), 0], atol=1e-15)


@pytest.mark.parametrize("block", [1, 3, 10])
def test_solver_matches_dense_svd_reconstruction_and_regression(block):
    field = select_domain(pressure_hpa(pressure_field()), "north")
    result = solve_eof(field, n_modes=6, block_latitudes=block)
    anomaly, _ = monthly_anomalies(field.values, field.time.dt.month.values)
    weighted = (anomaly * latitude_weights(field.latitude.values)[None, :, None]).reshape(len(field), -1)
    u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    eigenvalues = singular**2 / (len(field)-1)
    np.testing.assert_allclose(result.eigenvalue, eigenvalues[:6], rtol=1e-10)
    np.testing.assert_allclose(result.explained_variance_percent, 100*eigenvalues[:6]/eigenvalues.sum())
    np.testing.assert_allclose(result.standardized_pc.std("time", ddof=1), 1)
    np.testing.assert_allclose(result.pc.var("time", ddof=1), result.eigenvalue)
    modes = result.weighted_eof.values.reshape(6, -1)
    np.testing.assert_allclose(modes @ modes.T, np.eye(6), atol=1e-12)
    reconstruction = result.pc.values @ modes
    np.testing.assert_allclose(reconstruction, (u[:, :6]*singular[:6]) @ vt[:6], atol=1e-10)
    regression = result.standardized_pc.values.T @ anomaly.reshape(len(field), -1)/(len(field)-1)
    np.testing.assert_allclose(result.regression_map.values.reshape(6, -1), regression, atol=1e-11)
    reg_reconstruction = result.standardized_pc.values @ result.regression_map.values.reshape(6, -1)
    np.testing.assert_allclose(reg_reconstruction.reshape(anomaly.shape) *
                               latitude_weights(field.latitude.values)[None, :, None],
                               reconstruction.reshape(anomaly.shape), atol=1e-10)
    assert result.regression_map.attrs["units"] == "hPa"
    assert float(result.cumulative_variance_percent[-1]) < 100


def test_known_separable_modes_and_full_rank_reconstruction():
    field = select_domain(pressure_hpa(pressure_field(n=120)), "north")
    rng = np.random.default_rng(1)
    time = rng.normal(size=(len(field), 2))
    for month in range(12):
        time[month::12] -= time[month::12].mean(axis=0)
    time, _ = np.linalg.qr(time)
    spatial = rng.normal(size=(2, field.sizes["latitude"], field.sizes["longitude"]))
    spatial[:, -1] = 0  # exact pole contributes no measure
    spatial_flat = spatial.reshape(2, -1) * np.repeat(latitude_weights(field.latitude.values), 8)
    q, _ = np.linalg.qr(spatial_flat.T)
    weighted = (time * [10, 5]) @ q.T
    weights = latitude_weights(field.latitude.values)
    field.values[:] = np.divide(weighted.reshape(field.shape), weights[None, :, None],
                                 out=np.zeros(field.shape), where=weights[None, :, None] > 0) + 1000
    result = solve_eof(field, n_modes=2, block_latitudes=2)
    np.testing.assert_allclose(result.eigenvalue, np.array([100, 25])/119, atol=1e-12)
    np.testing.assert_allclose(result.explained_variance_percent, [80, 20], atol=1e-9)
    anomaly, _ = monthly_anomalies(field.values, field.time.dt.month.values)
    reconstructed = result.standardized_pc.values @ result.regression_map.values.reshape(2, -1)
    np.testing.assert_allclose(reconstructed.reshape(field.shape), anomaly, atol=1e-10)


def test_mode_matching_sign_alignment_and_subspace_angles():
    full = solve_eof(select_domain(pressure_hpa(pressure_field()), "north"), 3)
    modern = full.isel(mode=[2, 0, 1]).assign_coords(mode=[1, 2, 3]).copy(deep=True)
    modern["regression_map"] *= xr.DataArray([-1, 1, -1], dims="mode", coords={"mode": modern.mode})
    modern["weighted_eof"] *= xr.DataArray([-1, 1, -1], dims="mode", coords={"mode": modern.mode})
    modern["standardized_pc"] *= xr.DataArray([-1, 1, -1], dims="mode", coords={"mode": modern.mode})
    comparison = compare_periods(full, modern)
    np.testing.assert_array_equal(comparison.matched_modern_mode, [2, 3, 1])
    np.testing.assert_array_equal(comparison.modern_sign, [1, -1, -1])
    np.testing.assert_allclose(comparison.aligned_spatial_correlation, 1)
    np.testing.assert_allclose(comparison.common_pc_correlation, 1)
    np.testing.assert_allclose(comparison.principal_angle_degrees, 0, atol=3e-6)


def test_spatial_correlation_is_weighted_and_centred():
    a = np.array([[1., 3.], [4., 2.]])
    b = np.array([[1., 4.], [0., 9.]])
    lat = np.array([20., 60.])
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    x, y = a - np.average(a, weights=weights), b - np.average(b, weights=weights)
    expected = np.sum(weights*x*y)/np.sqrt(np.sum(weights*x*x)*np.sum(weights*y*y))
    assert spatial_correlation(a, b, lat) == pytest.approx(expected)
    assert spatial_correlation(a, 3*a+20, lat) == pytest.approx(1)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reverse", "short"])
def test_irregular_months_rejected(kind):
    time = pressure_field().time
    if kind == "missing":
        time = time.isel(time=[i for i in range(len(time)) if i != 19])
    elif kind == "duplicate":
        time = xr.concat([time, time[-1:]], dim="time")
    elif kind == "reverse":
        time = time[::-1]
    else:
        time = time[:12]
    with pytest.raises(ValueError, match="monthly"):
        validate_months(time)


def test_incomplete_final_year_retained():
    field = select_domain(pressure_hpa(pressure_field(n=70)), "south")
    result = solve_eof(field, 2)
    assert result.attrs["last_month"] == "2005-10"
    assert result.sizes["time"] == 70


@pytest.mark.parametrize("version", [False, True])
def test_single_multifile_and_expver_merge(tmp_path, version):
    field = pressure_field(descending=True).sel(latitude=[90, 60, 30, 0, -30, -60, -90])
    field = field.rename(latitude="lat", longitude="lon", time="valid_time")
    if version:
        field = xr.concat([field.where(field.valid_time.dt.year < 2003),
                           field.where(field.valid_time.dt.year >= 2003)], dim="expver")
        field = field.assign_coords(expver=[1, 5])
    field.isel(valid_time=slice(0, 36)).to_dataset().to_netcdf(tmp_path / "a.nc", engine="h5netcdf")
    field.isel(valid_time=slice(36, None)).to_dataset().to_netcdf(tmp_path / "b.nc", engine="h5netcdf")
    opened, metadata = open_mslp([str(tmp_path / "*.nc")])
    try:
        assert len(metadata["files"]) == 2
        assert metadata["first_month"] == "2000-01"
        result = solve_eof(select_domain(opened, "north"), 2)
        assert result.sizes["time"] == 72
    finally:
        opened.close()


def test_expver_conflicts_and_missing_values_rejected():
    field = select_domain(pressure_hpa(pressure_field()), "north")
    versions = xr.concat([field, field+1], dim="expver").assign_coords(expver=[1, 5])
    with pytest.raises(ValueError, match="Conflicting"):
        solve_eof(versions, 2)
    field.values[5, 1, 1] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        solve_eof(field, 2)


def test_grid_validation_rejects_duplicate_seam_and_wrong_primary_resolution():
    field = pressure_field().sortby("latitude")
    # Fixture has deliberately irregular latitude bands, unlike real ERA5.
    with pytest.raises(ValueError, match="regular"):
        validate_grid(field)
    regular = field.sel(latitude=[-90, -60, -30, 0, 30, 60, 90])
    validate_grid(regular)
    with pytest.raises(ValueError, match="native"):
        validate_grid(regular, .25)
    seam = xr.concat([regular, regular.isel(longitude=[0]).assign_coords(longitude=[360.])], dim="longitude")
    with pytest.raises(ValueError, match="globe"):
        validate_grid(seam)


def test_acf_matches_direct_calculation():
    pc = np.random.default_rng(123).normal(size=(100, 3))
    acf, crossing = autocorrelation(pc, 24)
    x = pc-pc.mean(axis=0)
    np.testing.assert_allclose(acf[1], np.sum(x[:-1]*x[1:], axis=0)/np.sum(x*x, axis=0))
    assert acf.shape == (25, 3)
    assert np.all(crossing >= 1)


def test_pattern_summary_and_polar_map_export(tmp_path):
    from app.mslp_eof_diagnostics import pattern_structure, plot_maps
    result = solve_eof(select_domain(pressure_hpa(pressure_field()), "north"), 2)
    structure = pattern_structure(result)
    assert np.all((structure["zonal_loading_energy_fraction"] >= 0) &
                  (structure["zonal_loading_energy_fraction"] <= 1))
    path = tmp_path / "polar.png"
    plot_maps(result, "north", path, dpi=40, coastlines=False)
    assert path.stat().st_size > 1000
    import matplotlib.pyplot as plt
    assert plt.imread(path).shape[:2] == (172, 520)


def test_declared_radian_coordinates_rejected():
    field = pressure_field().sel(latitude=[-90, -60, -30, 0, 30, 60, 90])
    field.latitude.attrs["units"] = "radians"
    with pytest.raises(ValueError, match="units"):
        validate_grid(field)


def test_inspected_interpretation_is_bound_to_exact_computed_results(tmp_path):
    from app.mslp_eof_diagnostics import interpretation_scope, write_report
    result = solve_eof(select_domain(pressure_hpa(pressure_field()), "north"), 2)
    results = {("north", 2000): result}
    metadata = {"files": [], "first_month": "2000-01", "last_month": "2005-12",
                "grid_spacing_deg": 45}
    (tmp_path / "interpretation_notes.md").write_text("Inspected discussion for this exact basis.")
    report = write_report(results, {}, metadata, tmp_path)
    assert "Inspected discussion" not in report.read_text()
    (tmp_path / "interpretation_scope.json").write_text(json.dumps(interpretation_scope(results, metadata)))
    write_report(results, {}, metadata, tmp_path)
    assert "Inspected discussion" in report.read_text()
    result.regression_map.values[0, 0, 0] += .1
    write_report(results, {}, metadata, tmp_path)
    assert "Inspected discussion" not in report.read_text()


def test_cli_end_to_end_native_grid_retained_in_synthetic_validation(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyse_mslp_eof.py"
    spec = importlib.util.spec_from_file_location("analyse_mslp_eof", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(99)
    time = np.arange(np.datetime64("1940-01"), np.datetime64("1981-11")).astype("datetime64[ns]")
    lat = np.arange(-90., 91., 10.)
    field = xr.DataArray(101000 + rng.normal(0, 100, (len(time), len(lat), 12)),
                         dims=("time", "latitude", "longitude"),
                         coords={"time": time, "latitude": lat, "longitude": np.arange(0., 360., 30.)},
                         name="msl", attrs={"units": "Pa"})
    source = tmp_path / "synthetic.nc"
    field.to_dataset().to_netcdf(source, engine="h5netcdf")
    output = tmp_path / "output"
    args = module.parse_args(["--input", str(source), "--output-dir", str(output),
                              "--allow-other-resolution", "--no-maps", "--dpi", "40"])
    report = module.run(args)
    assert report.exists()
    assert "1981-10" in report.read_text()
    assert "Summary of EOF analyses" in report.read_text()
    assert "AO/SAM index validation and sign conventions" in report.read_text()
    summary = json.loads((output / "reports" / "summary.json").read_text())
    assert len(summary["analyses"]) == 4
    assert summary["index_validation"] is None
    assert (output / "comparisons" / "north_matching.csv").exists()
    assert (output / "comparisons" / "south_matching.csv").exists()
    for hemi in ("northern", "southern"):
        for start in (1940, 1979):
            with xr.open_dataset(output / f"{hemi}_hemisphere" / f"{start}_1981" / "eof.nc") as ds:
                assert ds.sizes["mode"] == 6
                assert ds.sizes["longitude"] == 12
                assert ds.attrs["last_month"] == "1981-10"
    args.render_only = True
    assert module.run(args).exists()
    args.render_only = False
    args.reuse_existing = True
    args.n_modes = 5
    with pytest.raises(ValueError, match="mismatch"):
        module.run(args)
    args.n_modes = 6
    args.allow_other_resolution = False
    with pytest.raises(ValueError, match="native"):
        module.run(args)
