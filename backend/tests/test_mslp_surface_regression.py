"""Scientific invariants for common-sample surface regression."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from app.mslp_surface_regression import (Design, SPECS, SurfaceInputs,
    cell_areas, common_months, load_mask, load_pc, month_ids, process_variable, rate_factor)
from app.mslp_eof import monthly_anomalies
from app.mslp_surface_regression_diagnostics import numerical_summary, turbulent_consistency


def synthetic(seed=42, t=120, p=9):
    rng = np.random.default_rng(seed)
    months = np.arange(t)%12+1
    x, _ = monthly_anomalies(rng.normal(size=(t, 12)), months)
    q, _ = np.linalg.qr(x)
    x = q*np.sqrt(t-1)
    beta = rng.normal(size=(12, p))
    noise, _ = monthly_anomalies(rng.normal(size=(t, p)), months)
    noise -= q @ (q.T @ noise)
    return x, x@beta+noise, beta


def test_orthogonal_known_coefficients_predictions_and_variance():
    x, y, beta = synthetic()
    design = Design(x)
    individual, joint, comparison = design.fit(y)
    np.testing.assert_allclose(individual["coefficient"], beta, atol=1e-13)
    np.testing.assert_allclose(joint["coefficient"], beta, atol=1e-13)
    np.testing.assert_allclose(comparison["coefficient_difference"], 0, atol=1e-13)
    np.testing.assert_allclose(comparison["reconstruction_max_absolute_difference"], 0, atol=1e-13)
    np.testing.assert_allclose(comparison["r_squared_sum_difference"], 0, atol=1e-13)
    assert design.diagnostics()["orthogonal"]
    assert np.all(np.diff(joint["nested_r_squared"], axis=0) >= -1e-13)


def test_correlated_predictors_and_algebraic_diagnostic():
    x, y, _ = synthetic()
    x[:, 1] += .7*x[:, 0]
    individual, joint, comparison = Design(x).fit(y)
    assert not Design(x).diagnostics()["orthogonal"]
    assert np.max(abs(individual["coefficient"]-joint["coefficient"])) > .1
    np.testing.assert_allclose(comparison["correlated_design_identity_error"], 0, atol=1e-13)
    assert np.max(abs(comparison["r_squared_sum_difference"])) > .01


def test_subset_detects_lost_orthogonality_without_rescaling():
    x, y, _ = synthetic()
    subset = x[:-7]
    d = Design(subset)
    assert not d.diagnostics()["orthogonal"]
    np.testing.assert_array_equal(d.original, subset)
    individual, joint, _ = d.fit(y[:-7])
    assert np.max(abs(individual["coefficient"]-joint["coefficient"])) > .001


@pytest.mark.parametrize("kind", list(SPECS))
def test_rate_accumulated_units_signs_and_parameter_identity(kind):
    rate_id, acc_id, names, acc_name, units = SPECS[kind]
    f = xr.DataArray([-86400., 86400.], name=names[0], attrs={"GRIB_paramId": rate_id, "units": units})
    assert rate_factor(f, kind, {}) == 1
    f.name = acc_name
    f.attrs.update(GRIB_paramId=acc_id, units="J m**-2" if kind in ("latent", "sensible") else "N m**-2 s")
    factor = rate_factor(f, kind, {"history": '{"stream": ["moda"]}'})
    np.testing.assert_array_equal(f.values*factor, [-1, 1])
    with pytest.raises(ValueError, match="provenance"):
        rate_factor(f, kind, {})
    f.attrs["GRIB_paramId"] = 229
    with pytest.raises(ValueError, match="wrong GRIB"):
        rate_factor(f, kind, {})


def test_calendar_alignment_and_missing_duplicate_detection():
    dates = np.arange("1979-01", "1982-01", dtype="datetime64[M]").astype("datetime64[h]")
    a = xr.DataArray(dates, dims="time")
    b = xr.DataArray(dates+np.timedelta64(6, "h"), dims="time")
    np.testing.assert_array_equal(month_ids(a), month_ids(b))
    for values in (dates[np.arange(len(dates)) != 5], np.insert(dates, 5, dates[5])):
        with pytest.raises(ValueError):
            month_ids(xr.DataArray(values, dims="time"))
    np.testing.assert_array_equal(common_months([month_ids(a), month_ids(b)[2:]]), month_ids(a)[2:])


def test_dense_reference_intercepts_original_scaling_and_constants():
    x, y, _ = synthetic()
    x = (x+np.arange(12)) * np.arange(1, 13)
    y += 9
    y = np.column_stack((y, np.zeros(len(y)), np.full(len(y), 3.)))
    _, joint, _ = Design(x).fit(y)
    reference = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    np.testing.assert_allclose(joint["coefficient"], reference[1:], atol=1e-12)
    np.testing.assert_allclose(joint["intercept"], reference[0], atol=1e-12)
    assert np.isnan(joint["r_squared"][-2:]).all()
    assert joint["constant_target"][-2:].all()
    with pytest.raises(ValueError, match="finite"):
        Design(x).fit(np.full_like(y, np.nan))
    with pytest.raises(ValueError, match="Rank"):
        Design(np.ones_like(x))


def test_spherical_areas_and_partition():
    lat, lon = np.linspace(90, -90, 721), np.arange(0, 360, .25)
    area = cell_areas(lat, lon)
    np.testing.assert_allclose(area.sum(), 4*np.pi*6371000.**2, rtol=1e-14)
    partition = sum(cell_areas(lat, lon, bounds) for bounds in ((-90, -20), (-20, 20), (20, 90)))
    np.testing.assert_allclose(area, partition, atol=.01)


def make_files(tmp_path):
    x, y, _ = synthetic(t=48, p=12)
    dates = np.arange("1979-01", "1983-01", dtype="datetime64[M]").astype("datetime64[ns]")
    lat, lon = np.array([90., 30., -30., -90.]), np.array([0., 120., 240.])
    targets = {}
    for index, (kind, (param, _, aliases, _, units)) in enumerate(SPECS.items()):
        values = y.reshape(48, 4, 3)*(index+1)
        values[:, 1, 1] = 0
        values[5, 2, 1] = np.nan
        targets[aliases[0]] = (("valid_time", "latitude", "longitude"), values,
                              {"units": units, "GRIB_paramId": param})
    ds = xr.Dataset(targets, coords={"valid_time": dates+np.timedelta64(6, "h"), "latitude": lat, "longitude": lon})
    source = tmp_path/"surface.nc"
    ds.to_netcdf(source, engine="h5netcdf")
    mask = np.zeros((4, 3)); mask[1, 0] = 1; mask[2, 0] = .2
    m = xr.Dataset({"lsm": (("valid_time", "latitude", "longitude"), np.broadcast_to(mask, (48, 4, 3)))}, coords=ds.coords)
    maskpath = tmp_path/"mask.nc"
    m.to_netcdf(maskpath, engine="h5netcdf")
    pcpaths = []
    for hemi in ("north", "south"):
        p = tmp_path/f"{hemi}.nc"
        xr.Dataset({"standardized_pc": (("time", "mode"), x)}, coords={"time": dates, "mode": np.arange(1, 13)},
                   attrs={"hemisphere": hemi, "detrended": 0, "native_resolution_primary": 1}).to_netcdf(p, engine="h5netcdf")
        pcpaths.append(p)
    return source, maskpath, pcpaths, x


def test_chunked_io_missing_masks_linearity_and_summaries(tmp_path):
    source, maskpath, pcpaths, x = make_files(tmp_path)
    inputs = SurfaceInputs([str(source)], native=False)
    try:
        ref = inputs.fields["latent"]
        mask, meta = load_mask(maskpath, ref, tile_lat=2)
        assert meta["maximum_temporal_change"] == 0
        designs = {h: Design(x) for h in ("north", "south")}
        for tile, output in (((2, 2), tmp_path/"a"), ((4, 3), tmp_path/"b")):
            for kind in ("latent", "sensible", "turbulent"):
                fields = [inputs.fields["latent"], inputs.fields["sensible"]] if kind == "turbulent" else [inputs.fields[kind]]
                result = process_variable(kind, fields, mask, designs, output,
                    {"n_samples": 48, "first_month": "1979-01", "last_month": "1982-12"}, tile, progress=lambda x: None)
                assert result["incomplete_ocean_cells"] == 1
                assert result["valid_ocean_cells"] == 10
            summary = numerical_summary(output, ["latent", "sensible", "turbulent"], {h: d.diagnostics() for h, d in designs.items()})
            assert summary["all_coefficient_agreement_tests_pass"]
            assert summary["all_reconstruction_tests_pass"]
            assert all(r["agreement_pass"] for r in turbulent_consistency(output))
        with xr.open_dataset(tmp_path/"a/northern_hemisphere/joint/latent.nc") as a, xr.open_dataset(tmp_path/"b/northern_hemisphere/joint/latent.nc") as b:
            xr.testing.assert_allclose(a, b)
            assert np.isnan(a.coefficient.values[:, 1, 0]).all()
            assert np.isnan(a.coefficient.values[:, 2, 1]).all()
            assert a.finite_sample_count.values[2, 1] == 47
            assert np.isnan(a.r_squared.values[1, 1])
        ids, loaded, _ = load_pc(pcpaths[0], "north")
        np.testing.assert_array_equal(loaded, x)
        assert len(ids) == 48
    finally:
        inputs.close()


def test_loader_rejects_ambiguous_inputs_changed_mask_and_wrong_pc(tmp_path):
    source, maskpath, pcpaths, _ = make_files(tmp_path)
    with xr.open_dataset(source) as raw:
        ds = raw.load()
    ds["mslhf"] = ds.avg_slhtf.copy()
    ds.to_netcdf(tmp_path/"ambiguous.nc", engine="h5netcdf")
    with pytest.raises(ValueError, match="Ambiguous"):
        SurfaceInputs([str(tmp_path/"ambiguous.nc")], native=False)
    with xr.open_dataset(maskpath) as raw:
        m = raw.load()
    m.lsm.values[1, 0, 0] = .1
    m.to_netcdf(tmp_path/"changed.nc", engine="h5netcdf")
    with pytest.raises(ValueError, match="changes"):
        load_mask(tmp_path/"changed.nc", ds.avg_slhtf.rename(valid_time="time"), tile_lat=1)
    with pytest.raises(ValueError, match="hemisphere"):
        load_pc(pcpaths[0], "south")


def test_mask_float32_precision_allowed_but_threshold_crossing_rejected(tmp_path):
    source, maskpath, _, _ = make_files(tmp_path)
    with xr.open_dataset(maskpath) as raw:
        m = raw.load()
    m.lsm.values[:, 0, 0] = .3
    m.lsm.values[1, 0, 0] += np.finfo(np.float32).eps
    m.to_netcdf(tmp_path/"precision.nc", engine="h5netcdf")
    with xr.open_dataset(source) as reference:
        load_mask(tmp_path/"precision.nc", reference.avg_slhtf.rename(valid_time="time"))
        m.lsm.values[:, 0, 0] = .5
        m.lsm.values[1, 0, 0] -= np.finfo(np.float32).eps
        m.to_netcdf(tmp_path/"crossing.nc", engine="h5netcdf")
        with pytest.raises(ValueError, match="changes"):
            load_mask(tmp_path/"crossing.nc", reference.avg_slhtf.rename(valid_time="time"))


def test_cli_end_to_end_and_overwrite_guard(tmp_path):
    source, maskpath, pcs, _ = make_files(tmp_path)
    script = Path(__file__).resolve().parents[1]/"scripts/analyse_mslp_surface_regression.py"
    spec = importlib.util.spec_from_file_location("surface_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    argv = ["--input", str(source), "--mask", str(maskpath), "--nh-pcs", str(pcs[0]),
            "--sh-pcs", str(pcs[1]), "--output-dir", str(tmp_path/"output"), "--allow-other-resolution",
            "--turbulent", "--no-maps", "--tile-latitudes", "2", "--tile-longitudes", "2"]
    result = cli.run(cli.parse_args(argv))
    assert result["all_coefficient_agreement_tests_pass"]
    assert (tmp_path/"output/reports/surface_regression_report.md").exists()
    with pytest.raises(FileExistsError):
        cli.run(cli.parse_args(argv))


def test_native_projected_mesh_preserves_values_and_splits_dateline():
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from app.mslp_surface_regression_diagnostics import map_panel, _MESH_CACHE
    lat, lon = np.array([90., 0., -90.]), np.array([0., 90., 180., 270.])
    data = np.arange(12.).reshape(3, 4)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = map_panel(ax, data, lat, lon, "Native cells", coastlines=False)
    np.testing.assert_array_equal(mesh.get_array(), data[:, [2, 3, 0, 1, 2]])
    assert np.isfinite(mesh.get_coordinates()).all()
    assert len(_MESH_CACHE) >= 1
    plt.close(fig)


def test_stress_arrow_coordinates_preserve_components_and_omit_poles():
    from app.mslp_surface_regression_diagnostics import quiver_samples
    lat, lon = np.array([90., 45., 0., -45., -90.]), np.array([0., 90., 180., 270.])
    u, v = np.ones((5, 4)), np.full((5, 4), -2.)
    u[2, 1] = np.nan
    xx, yy, uu, vv = quiver_samples(lat, lon, u, v, 1, 1)
    assert len(xx) == 11
    assert (np.abs(yy) < 90).all()
    assert ((xx >= -180) & (xx < 180)).all()
    np.testing.assert_array_equal(uu, np.ones(11))
    np.testing.assert_array_equal(vv, np.full(11, -2.))
