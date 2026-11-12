"""Numerical acceptance for annual fixed-model and seasonal joint experiments."""
from pathlib import Path
import json

import numpy as np
import pytest
import xarray as xr

from app.mslp_eof import monthly_anomalies
from app.mslp_surface_regression import Design, COUNTS, SPECS, cell_areas
from app.mslp_surface_sensitivity import (
    annual_groups, annual_mean, evaluate, seasonal_designs,
    RegionalAccumulator, run_sensitivity,
)


def predictors(seed=13, years=12):
    rng = np.random.default_rng(seed)
    raw, _ = monthly_anomalies(rng.normal(size=(years*12, 12)), np.tile(np.arange(1, 13), years))
    q, _ = np.linalg.qr(raw)
    return q*np.sqrt(len(q)-1), rng


def test_annual_complete_years_weights_and_intercept_identity():
    ids = np.arange(1979*12, 2026*12+10)
    years, groups = annual_groups(ids)
    np.testing.assert_array_equal(years, np.arange(1979, 2026))
    assert groups.shape == (47, 12)
    rng = np.random.default_rng(9)
    pc, beta, intercept = rng.normal(size=(len(ids), 12)), rng.normal(size=(12, 3)), np.array([2., -8., .5])
    np.testing.assert_allclose(annual_mean(intercept+pc@beta, groups), intercept+annual_mean(pc, groups)@beta, atol=2e-15)
    _, missing = annual_groups(np.delete(ids, 8))
    assert len(missing) == 46
    with pytest.raises(ValueError, match="unique"):
        annual_groups(np.insert(ids, 5, ids[5]))
    y = np.arange(len(ids))[:, None]
    np.testing.assert_array_equal(annual_mean(y, groups)[:, 0], y[groups, 0].mean(1))


def test_known_seasonal_coefficients_intercepts_and_correlated_nested_fits():
    x, rng = predictors()
    x[:, 1] += .8*x[:, 0]
    months = np.tile(np.arange(1, 13), len(x)//12)
    designs, diagnostics = seasonal_designs(x, months)
    for season, (ix, design) in designs.items():
        beta, intercept = rng.normal(size=(12, 5)), rng.normal(size=5)
        y = intercept+x[ix]@beta
        fits = dict((k, (b, a)) for k, b, a in design.joint_fits(y))
        np.testing.assert_allclose(fits[12][0], beta, atol=2e-14)
        np.testing.assert_allclose(fits[12][1], intercept, atol=2e-14)
        for k in COUNTS:
            dense = np.linalg.lstsq(np.column_stack((np.ones(len(ix)), x[ix, :k])), y, rcond=None)[0]
            np.testing.assert_allclose(fits[k][1], dense[0], atol=2e-14)
            np.testing.assert_allclose(fits[k][0], dense[1:], atol=2e-14)
        assert not np.allclose(fits[3][0], fits[12][0][:3])
        assert diagnostics[season]["augmented_rank"] == 13


def test_seasonal_subsets_destroy_orthogonality_without_changing_scaling():
    x, _ = predictors()
    np.testing.assert_allclose(np.corrcoef(x.T), np.eye(12), atol=1e-14)
    designs, diagnostics = seasonal_designs(x, np.tile(np.arange(1, 13), len(x)//12))
    assert max(d["maximum_offdiagonal_correlation"] for d in diagnostics.values()) > .1
    for ix, design in designs.values():
        np.testing.assert_array_equal(design.original, x[ix])


def test_constant_relationship_is_recovered_and_turbulent_linearity():
    x, rng = predictors()
    designs, _ = seasonal_designs(x, np.tile(np.arange(1, 13), len(x)//12))
    latent, sensible = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    for ix, design in designs.values():
        l, h = 2+x[ix]@latent, -3+x[ix]@sensible
        lf, hf, tf = [dict((k, (b, a)) for k, b, a in design.joint_fits(y)) for y in (l, h, l+h)]
        np.testing.assert_allclose(lf[12][0], latent, atol=1e-14)
        np.testing.assert_allclose(hf[12][0], sensible, atol=1e-14)
        for k in COUNTS:
            np.testing.assert_allclose(tf[k][0], lf[k][0]+hf[k][0], atol=1e-14)
            np.testing.assert_allclose(tf[k][1], lf[k][1]+hf[k][1], atol=1e-14)
    # The variance denominator includes cross-covariance, so component R² is not additive.
    y = x@latent
    a, b = evaluate(y, .5*y), evaluate(2*y, y)
    np.testing.assert_allclose(a["r_squared"], b["r_squared"])
    assert not np.allclose(a["r_squared"]+a["r_squared"], b["r_squared"])


def test_prediction_r_squared_negative_constants_bias_and_pooled_seasons():
    y = np.array([[0., 2.], [1., 2.], [2., 2.], [3., 2.]])
    p = y+4
    metrics = evaluate(y, p)
    assert metrics["r_squared"][0] < 0
    assert metrics["correlation"][0] == pytest.approx(1)
    assert np.isnan(metrics["r_squared"][1])
    assert metrics["constant_target"][1]
    np.testing.assert_allclose(metrics["rmse"], 4)
    np.testing.assert_allclose(metrics["residual_variance"], 0)
    np.testing.assert_allclose(metrics["bias"], 4)
    yy = np.r_[[-10., -9., -8.], [10., 12., 14.]].reshape(-1, 1)
    pp = yy+1
    pooled = evaluate(yy, pp)["r_squared"][0]
    averaged = np.mean([evaluate(yy[ix], pp[ix])["r_squared"][0] for ix in (slice(0, 3), slice(3, 6))])
    assert pooled != pytest.approx(averaged)


def test_regional_sse_sst_and_chunk_independence():
    rng = np.random.default_rng(8)
    y, prediction = rng.normal(size=(24, 6)), rng.normal(size=(24, 6))
    weights = np.arange(1., 7.)
    key = ("north", "latent", "year_round", 12, "monthly_full", "global_ocean", "primary")
    outputs = []
    for chunks in ((slice(None),), (slice(0, 2), slice(2, 6))):
        acc = RegionalAccumulator()
        for chunk in chunks:
            acc.add(key, y[:, chunk], prediction[:, chunk], evaluate(y[:, chunk], prediction[:, chunk]), weights[chunk])
        rows, _ = acc.finish(np.arange(1979*12, 1981*12), np.array([1979, 1980]))
        outputs.append(rows[0])
    m = evaluate(y, prediction)
    assert outputs[0]["total_variance_fraction_explained"] == pytest.approx(1-np.sum(weights*m["sse"])/np.sum(weights*m["sst"]))
    assert outputs[0]["area_weighted_rmse"] == pytest.approx(np.sqrt(np.sum(weights*m["sse"])/(24*weights.sum())))
    for key in outputs[0]:
        if isinstance(outputs[0][key], (int, float)):
            assert outputs[0][key] == pytest.approx(outputs[1][key])


def test_constant_cells_still_contribute_prediction_error_to_field_r_squared():
    y = np.column_stack((np.arange(24.), np.full(24, 3.)))
    prediction = y.copy()
    prediction[:, 1] += 5
    acc = RegionalAccumulator()
    key = ("north", "latent", "year_round", 12, "monthly_full", "global_ocean", "primary")
    acc.add(key, y, prediction, evaluate(y, prediction), np.ones(2))
    rows, _ = acc.finish(np.arange(1979*12, 1981*12), np.array([1979, 1980]))
    expected = 1-np.sum((prediction-y)**2)/np.sum((y-y.mean(0))**2)
    assert rows[0]["total_variance_fraction_explained"] == pytest.approx(expected)
    assert rows[0]["mean_local_r_squared"] == pytest.approx(1)


def make_baseline(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1]/"scripts/analyse_mslp_surface_regression.py"
    spec = importlib.util.spec_from_file_location("sensitivity_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    x, rng = predictors()
    time = np.arange("1979-01", "1991-01", dtype="datetime64[M]").astype("datetime64[ns]")
    lat, lon = [90., 30., -30., -90.], [0., 120., 240.]
    variables = {}
    for kind, (param, _, names, _, units) in SPECS.items():
        y = x@rng.normal(size=(12, 12))+rng.normal(size=(144, 12))
        y[:, 1] = 0
        variables[names[0]] = (("valid_time", "latitude", "longitude"), y.reshape(144, 4, 3), {"GRIB_paramId": param, "units": units})
    source = tmp_path/"surface.nc"
    xr.Dataset(variables, coords={"valid_time": time+np.timedelta64(6, "h"), "latitude": lat, "longitude": lon}).to_netcdf(source, engine="h5netcdf")
    mask = tmp_path/"mask.nc"
    land = np.zeros((4, 3))
    land[1, 0] = 1
    land[2, 0] = .2
    xr.Dataset({"lsm": (("latitude", "longitude"), land)}, coords={"latitude": lat, "longitude": lon}).to_netcdf(mask, engine="h5netcdf")
    paths = []
    for h in ("north", "south"):
        p = tmp_path/f"{h}.nc"
        xr.Dataset({"standardized_pc": (("time", "mode"), x)}, coords={"time": time, "mode": np.arange(1, 13)}, attrs={"hemisphere": h, "detrended": 0, "native_resolution_primary": 1}).to_netcdf(p, engine="h5netcdf")
        paths.append(p)
    baseline = tmp_path/"baseline"
    args = cli.parse_args(["--input", str(source), "--mask", str(mask), "--nh-pcs", str(paths[0]), "--sh-pcs", str(paths[1]), "--output-dir", str(baseline), "--allow-other-resolution", "--turbulent", "--no-maps", "--tile-latitudes", "2", "--tile-longitudes", "2"])
    cli.run(args)
    args.sensitivity = True
    args.baseline_dir = baseline
    return args


def test_end_to_end_baseline_gate_linearity_chunking_and_overwrite(tmp_path):
    args = make_baseline(tmp_path)
    baseline_files = {p: p.read_bytes() for p in args.baseline_dir.rglob("*.json")}
    args.output_dir = tmp_path/"sensitivity_a"
    metadata = run_sensitivity(args, progress=lambda _: None)
    assert metadata["completed"]
    assert metadata["annual_samples"] == 12
    checks = json.loads((args.output_dir/"reports/acceptance.json").read_text())
    assert checks["regional_baseline_pass"] and checks["seasonal_turbulent_linearity_pass"]
    with pytest.raises(FileExistsError):
        run_sensitivity(args, progress=lambda _: None)
    args.output_dir = tmp_path/"sensitivity_b"
    args.tile_latitudes, args.tile_longitudes = 4, 3
    run_sensitivity(args, progress=lambda _: None)
    for label in ("annual_variability/north/latent_diagnostics.nc", "seasonal_regressions/south/turbulent_coefficients.nc"):
        with xr.open_dataset(tmp_path/"sensitivity_a"/label) as a, xr.open_dataset(tmp_path/"sensitivity_b"/label) as b:
            xr.testing.assert_allclose(a, b)
    for p, data in baseline_files.items():
        assert p.read_bytes() == data
    # Renderer is exercised on the same tiny grid, including stress vectors.
    from app.mslp_surface_sensitivity_diagnostics import render
    render(args.output_dir, args.baseline_dir, dpi=30, coastlines=False, figures=True, progress=lambda _: None)
    assert len(list(args.output_dir.rglob("*.png"))) > 40
