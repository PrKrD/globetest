from copy import deepcopy
from dataclasses import fields as dataclass_fields, is_dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from app.globe_export import export_globe_json
from app.mld import RegriddedMixedLayerDepth
from app.ocean_fields import OceanFields
from app.subduction import GeostrophicSubduction, SubductionConservation
from app.velocity import GeostrophicVelocity


@pytest.fixture
def ocean_fields() -> OceanFields:
    shape = (3, 4)
    ocean = np.ones(shape, dtype=bool)
    ocean[0, 0] = False
    mld_valid = ocean.copy()
    velocity_valid = ocean.copy()
    velocity_valid[1, 1] = False  # Ocean without topography coverage.
    east = np.full(shape, 0.123456789)
    east[1, 0] = 100.0  # Experimental estimate must not affect the colour scale.
    east[~velocity_valid] = np.nan
    north = np.full(shape, -0.234567891)
    north[~velocity_valid] = np.nan
    depth = np.where(mld_valid, 123.456789, np.nan)
    counts = np.where(mld_valid, 11, 0)
    rate = np.where(velocity_valid, -1.234567891234e-6, np.nan)
    velocity = GeostrophicVelocity(
        east, north, velocity_valid,
        east.copy(), north.copy(), velocity_valid.copy(),
        east.copy(), north.copy(), velocity_valid.copy(), np.zeros(shape),
    )
    conservation = SubductionConservation(
        coastline_face_count=2,
        missing_data_face_count=4,
        open_domain_face_count=8,
        coastline_outward_m3_per_s=0.0,
        missing_data_outward_m3_per_s=-1234.56789,
        open_domain_outward_m3_per_s=0.0,
        total_outward_m3_per_s=-1234.56789,
        area_integrated_subduction_m3_per_s=1234.56789,
        closure_residual_m3_per_s=1e-8,
    )
    subduction = GeostrophicSubduction(
        rate, velocity_valid.copy(), rate.copy(), np.zeros(shape),
        np.ones(shape), conservation,
    )
    return OceanFields(
        np.array([-135.0, -45.0, 45.0, 135.0]),
        np.array([-4.0, 0.0, 4.0]), ocean,
        RegriddedMixedLayerDepth(depth, counts, mld_valid), velocity, subduction,
    )


def assert_unchanged(actual, before) -> None:
    if isinstance(before, np.ndarray):
        np.testing.assert_array_equal(actual, before)
    elif is_dataclass(before):
        for field in dataclass_fields(before):
            assert_unchanged(getattr(actual, field.name), getattr(before, field.name))
    else:
        assert actual == before


def test_export_preserves_schema_flags_precision_and_scientific_fields(
    ocean_fields, tmp_path, capsys, monkeypatch,
) -> None:
    before = deepcopy(ocean_fields)
    output = tmp_path / "nested" / "globe.json"
    output.parent.mkdir()
    output.write_text("previous product", encoding="utf-8")
    original_replace = Path.replace
    replacements = []

    def check_replace(temporary, target):
        assert temporary == output.with_suffix(".json.tmp")
        assert target == output
        assert output.read_text(encoding="utf-8") == "previous product"
        assert len(json.loads(temporary.read_text(encoding="utf-8"))["cells"]) == 12
        replacements.append(target)
        return original_replace(temporary, target)

    monkeypatch.setattr(Path, "replace", check_replace)
    export_globe_json(ocean_fields, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    cells = payload.pop("cells")
    assert payload == {
        "dataset_type": "maximenko_ocean_diagnostics",
        "units": {"velocity": "m/s", "mld_max": "m", "subduction": "m/s"},
        "resolution_deg": 90.0,
        "n_lat": 3,
        "n_lon": 4,
        "velocity_p98_m_per_s": round(float(np.hypot(0.123456789, -0.234567891)), 6),
        "mld_max_p98_m": 123.457,
        "subduction_abs_p98_m_per_s": 0.000001234568,
        "mld_source_resolution_deg": 1.0,
        "mld_temporal_statistic": "maximum_of_available_monthly_climatologies",
        "mld_regridding_method": "periodic_mask_aware_bilinear",
        "equatorial_method": "cubic_spline_beta_plane_blended_to_f_plane",
        "subduction_method": "conservative_spherical_finite_volume_of_mld_times_geostrophic_velocity",
        "subduction_face_transport": "arithmetic_mean_Hu_Hv_on_shared_wet_faces",
        "subduction_boundary_policy": {
            "physical_coastline": "zero_normal_flux",
            "ocean_missing_data": "wet_side_transport",
            "north_south_domain_edge": "wet_side_transport",
        },
        "physical_ocean_mask_method": "mld_source_mask_spherical_overlap_majority_ties_ocean",
        "subduction_conservation": {
            "area_integrated_m3_per_s": 1234.568,
            "outward_boundary_m3_per_s": -1234.568,
            "closure_residual_m3_per_s": 0.0,
        },
        "beta_plane_core_latitude_deg": 0.25,
        "beta_plane_blend_latitude_deg": 3.0,
    }
    assert [cell["id"] for cell in cells] == list(range(12))
    assert [(cell["clat"], cell["clon"]) for cell in cells] == [
        (lat, lon) for lat in [-4.0, 0.0, 4.0] for lon in [-135.0, -45.0, 45.0, 135.0]
    ]
    assert cells[0] == {
        "id": 0, "lat0": -6.0, "lat1": -2.0, "lon0": -135.0, "lon1": -45.0,
        "clat": -4.0, "clon": -135.0, "is_land": 1, "has_velocity": 0,
        "velocity_speed_m_per_s": 0.0, "equatorial_uncertainty": 0,
        "equatorial_estimate": 0, "has_mld": 0, "mld_max_m": 0.0,
        "mld_month_count_min": 0, "has_subduction": 0, "subduction_m_per_s": 0.0,
    }
    assert cells[4]["equatorial_uncertainty"] == cells[4]["equatorial_estimate"] == 1
    assert cells[4]["velocity_speed_m_per_s"] == round(float(np.hypot(100.0, -0.234567891)), 6)
    assert cells[4]["mld_max_m"] == 123.457
    assert cells[4]["subduction_m_per_s"] == -0.000001234568
    assert cells[5]["is_land"] == 0
    assert cells[5]["has_mld"] == 1
    assert cells[5]["mld_month_count_min"] == 11
    assert cells[5]["equatorial_uncertainty"] == 1
    assert cells[5]["equatorial_estimate"] == cells[5]["has_velocity"] == 0
    assert cells[5]["has_subduction"] == 0
    assert cells[-1]["lon1"] == 225.0
    assert replacements == [output]
    assert not output.with_suffix(".json.tmp").exists()
    assert_unchanged(ocean_fields, before)
    summary = capsys.readouterr().out
    assert f"Wrote 12 cells to {output}" in summary
    assert "Native grid: 3 x 4 at 90°" in summary
    assert "valid velocity cells: 10" in summary
    assert "valid MLD cells: 11" in summary
    assert "valid subduction cells: 10" in summary
    assert "Subduction conservation residual: 1.000e-08 m^3/s" in summary


def test_failed_serialization_keeps_previous_product(ocean_fields, tmp_path, monkeypatch) -> None:
    output = tmp_path / "globe.json"
    output.write_text("previous product", encoding="utf-8")

    def fail_serialization(*args, **kwargs):
        raise ValueError("serialization failed")

    monkeypatch.setattr("app.globe_export.json.dumps", fail_serialization)
    with pytest.raises(ValueError, match="serialization failed"):
        export_globe_json(ocean_fields, output)
    assert output.read_text(encoding="utf-8") == "previous product"
