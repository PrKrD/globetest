from dataclasses import FrozenInstanceError, fields as dataclass_fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

from app.dyntopo import load_maximenko
from app.mld import load_mld, maximum_mld, regrid_maximum_mld, regrid_ocean_mask_by_overlap
from app.ocean_fields import OceanFields, compute_ocean_fields
from app.subduction import geostrophic_subduction_rate
from app.velocity import reconstruct_geostrophic_velocity


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def assert_fields_equal(actual, expected) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in dataclass_fields(expected):
            assert_fields_equal(getattr(actual, field.name), getattr(expected, field.name))
    else:
        assert actual == expected


def test_compute_preserves_original_scientific_sequence_without_file_output(monkeypatch) -> None:
    topography = load_maximenko(str(DATA_DIR / "Maximenko_dyntopo.nc"))
    source = load_mld(str(DATA_DIR / "mld_dr003_ref10m_v2023.nc"))

    # The original builder's scientific call sequence is the regression oracle.
    maximum = maximum_mld(source.monthly_m)
    ocean = regrid_ocean_mask_by_overlap(
        source.source_ocean, source.longitude_deg, source.latitude_deg,
        topography.longitude_deg, topography.latitude_deg,
    )
    mld = regrid_maximum_mld(
        maximum.maximum_m, maximum.month_count,
        source.longitude_deg, source.latitude_deg,
        topography.longitude_deg, topography.latitude_deg, ocean,
    )
    velocity = reconstruct_geostrophic_velocity(
        topography.height_m, topography.longitude_deg, topography.latitude_deg,
    )
    subduction = geostrophic_subduction_rate(
        mld.maximum_m, velocity.east, velocity.north,
        topography.longitude_deg, topography.latitude_deg,
        mld.valid & velocity.valid, ocean,
    )
    expected = OceanFields(
        topography.longitude_deg, topography.latitude_deg, ocean, mld, velocity, subduction,
    )

    def forbid_output(*args, **kwargs):
        pytest.fail("Scientific computation attempted file output")

    monkeypatch.setattr(Path, "write_text", forbid_output)
    monkeypatch.setattr(Path, "open", forbid_output)
    monkeypatch.setattr(Path, "mkdir", forbid_output)
    actual = compute_ocean_fields(topography, source)

    assert_fields_equal(actual, expected)
    assert actual.longitude_deg is topography.longitude_deg
    assert actual.latitude_deg is topography.latitude_deg
    assert np.isnan(actual.mld.maximum_m).any()
    assert np.isnan(actual.velocity.east).any()
    assert np.isnan(actual.subduction_geo.rate_m_per_s).any()
    with pytest.raises(FrozenInstanceError):
        actual.mld = mld
