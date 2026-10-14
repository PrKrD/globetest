from pathlib import Path

import numpy as np

from app.dyntopo import load_maximenko
from app.mld import (
    load_mld,
    maximum_mld,
    regrid_maximum_mld,
    regrid_ocean_mask_by_overlap,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mld_dr003_ref10m_v2023.nc"
TOPOGRAPHY_PATH = Path(__file__).resolve().parents[1] / "data" / "Maximenko_dyntopo.nc"


def test_load_mld_filters_fill_values_and_source_land() -> None:
    source = load_mld(str(DATA_PATH))

    assert source.monthly_m.shape == (12, 180, 360)
    assert np.isnan(source.monthly_m).any()
    assert not np.any(source.monthly_m == -99999.0)
    assert np.all(np.isnan(source.monthly_m[:, ~source.source_ocean]))


def test_maximum_mld_uses_available_months_and_counts_them() -> None:
    monthly = np.full((12, 1, 3), np.nan)
    monthly[:, 0, 0] = np.arange(12.0)
    monthly[:4, 0, 1] = [5.0, 12.0, 7.0, 3.0]

    result = maximum_mld(monthly)

    assert np.allclose(result.maximum_m[0, :2], [11.0, 12.0])
    assert np.isnan(result.maximum_m[0, 2])
    assert result.month_count.tolist() == [[12, 4, 0]]


def test_regrid_is_bilinear_exact_at_centres_and_periodic_at_dateline() -> None:
    longitude = np.array([-179.5, -178.5, -177.5, -176.5])
    latitude = np.array([0.5, 1.5])
    maximum = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    counts = np.full_like(maximum, 12, dtype=np.int16)
    target_lon = np.array([-180.0, -179.5, -179.0])
    target_lat = np.array([0.5, 1.0])
    target_mask = np.ones((2, 3), dtype=bool)

    result = regrid_maximum_mld(
        maximum, counts, longitude, latitude, target_lon, target_lat, target_mask
    )

    assert np.allclose(result.maximum_m[0], [25.0, 10.0, 15.0])
    assert np.allclose(result.maximum_m[1], [45.0, 30.0, 35.0])
    assert np.all(result.valid)


def test_regrid_renormalizes_missing_corners_and_uses_minimum_coverage() -> None:
    longitude = np.array([-179.5, -178.5, -177.5, -176.5])
    latitude = np.array([0.5, 1.5])
    maximum = np.array([[10.0, np.nan, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    counts = np.array([[12, 0, 12, 12], [9, 11, 12, 12]], dtype=np.int16)
    target_mask = np.array([[True, False]])

    result = regrid_maximum_mld(
        maximum,
        counts,
        longitude,
        latitude,
        np.array([-179.0, -179.0]),
        np.array([1.0]),
        target_mask,
    )

    # Three valid, equally weighted corners remain after the missing corner is removed.
    assert np.isclose(result.maximum_m[0, 0], 40.0)
    assert result.month_count_min[0, 0] == 9
    assert result.valid.tolist() == [[True, False]]
    assert np.isnan(result.maximum_m[0, 1])
    assert result.month_count_min[0, 1] == 0


def test_ocean_mask_overlap_majority_keeps_ties_as_ocean() -> None:
    source_longitude = np.array([-135.0, -45.0, 45.0, 135.0])
    source_latitude = np.array([-45.0, 45.0])
    source_ocean = np.array(
        [
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    target_longitude = np.arange(-180.0, 180.0, 45.0)
    target_latitude = np.array([-67.5, -22.5, 22.5, 67.5])

    result = regrid_ocean_mask_by_overlap(
        source_ocean,
        source_longitude,
        source_latitude,
        target_longitude,
        target_latitude,
    )

    # -90 degrees straddles the ocean and land source cells equally.
    tie_column = int(np.flatnonzero(target_longitude == -90.0)[0])
    assert np.all(result[:, tie_column])
    assert np.all(result[:, target_longitude == -135.0])
    assert not np.any(result[:, target_longitude == -45.0])


def test_remapped_source_mask_preserves_all_valid_transport_cells() -> None:
    topography = load_maximenko(str(TOPOGRAPHY_PATH))
    mld = load_mld(str(DATA_PATH))
    physical_ocean = regrid_ocean_mask_by_overlap(
        mld.source_ocean,
        mld.longitude_deg,
        mld.latitude_deg,
        topography.longitude_deg,
        topography.latitude_deg,
    )

    assert np.all(physical_ocean[topography.valid])
