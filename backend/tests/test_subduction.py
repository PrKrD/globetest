from typing import Optional

import numpy as np

from app.dyntopo import EARTH_RADIUS_METERS
from app.subduction import geostrophic_subduction_rate


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(-180.0, 180.0, 90.0), np.array([-45.0, 0.0, 45.0])


def calculate(
    depth: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    valid: Optional[np.ndarray] = None,
    ocean: Optional[np.ndarray] = None,
):
    longitude, latitude = coordinates()
    shape = (latitude.size, longitude.size)
    return geostrophic_subduction_rate(
        depth,
        east,
        north,
        longitude,
        latitude,
        np.ones(shape, dtype=bool) if valid is None else valid,
        np.ones(shape, dtype=bool) if ocean is None else ocean,
    )


def test_zero_flux_has_zero_subduction_and_closes() -> None:
    shape = (3, 4)
    result = calculate(np.full(shape, 100.0), np.zeros(shape), np.zeros(shape))

    assert np.all(result.valid)
    assert np.allclose(result.rate_m_per_s, 0.0)
    assert result.conservation.total_outward_m3_per_s == 0.0
    assert result.conservation.closure_residual_m3_per_s == 0.0


def test_shared_face_uses_arithmetic_mean_transport_and_exact_area() -> None:
    shape = (3, 4)
    east_transport = np.zeros(shape)
    east_transport[:, 1] = 2.0
    result = calculate(np.ones(shape), east_transport, np.zeros(shape))

    face_length = EARTH_RADIUS_METERS * np.deg2rad(45.0)
    expected = -face_length / result.area_m2[:, 0]
    assert np.allclose(result.zonal_rate_m_per_s[:, 0], expected)
    assert np.allclose(result.zonal_rate_m_per_s[:, 1], 0.0)
    assert np.allclose(result.zonal_rate_m_per_s[:, 2], -expected)


def test_shared_dateline_face_is_periodic() -> None:
    shape = (3, 4)
    east_transport = np.zeros(shape)
    east_transport[:, 0] = 2.0
    result = calculate(np.ones(shape), east_transport, np.zeros(shape))

    face_length = EARTH_RADIUS_METERS * np.deg2rad(45.0)
    expected = face_length / result.area_m2[:, 0]
    assert np.allclose(result.zonal_rate_m_per_s[:, 0], 0.0)
    assert np.allclose(result.zonal_rate_m_per_s[:, 1], expected)
    assert np.allclose(result.zonal_rate_m_per_s[:, -1], -expected)


def test_meridional_flux_uses_spherical_face_length() -> None:
    shape = (3, 4)
    result = calculate(np.ones(shape), np.zeros(shape), np.ones(shape))
    latitude_edges = np.deg2rad(np.array([-67.5, -22.5, 22.5, 67.5]))
    longitude_width = np.deg2rad(90.0)
    face_lengths = EARTH_RADIUS_METERS * np.cos(latitude_edges) * longitude_width
    expected = -(face_lengths[1:] - face_lengths[:-1]) / result.area_m2[:, 0]
    assert np.allclose(result.meridional_rate_m_per_s[:, 0], expected)


def test_coast_and_missing_data_faces_are_distinct() -> None:
    shape = (3, 4)
    valid = np.zeros(shape, dtype=bool)
    valid[1, 1] = True
    ocean = np.zeros(shape, dtype=bool)
    ocean[1, 1] = True
    ocean[1, 2] = True
    ocean[2, 1] = True
    east = np.zeros(shape)
    north = np.zeros(shape)
    east[1, 1] = 3.0
    north[1, 1] = 4.0

    result = calculate(np.ones(shape), east, north, valid, ocean)
    budget = result.conservation
    east_length = EARTH_RADIUS_METERS * np.deg2rad(45.0)
    north_length = EARTH_RADIUS_METERS * np.cos(np.deg2rad(22.5)) * np.deg2rad(90.0)
    expected_outward = 3.0 * east_length + 4.0 * north_length

    assert budget.coastline_face_count == 2
    assert budget.missing_data_face_count == 2
    assert budget.coastline_outward_m3_per_s == 0.0
    assert np.isclose(budget.missing_data_outward_m3_per_s, expected_outward)
    assert np.isclose(
        result.rate_m_per_s[1, 1],
        -expected_outward / result.area_m2[1, 1],
    )
    assert abs(budget.closure_residual_m3_per_s) < 1e-6


def test_open_domain_flux_uses_wet_side_and_closes() -> None:
    shape = (3, 4)
    result = calculate(np.ones(shape), np.zeros(shape), np.ones(shape))
    budget = result.conservation
    longitude_width = np.deg2rad(90.0)
    expected = 4.0 * EARTH_RADIUS_METERS * longitude_width * (
        np.cos(np.deg2rad(67.5)) - np.cos(np.deg2rad(-67.5))
    )

    assert budget.open_domain_face_count == 8
    assert np.isclose(budget.open_domain_outward_m3_per_s, expected)
    assert budget.missing_data_face_count == 0
    assert budget.coastline_face_count == 0
    assert abs(budget.closure_residual_m3_per_s) < 1e-6


def test_global_budget_uses_the_same_faces_as_cell_budgets() -> None:
    rng = np.random.default_rng(42)
    shape = (3, 4)
    depth = rng.uniform(10.0, 200.0, shape)
    east = rng.normal(0.0, 0.2, shape)
    north = rng.normal(0.0, 0.2, shape)
    valid = np.ones(shape, dtype=bool)
    valid[1, 1] = False
    ocean = np.ones(shape, dtype=bool)

    result = calculate(depth, east, north, valid, ocean)
    scale = max(
        abs(result.conservation.total_outward_m3_per_s),
        abs(result.conservation.area_integrated_subduction_m3_per_s),
        1.0,
    )
    assert abs(result.conservation.closure_residual_m3_per_s) / scale < 1e-14
    assert abs(result.conservation.relative_closure_residual) < 1e-14


def test_missing_input_at_center_is_unavailable() -> None:
    shape = (3, 4)
    depth = np.ones(shape)
    depth[1, 1] = np.nan
    result = calculate(depth, np.ones(shape), np.ones(shape))

    assert not result.valid[1, 1]
    assert np.isnan(result.rate_m_per_s[1, 1])


def test_usable_cell_must_be_physical_ocean() -> None:
    shape = (3, 4)
    ocean = np.ones(shape, dtype=bool)
    ocean[1, 1] = False
    with np.testing.assert_raises_regex(ValueError, "physical ocean"):
        calculate(np.ones(shape), np.ones(shape), np.ones(shape), ocean=ocean)
