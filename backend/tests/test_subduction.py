import numpy as np

from app.dyntopo import EARTH_RADIUS_METERS
from app.subduction import geostrophic_subduction_rate


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(-180.0, 180.0, 90.0), np.array([-45.0, 0.0, 45.0])


def test_zero_flux_has_zero_subduction() -> None:
    longitude, latitude = coordinates()
    shape = (latitude.size, longitude.size)
    result = geostrophic_subduction_rate(
        np.full(shape, 100.0),
        np.zeros(shape),
        np.zeros(shape),
        longitude,
        latitude,
        np.ones(shape, dtype=bool),
    )

    assert np.all(result.valid)
    assert np.allclose(result.rate_m_per_s, 0.0)


def test_spherical_divergence_wraps_at_dateline_and_uses_latitude_metric() -> None:
    longitude, latitude = coordinates()
    longitude_rad = np.deg2rad(longitude)
    shape = (latitude.size, longitude.size)
    depth = np.ones(shape)
    east_velocity = np.tile(np.sin(longitude_rad), (latitude.size, 1))
    result = geostrophic_subduction_rate(
        depth,
        east_velocity,
        np.zeros(shape),
        longitude,
        latitude,
        np.ones(shape, dtype=bool),
    )

    delta = np.deg2rad(90.0)
    discrete_derivative_at_seam = 1.0 / delta
    expected = discrete_derivative_at_seam / (
        EARTH_RADIUS_METERS * np.cos(np.deg2rad(latitude))
    )
    assert np.allclose(result.rate_m_per_s[:, 0], expected)


def test_meridional_flux_includes_cosine_metric_term() -> None:
    longitude = np.arange(-180.0, 180.0, 90.0)
    latitude = np.arange(-60.0, 61.0, 30.0)
    shape = (latitude.size, longitude.size)
    result = geostrophic_subduction_rate(
        np.ones(shape),
        np.zeros(shape),
        np.ones(shape),
        longitude,
        latitude,
        np.ones(shape, dtype=bool),
    )

    latitude_rad = np.deg2rad(latitude)
    row = 3  # 30 degrees north
    discrete_derivative = (
        np.cos(latitude_rad[row + 1]) - np.cos(latitude_rad[row - 1])
    ) / (2.0 * np.deg2rad(30.0))
    expected = -discrete_derivative / (
        EARTH_RADIUS_METERS * np.cos(latitude_rad[row])
    )
    assert np.allclose(result.rate_m_per_s[row], expected)


def test_one_sided_difference_is_used_when_one_neighbour_is_missing() -> None:
    longitude, latitude = coordinates()
    shape = (latitude.size, longitude.size)
    east_velocity = np.tile(np.arange(longitude.size, dtype=float), (latitude.size, 1))
    valid = np.ones(shape, dtype=bool)
    valid[1, 0] = False

    result = geostrophic_subduction_rate(
        np.ones(shape),
        east_velocity,
        np.zeros(shape),
        longitude,
        latitude,
        valid,
    )

    expected = -1.0 / (EARTH_RADIUS_METERS * np.deg2rad(90.0))
    assert result.valid[1, 1]
    assert np.isclose(result.rate_m_per_s[1, 1], expected)


def test_total_is_unavailable_when_either_direction_cannot_be_formed() -> None:
    longitude, latitude = coordinates()
    shape = (latitude.size, longitude.size)
    valid = np.ones(shape, dtype=bool)
    valid[1, 0] = False
    valid[1, 2] = False

    result = geostrophic_subduction_rate(
        np.ones(shape),
        np.ones(shape),
        np.ones(shape),
        longitude,
        latitude,
        valid,
    )

    assert not result.valid[1, 1]
    assert np.isnan(result.rate_m_per_s[1, 1])

    valid = np.ones(shape, dtype=bool)
    valid[0, 1] = False
    valid[2, 1] = False
    result = geostrophic_subduction_rate(
        np.ones(shape),
        np.ones(shape),
        np.ones(shape),
        longitude,
        latitude,
        valid,
    )
    assert not result.valid[1, 1]
    assert np.isnan(result.rate_m_per_s[1, 1])


def test_missing_input_at_center_is_unavailable() -> None:
    longitude, latitude = coordinates()
    shape = (latitude.size, longitude.size)
    depth = np.ones(shape)
    depth[1, 1] = np.nan

    result = geostrophic_subduction_rate(
        depth,
        np.ones(shape),
        np.ones(shape),
        longitude,
        latitude,
        np.ones(shape, dtype=bool),
    )

    assert not result.valid[1, 1]
