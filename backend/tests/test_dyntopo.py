import numpy as np

from app.dyntopo import (
    EARTH_RADIUS_METERS,
    EARTH_ROTATION_RAD_PER_S,
    GRAVITY_M_PER_S2,
    coordinate_edges,
    equatorial_beta_plane_velocity,
    spherical_gradient,
)


def test_spherical_gradient_handles_dateline_periodically() -> None:
    longitude = np.arange(-180.0, 180.0, 90.0)
    latitude = np.array([-45.0, 0.0, 45.0])
    # A longitude-independent field must have no eastward gradient at the seam.
    field = np.tile(latitude[:, None] * 0.01, (1, longitude.size))
    east, north, magnitude = spherical_gradient(field, longitude, latitude)
    assert np.allclose(east, 0.0)
    assert np.all(np.isfinite(north))
    assert np.all(np.isfinite(magnitude))


def test_spherical_gradient_uses_smaller_east_west_distance_at_high_latitude() -> None:
    longitude = np.arange(-180.0, 180.0, 90.0)
    latitude = np.array([-60.0, 0.0, 60.0])
    field = np.tile(np.sin(np.deg2rad(longitude)), (latitude.size, 1))
    east, _, _ = spherical_gradient(field, longitude, latitude)
    assert abs(east[0, 0]) > abs(east[1, 0])


def test_coordinate_edges_are_centered() -> None:
    assert np.allclose(coordinate_edges(np.array([-1.0, 0.0, 1.0])), [-1.5, -0.5, 0.5, 1.5])


def test_beta_plane_spline_recovers_zonal_velocity_at_equator() -> None:
    longitude = np.arange(-180.0, 180.0, 30.0)
    latitude = np.arange(-4.0, 4.5, 0.5)
    y_m = EARTH_RADIUS_METERS * np.deg2rad(latitude)
    beta = 2.0 * EARTH_ROTATION_RAD_PER_S / EARTH_RADIUS_METERS
    expected_u = 0.35
    # eta_yy = -beta * u / g, so the beta-plane equation has an exact answer.
    height = np.tile(
        -beta * expected_u * y_m[:, None] ** 2 / (2.0 * GRAVITY_M_PER_S2),
        (1, longitude.size),
    )

    east, north, valid = equatorial_beta_plane_velocity(height, longitude, latitude)
    equator = np.where(latitude == 0.0)[0][0]
    assert np.all(valid[equator])
    assert np.allclose(east[equator], expected_u, atol=1e-6)
    assert np.allclose(north[equator], 0.0, atol=1e-10)
