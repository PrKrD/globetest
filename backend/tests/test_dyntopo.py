import numpy as np

from app.dyntopo import coordinate_edges, spherical_gradient


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
