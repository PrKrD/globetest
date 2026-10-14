import numpy as np

from app.velocity import reconstruct_geostrophic_velocity


def synthetic_height() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.arange(-180.0, 180.0, 30.0)
    latitude = np.arange(-4.0, 4.5, 0.5)
    longitude_rad = np.deg2rad(longitude)[None, :]
    latitude_rad = np.deg2rad(latitude)[:, None]
    height = 0.2 * latitude_rad**2 + 0.01 * np.sin(longitude_rad) * latitude_rad
    return height, longitude, latitude


def test_reconstruction_uses_beta_core_blend_and_ordinary_solution() -> None:
    height, longitude, latitude = synthetic_height()
    result = reconstruct_geostrophic_velocity(height, longitude, latitude)

    equator = int(np.flatnonzero(latitude == 0.0)[0])
    transition = int(np.flatnonzero(latitude == 1.5)[0])
    ordinary = int(np.flatnonzero(latitude == 4.0)[0])

    assert np.all(result.valid[equator])
    assert np.allclose(result.east[equator], result.beta_east[equator])
    assert np.allclose(result.north[equator], result.beta_north[equator])
    assert np.all((result.blend_weight[transition] > 0.0))
    assert np.all((result.blend_weight[transition] < 1.0))
    assert np.allclose(
        result.east[transition],
        (1.0 - result.blend_weight[transition]) * result.beta_east[transition]
        + result.blend_weight[transition] * result.ordinary_east[transition],
    )
    assert np.allclose(result.east[ordinary], result.ordinary_east[ordinary])
    assert np.allclose(result.north[ordinary], result.ordinary_north[ordinary])


def test_reconstruction_preserves_invalid_input_cells() -> None:
    height, longitude, latitude = synthetic_height()
    height[8, 3] = np.nan

    result = reconstruct_geostrophic_velocity(height, longitude, latitude)

    assert not result.valid[8, 3]
    assert np.isnan(result.east[8, 3])
    assert np.isnan(result.north[8, 3])


def test_reconstruction_rejects_an_empty_blend_interval() -> None:
    height, longitude, latitude = synthetic_height()

    with np.testing.assert_raises_regex(ValueError, "Blend half-width"):
        reconstruct_geostrophic_velocity(
            height,
            longitude,
            latitude,
            core_latitude_deg=3.0,
            blend_half_width_deg=3.0,
        )
