"""Geostrophic contribution to mixed-layer subduction on a spherical grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dyntopo import EARTH_RADIUS_METERS


@dataclass(frozen=True)
class GeostrophicSubduction:
    """Signed subduction rate and its availability mask."""

    rate_m_per_s: np.ndarray
    valid: np.ndarray


def geostrophic_subduction_rate(
    mixed_layer_depth_m: np.ndarray,
    velocity_east_m_per_s: np.ndarray,
    velocity_north_m_per_s: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    valid: np.ndarray,
) -> GeostrophicSubduction:
    """Return ``-div(H * velocity)`` using spherical vector divergence.

    Longitude is periodic. Each directional derivative is centered when both
    neighbours are usable and one-sided when only one is usable. The total is
    unavailable unless both directional derivatives can be formed.
    """
    depth = np.asarray(mixed_layer_depth_m, dtype=np.float64)
    east_velocity = np.asarray(velocity_east_m_per_s, dtype=np.float64)
    north_velocity = np.asarray(velocity_north_m_per_s, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool).copy()
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    shape = (latitude.size, longitude.size)

    if depth.shape != shape or east_velocity.shape != shape or north_velocity.shape != shape:
        raise ValueError("Subduction fields do not match their coordinates")
    if usable.shape != shape:
        raise ValueError("Subduction validity mask does not match its coordinates")
    if longitude.size < 3 or latitude.size < 3:
        raise ValueError("Subduction requires at least three coordinates in each direction")
    if not (np.all(np.diff(longitude) > 0.0) and np.all(np.diff(latitude) > 0.0)):
        raise ValueError("Coordinates must be strictly increasing")
    longitude_spacing = np.diff(longitude)
    latitude_spacing = np.diff(latitude)
    if not np.allclose(longitude_spacing, longitude_spacing[0]):
        raise ValueError("Longitude coordinates must be evenly spaced")
    if not np.allclose(latitude_spacing, latitude_spacing[0]):
        raise ValueError("Latitude coordinates must be evenly spaced")
    if not np.isclose(longitude[-1] - longitude[0] + longitude_spacing[0], 2.0 * np.pi):
        raise ValueError("Longitude coordinates must cover one periodic revolution")

    usable &= (
        np.isfinite(depth)
        & np.isfinite(east_velocity)
        & np.isfinite(north_velocity)
    )
    cosine_latitude = np.cos(latitude)[:, None]
    if np.any(cosine_latitude <= 0.0):
        raise ValueError("Latitude coordinates must lie strictly between the poles")

    east_flux = depth * east_velocity
    north_flux_cosine = depth * north_velocity * cosine_latitude

    west_flux = np.roll(east_flux, 1, axis=1)
    east_flux_neighbour = np.roll(east_flux, -1, axis=1)
    west_valid = np.roll(usable, 1, axis=1)
    east_valid = np.roll(usable, -1, axis=1)
    derivative_longitude = np.full(shape, np.nan)
    centered_x = usable & west_valid & east_valid
    forward_x = usable & ~west_valid & east_valid
    backward_x = usable & west_valid & ~east_valid
    delta_longitude = float(longitude_spacing[0])
    derivative_longitude[centered_x] = (
        (east_flux_neighbour - west_flux) / (2.0 * delta_longitude)
    )[centered_x]
    derivative_longitude[forward_x] = (
        (east_flux_neighbour - east_flux) / delta_longitude
    )[forward_x]
    derivative_longitude[backward_x] = (
        (east_flux - west_flux) / delta_longitude
    )[backward_x]

    south_flux = np.full(shape, np.nan)
    north_flux = np.full(shape, np.nan)
    south_flux[1:] = north_flux_cosine[:-1]
    north_flux[:-1] = north_flux_cosine[1:]
    south_valid = np.zeros(shape, dtype=bool)
    north_valid = np.zeros(shape, dtype=bool)
    south_valid[1:] = usable[:-1]
    north_valid[:-1] = usable[1:]
    derivative_latitude = np.full(shape, np.nan)
    centered_y = usable & south_valid & north_valid
    forward_y = usable & ~south_valid & north_valid
    backward_y = usable & south_valid & ~north_valid
    delta_latitude = float(latitude_spacing[0])
    derivative_latitude[centered_y] = (
        (north_flux - south_flux) / (2.0 * delta_latitude)
    )[centered_y]
    derivative_latitude[forward_y] = (
        (north_flux - north_flux_cosine) / delta_latitude
    )[forward_y]
    derivative_latitude[backward_y] = (
        (north_flux_cosine - south_flux) / delta_latitude
    )[backward_y]

    derivative_x_valid = centered_x | forward_x | backward_x
    derivative_y_valid = centered_y | forward_y | backward_y
    result_valid = derivative_x_valid & derivative_y_valid
    rate = np.full(shape, np.nan)
    divergence = (derivative_longitude + derivative_latitude) / (
        EARTH_RADIUS_METERS * cosine_latitude
    )
    rate[result_valid] = -divergence[result_valid]
    return GeostrophicSubduction(rate, result_valid)
