"""Maximenko dynamic-topography loading and spherical-gradient export helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import UnivariateSpline


EARTH_RADIUS_METERS = 6_371_008.8
EARTH_ROTATION_RAD_PER_S = 7.2921159e-5
GRAVITY_M_PER_S2 = 9.80665


@dataclass(frozen=True)
class DynamicTopography:
    """Mean dynamic topography sampled on its source latitude/longitude grid."""

    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    height_m: np.ndarray
    valid: np.ndarray


def load_maximenko(path: str) -> DynamicTopography:
    """Load the classic-NetCDF Maximenko product and remove its duplicated seam."""
    with netcdf_file(path, "r", mmap=False) as dataset:
        longitude = np.asarray(dataset.variables["LON0_720"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["LAT"][:], dtype=np.float64)
        variable = dataset.variables["MDOT"]
        raw_cm = np.squeeze(np.asarray(variable[:], dtype=np.float64))
        fill_value = float(getattr(variable, "_FillValue", getattr(variable, "missing_value")))

    if raw_cm.shape != (latitude.size, longitude.size):
        raise ValueError("MDOT dimensions do not match the LAT/LON0_720 coordinate arrays")
    if longitude.size < 2 or not np.isclose(longitude[-1] - longitude[0], 360.0):
        raise ValueError("Expected a duplicated 360° longitude endpoint in LON0_720")

    # The endpoints -180° and 180° describe the same meridian. Keep one value
    # so centered differences can wrap across the dateline without a duplicate cell.
    longitude = longitude[:-1]
    raw_cm = raw_cm[:, :-1]
    valid = np.isfinite(raw_cm) & ~np.isclose(raw_cm, fill_value)
    height_m = np.where(valid, raw_cm / 100.0, np.nan)
    return DynamicTopography(longitude, latitude, height_m, valid)


def spherical_gradient(
    height_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eastward, northward, and magnitude gradients in m/m.

    Longitude is periodic. Distances are calculated on a sphere, including the
    cos(latitude) factor, and never divide by the polar singularity. One-sided
    differences retain valid coastal samples when the opposite neighbour is land.
    """
    values = np.asarray(height_m, dtype=np.float64)
    if values.shape != (latitude_deg.size, longitude_deg.size):
        raise ValueError("Height field dimensions do not match its coordinates")

    longitude_rad = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    latitude_rad = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    if not (np.all(np.diff(longitude_rad) > 0) and np.all(np.diff(latitude_rad) > 0)):
        raise ValueError("Coordinates must be strictly increasing")

    valid = np.isfinite(values)
    west = np.roll(values, 1, axis=1)
    east = np.roll(values, -1, axis=1)
    west_valid = np.roll(valid, 1, axis=1)
    east_valid = np.roll(valid, -1, axis=1)
    dlon = float(np.median(np.diff(longitude_rad)))
    dx = EARTH_RADIUS_METERS * np.maximum(np.cos(latitude_rad), 1e-6)[:, None] * dlon

    grad_east = np.full_like(values, np.nan)
    both_x = valid & west_valid & east_valid
    forward_x = valid & east_valid & ~west_valid
    backward_x = valid & west_valid & ~east_valid
    grad_east[both_x] = ((east - west) / (2.0 * dx))[both_x]
    grad_east[forward_x] = ((east - values) / dx)[forward_x]
    grad_east[backward_x] = ((values - west) / dx)[backward_x]

    south = np.empty_like(values)
    north = np.empty_like(values)
    south[0] = np.nan
    south[1:] = values[:-1]
    north[-1] = np.nan
    north[:-1] = values[1:]
    south_valid = np.isfinite(south)
    north_valid = np.isfinite(north)
    dy_forward = EARTH_RADIUS_METERS * np.diff(latitude_rad)
    dy = float(np.median(dy_forward))
    grad_north = np.full_like(values, np.nan)
    both_y = valid & south_valid & north_valid
    forward_y = valid & north_valid & ~south_valid
    backward_y = valid & south_valid & ~north_valid
    grad_north[both_y] = ((north - south) / (2.0 * dy))[both_y]
    grad_north[forward_y] = ((north - values) / dy)[forward_y]
    grad_north[backward_y] = ((values - south) / dy)[backward_y]

    magnitude = np.hypot(grad_east, grad_north)
    return grad_east, grad_north, magnitude


def coordinate_edges(coordinates: np.ndarray) -> np.ndarray:
    """Infer cell edges from a strictly increasing array of cell centres."""
    spacing = np.diff(coordinates)
    if coordinates.size < 2 or not np.allclose(spacing, spacing[0]):
        raise ValueError("Coordinates must be evenly spaced")
    half = spacing[0] / 2.0
    return np.concatenate(([coordinates[0] - half], coordinates[:-1] + half, [coordinates[-1] + half]))


def equatorial_beta_plane_velocity(
    height_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    fit_half_width_deg: float = 3.5,
    smoothing_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate equatorial velocity from cubic-spline height curvature.

    The ordinary f-plane expression is singular at the equator.  On an
    equatorial beta plane, differentiating the zonal momentum balance gives
    u = -g eta_yy / beta at y=0.  We extend that smooth estimate through a
    narrow tropical band, then callers blend it with the normal f-plane
    result away from the equator.  The meridional component uses
    v = g eta_xy / beta.

    Cubic smoothing splines are fitted independently to contiguous valid
    latitude runs at each longitude.  This avoids taking second differences
    of the raw, noisy grid and avoids bridging land/no-data gaps.
    """
    values = np.asarray(height_m, dtype=np.float64)
    if values.shape != (latitude_deg.size, longitude_deg.size):
        raise ValueError("Height field dimensions do not match its coordinates")

    latitude = np.asarray(latitude_deg, dtype=np.float64)
    y_m = EARTH_RADIUS_METERS * np.deg2rad(latitude)
    slope_y = np.full_like(values, np.nan)
    curvature_yy = np.full_like(values, np.nan)
    in_band = np.abs(latitude) <= fit_half_width_deg

    for j in range(values.shape[1]):
        valid_indices = np.flatnonzero(in_band & np.isfinite(values[:, j]))
        if valid_indices.size < 4:
            continue
        split_points = np.where(np.diff(valid_indices) != 1)[0] + 1
        for indices in np.split(valid_indices, split_points):
            if indices.size < 4:
                continue
            spline = UnivariateSpline(
                y_m[indices],
                values[indices, j],
                k=3,
                s=(smoothing_m**2) * indices.size,
            )
            slope_y[indices, j] = spline(y_m[indices], nu=1)
            curvature_yy[indices, j] = spline(y_m[indices], nu=2)

    eta_xy, _, _ = spherical_gradient(slope_y, longitude_deg, latitude_deg)
    beta = 2.0 * EARTH_ROTATION_RAD_PER_S / EARTH_RADIUS_METERS
    velocity_east = -GRAVITY_M_PER_S2 * curvature_yy / beta
    velocity_north = GRAVITY_M_PER_S2 * eta_xy / beta
    valid = np.isfinite(velocity_east) & np.isfinite(velocity_north)
    velocity_east[~valid] = np.nan
    velocity_north[~valid] = np.nan
    return velocity_east, velocity_north, valid
