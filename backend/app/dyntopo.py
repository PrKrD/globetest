"""Maximenko dynamic-topography loading and spherical-gradient export helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import netcdf_file


EARTH_RADIUS_METERS = 6_371_008.8


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
