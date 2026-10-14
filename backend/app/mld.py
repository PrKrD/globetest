"""Mixed-layer-depth loading, temporal reduction, and grid interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import netcdf_file


@dataclass(frozen=True)
class MixedLayerDepth:
    """Monthly mixed-layer depth on the source one-degree grid."""

    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    monthly_m: np.ndarray
    source_ocean: np.ndarray


@dataclass(frozen=True)
class MaximumMixedLayerDepth:
    """Maximum MLD and temporal coverage on the source grid."""

    maximum_m: np.ndarray
    month_count: np.ndarray


@dataclass(frozen=True)
class RegriddedMixedLayerDepth:
    """Maximum MLD and conservative coverage on a target grid."""

    maximum_m: np.ndarray
    month_count_min: np.ndarray
    valid: np.ndarray


def regrid_ocean_mask_by_overlap(
    source_ocean: np.ndarray,
    source_longitude_deg: np.ndarray,
    source_latitude_deg: np.ndarray,
    target_longitude_deg: np.ndarray,
    target_latitude_deg: np.ndarray,
) -> np.ndarray:
    """Remap a regular land/ocean mask by exact spherical overlap area.

    A target cell is ocean when at least half of its spherical area overlaps
    source ocean. Longitude is periodic and a 50/50 tie is classified as ocean.
    """
    source_mask = np.asarray(source_ocean, dtype=bool)
    source_lon = np.asarray(source_longitude_deg, dtype=np.float64)
    source_lat = np.asarray(source_latitude_deg, dtype=np.float64)
    target_lon = np.asarray(target_longitude_deg, dtype=np.float64)
    target_lat = np.asarray(target_latitude_deg, dtype=np.float64)
    if source_mask.shape != (source_lat.size, source_lon.size):
        raise ValueError("Source ocean mask does not match its coordinates")
    if source_lon.size < 2 or source_lat.size < 2 or target_lon.size < 2 or target_lat.size < 2:
        raise ValueError("Ocean-mask remapping requires at least two coordinates per axis")
    for values, name in (
        (source_lon, "source longitude"),
        (source_lat, "source latitude"),
        (target_lon, "target longitude"),
        (target_lat, "target latitude"),
    ):
        spacing = np.diff(values)
        if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0]):
            raise ValueError(f"Expected {name} to be strictly increasing and regular")
    if not np.isclose(source_lon[-1] - source_lon[0] + np.diff(source_lon)[0], 360.0):
        raise ValueError("Source longitude must cover one periodic revolution")
    if not np.isclose(target_lon[-1] - target_lon[0] + np.diff(target_lon)[0], 360.0):
        raise ValueError("Target longitude must cover one periodic revolution")

    source_lon_step = float(np.diff(source_lon)[0])
    source_lat_step = float(np.diff(source_lat)[0])
    target_lon_step = float(np.diff(target_lon)[0])
    target_lat_step = float(np.diff(target_lat)[0])
    source_lat_edges = np.concatenate(
        ([source_lat[0] - source_lat_step / 2.0], source_lat + source_lat_step / 2.0)
    )
    target_lat_edges = np.concatenate(
        ([target_lat[0] - target_lat_step / 2.0], target_lat + target_lat_step / 2.0)
    )
    if target_lat_edges[0] < source_lat_edges[0] or target_lat_edges[-1] > source_lat_edges[-1]:
        raise ValueError("Target latitude cells extend beyond the source mask")

    # Spherical rectangle area is separable into longitude overlap and the
    # difference of sine(latitude) at the overlap edges.
    latitude_overlap = np.zeros((target_lat.size, source_lat.size), dtype=np.float64)
    for i in range(target_lat.size):
        lower = np.maximum(target_lat_edges[i], source_lat_edges[:-1])
        upper = np.minimum(target_lat_edges[i + 1], source_lat_edges[1:])
        overlap = upper > lower
        latitude_overlap[i, overlap] = (
            np.sin(np.deg2rad(upper[overlap])) - np.sin(np.deg2rad(lower[overlap]))
        )

    longitude_overlap = np.zeros((target_lon.size, source_lon.size), dtype=np.float64)
    source_start = source_lon - source_lon_step / 2.0
    source_end = source_lon + source_lon_step / 2.0
    for j, center in enumerate(target_lon):
        target_start = center - target_lon_step / 2.0
        target_end = center + target_lon_step / 2.0
        for shift in (-360.0, 0.0, 360.0):
            lower = np.maximum(target_start, source_start + shift)
            upper = np.minimum(target_end, source_end + shift)
            longitude_overlap[j] += np.maximum(upper - lower, 0.0)

    ocean_area = latitude_overlap @ source_mask.astype(np.float64) @ longitude_overlap.T
    total_area = latitude_overlap.sum(axis=1)[:, None] * longitude_overlap.sum(axis=1)[None, :]
    return ocean_area >= 0.5 * total_area


def _require_regular_coordinate(values: np.ndarray, expected_size: int, name: str) -> None:
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(f"Expected {name} to contain {expected_size} coordinates")
    spacing = np.diff(values)
    if not np.all(spacing > 0.0) or not np.allclose(spacing, 1.0):
        raise ValueError(f"Expected {name} to be a strictly increasing 1-degree grid")


def _attribute_text(value: object) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def load_mld(path: str) -> MixedLayerDepth:
    """Load and validate the de Boyer Montegut v2023 monthly climatology."""
    with netcdf_file(path, "r", mmap=False) as dataset:
        required = {"lon", "lat", "time", "mask", "mld_dr003"}
        missing = required.difference(dataset.variables)
        if missing:
            raise ValueError(f"MLD dataset is missing variables: {', '.join(sorted(missing))}")

        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        longitude_unit = getattr(dataset.variables["lon"], "unit", b"")
        latitude_unit = getattr(dataset.variables["lat"], "unit", b"")
        time = np.asarray(dataset.variables["time"][:])
        source_ocean = np.asarray(dataset.variables["mask"][:]) == 1
        variable = dataset.variables["mld_dr003"]
        raw = np.asarray(variable[:], dtype=np.float64)
        fill_value = getattr(variable, "_FillValue", None)
        mld_unit = getattr(variable, "unit", b"")

    _require_regular_coordinate(longitude, 360, "longitude")
    _require_regular_coordinate(latitude, 180, "latitude")
    if not np.allclose(longitude[[0, -1]], [-179.5, 179.5]):
        raise ValueError("Expected longitude cell centres from -179.5 to 179.5 degrees")
    if not np.allclose(latitude[[0, -1]], [-89.5, 89.5]):
        raise ValueError("Expected latitude cell centres from -89.5 to 89.5 degrees")
    if _attribute_text(longitude_unit) != "degrees_east":
        raise ValueError("Expected longitude units to be degrees_east")
    if _attribute_text(latitude_unit) != "degrees_north":
        raise ValueError("Expected latitude units to be degrees_north")
    if _attribute_text(mld_unit) != "m":
        raise ValueError("Expected mld_dr003 units to be metres")
    if time.ndim != 1 or time.size != 12:
        raise ValueError("Expected exactly 12 monthly climatology time steps")
    if raw.shape != (12, 180, 360):
        raise ValueError("Expected mld_dr003 dimensions to be (time=12, lat=180, lon=360)")
    if source_ocean.shape != (180, 360):
        raise ValueError("MLD mask dimensions do not match the latitude/longitude grid")
    if fill_value is None:
        raise ValueError("Expected mld_dr003 to define an _FillValue")

    valid = np.isfinite(raw) & ~np.isclose(raw, float(fill_value)) & source_ocean[None, :, :]
    monthly_m = np.where(valid, raw, np.nan)
    return MixedLayerDepth(longitude, latitude, monthly_m, source_ocean)


def maximum_mld(monthly_m: np.ndarray) -> MaximumMixedLayerDepth:
    """Take the maximum of available months and retain the valid-month count."""
    values = np.asarray(monthly_m, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 12:
        raise ValueError("Monthly MLD must have shape (12, latitude, longitude)")

    valid = np.isfinite(values)
    month_count = valid.sum(axis=0).astype(np.int16)
    maximum = np.max(np.where(valid, values, -np.inf), axis=0)
    maximum[month_count == 0] = np.nan
    return MaximumMixedLayerDepth(maximum, month_count)


def regrid_maximum_mld(
    source_maximum_m: np.ndarray,
    source_month_count: np.ndarray,
    source_longitude_deg: np.ndarray,
    source_latitude_deg: np.ndarray,
    target_longitude_deg: np.ndarray,
    target_latitude_deg: np.ndarray,
    target_valid: np.ndarray,
) -> RegriddedMixedLayerDepth:
    """Periodically, mask-aware bilinearly interpolate H to a target grid.

    Invalid source corners are excluded and the remaining bilinear weights are
    renormalized. Coverage is the minimum month count among valid contributors
    with nonzero weight.
    """
    source = np.asarray(source_maximum_m, dtype=np.float64)
    counts = np.asarray(source_month_count)
    source_lon = np.asarray(source_longitude_deg, dtype=np.float64)
    source_lat = np.asarray(source_latitude_deg, dtype=np.float64)
    target_lon = np.asarray(target_longitude_deg, dtype=np.float64)
    target_lat = np.asarray(target_latitude_deg, dtype=np.float64)
    target_mask = np.asarray(target_valid, dtype=bool)

    if source.shape != (source_lat.size, source_lon.size) or counts.shape != source.shape:
        raise ValueError("Source MLD fields do not match their coordinates")
    if target_mask.shape != (target_lat.size, target_lon.size):
        raise ValueError("Target mask does not match its coordinates")
    _require_regular_coordinate(source_lon, source_lon.size, "source longitude")
    _require_regular_coordinate(source_lat, source_lat.size, "source latitude")
    if target_lat.min() < source_lat.min() or target_lat.max() > source_lat.max():
        raise ValueError("Target latitudes extend beyond the MLD source grid")

    lon_position = np.mod(target_lon - source_lon[0], 360.0)
    lon_lower = np.floor(lon_position).astype(int) % source_lon.size
    lon_fraction = lon_position - np.floor(lon_position)
    lon_upper = (lon_lower + 1) % source_lon.size

    lat_position = target_lat - source_lat[0]
    lat_lower = np.floor(lat_position).astype(int)
    lat_fraction = lat_position - lat_lower
    lat_upper = np.minimum(lat_lower + 1, source_lat.size - 1)

    fy = lat_fraction[:, None]
    fx = lon_fraction[None, :]
    numerator = np.zeros(target_mask.shape, dtype=np.float64)
    denominator = np.zeros(target_mask.shape, dtype=np.float64)
    coverage = np.full(target_mask.shape, 13, dtype=np.int16)

    contributors = (
        (lat_lower, lon_lower, (1.0 - fy) * (1.0 - fx)),
        (lat_lower, lon_upper, (1.0 - fy) * fx),
        (lat_upper, lon_lower, fy * (1.0 - fx)),
        (lat_upper, lon_upper, fy * fx),
    )
    for latitude_index, longitude_index, weight in contributors:
        contributor = source[np.ix_(latitude_index, longitude_index)]
        contributor_count = counts[np.ix_(latitude_index, longitude_index)]
        usable = np.isfinite(contributor) & (weight > 0.0)
        effective_weight = np.where(usable, weight, 0.0)
        numerator += np.where(usable, contributor, 0.0) * effective_weight
        denominator += effective_weight
        coverage = np.where(usable, np.minimum(coverage, contributor_count), coverage)

    valid = target_mask & (denominator > 0.0)
    maximum = np.full(target_mask.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=maximum, where=valid)
    coverage[~valid] = 0
    return RegriddedMixedLayerDepth(maximum, coverage, valid)
