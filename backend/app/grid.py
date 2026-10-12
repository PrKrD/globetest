"""Latitude/longitude grid definition and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridDefinition:
    """Regular global lat/lon grid.

    Indexing convention (row-major):
    - lat index i increases south -> north
    - lon index j increases west -> east
    - cell_id = i * n_lon + j

    Edges store cell boundaries, centers store cell center coordinates.
    """

    resolution_deg: float
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    lat_centers: np.ndarray
    lon_centers: np.ndarray
    cell_ids: np.ndarray

    @property
    def n_lat(self) -> int:
        return int(self.lat_centers.size)

    @property
    def n_lon(self) -> int:
        return int(self.lon_centers.size)


def _validated_cell_count(span_deg: float, resolution_deg: float, axis_name: str) -> int:
    raw_count = span_deg / resolution_deg
    rounded = int(round(raw_count))
    if not np.isclose(raw_count, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{axis_name} span {span_deg}° is not evenly divisible by resolution {resolution_deg}°"
        )
    return rounded


def build_global_grid(resolution_deg: float) -> GridDefinition:
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")

    n_lon = _validated_cell_count(360.0, resolution_deg, "Longitude")
    n_lat = _validated_cell_count(180.0, resolution_deg, "Latitude")

    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1, dtype=np.float64)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1, dtype=np.float64)
    lon_centers = lon_edges[:-1] + (resolution_deg / 2.0)
    lat_centers = lat_edges[:-1] + (resolution_deg / 2.0)

    # row-major unique integer IDs for every cell
    cell_ids = np.arange(n_lat * n_lon, dtype=np.int64).reshape(n_lat, n_lon)

    grid = GridDefinition(
        resolution_deg=resolution_deg,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        cell_ids=cell_ids,
    )
    validate_grid(grid)
    return grid


def validate_grid(grid: GridDefinition) -> None:
    expected_n_lon = int(round(360.0 / grid.resolution_deg))
    expected_n_lat = int(round(180.0 / grid.resolution_deg))

    if grid.n_lon != expected_n_lon:
        raise ValueError(f"Expected {expected_n_lon} longitude cells, got {grid.n_lon}")
    if grid.n_lat != expected_n_lat:
        raise ValueError(f"Expected {expected_n_lat} latitude cells, got {grid.n_lat}")

    if not np.isclose(grid.lon_edges[0], -180.0) or not np.isclose(grid.lon_edges[-1], 180.0):
        raise ValueError("Longitude edges must span [-180, 180]")
    if not np.isclose(grid.lat_edges[0], -90.0) or not np.isclose(grid.lat_edges[-1], 90.0):
        raise ValueError("Latitude edges must span [-90, 90]")

    lon_step = np.diff(grid.lon_edges)
    lat_step = np.diff(grid.lat_edges)
    if np.any(~np.isclose(lon_step, grid.resolution_deg)):
        raise ValueError("Longitude grid contains gaps or uneven spacing")
    if np.any(~np.isclose(lat_step, grid.resolution_deg)):
        raise ValueError("Latitude grid contains gaps or uneven spacing")

    if np.unique(grid.lon_edges).size != grid.lon_edges.size:
        raise ValueError("Duplicate longitude edges found (dateline duplication)")
    if np.unique(grid.cell_ids).size != grid.cell_ids.size:
        raise ValueError("Cell IDs must be unique")

    if np.min(grid.lat_edges) < -90.0 or np.max(grid.lat_edges) > 90.0:
        raise ValueError("Latitude bounds must remain within [-90, 90]")
