"""Geostrophic contribution to mixed-layer subduction on a spherical grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dyntopo import EARTH_RADIUS_METERS, coordinate_edges


@dataclass(frozen=True)
class SubductionConservation:
    """Finite-volume transport budget, with outward fluxes in cubic metres per second."""

    coastline_face_count: int
    missing_data_face_count: int
    open_domain_face_count: int
    coastline_outward_m3_per_s: float
    missing_data_outward_m3_per_s: float
    open_domain_outward_m3_per_s: float
    total_outward_m3_per_s: float
    area_integrated_subduction_m3_per_s: float
    closure_residual_m3_per_s: float

    @property
    def relative_closure_residual(self) -> float:
        """Return closure error relative to the larger global budget term."""
        scale = max(
            abs(self.area_integrated_subduction_m3_per_s),
            abs(self.total_outward_m3_per_s),
            1e-30,
        )
        return self.closure_residual_m3_per_s / scale


@dataclass(frozen=True)
class GeostrophicSubduction:
    """Signed finite-volume subduction rate and its conservation budget."""

    rate_m_per_s: np.ndarray
    valid: np.ndarray
    zonal_rate_m_per_s: np.ndarray
    meridional_rate_m_per_s: np.ndarray
    area_m2: np.ndarray
    conservation: SubductionConservation


def geostrophic_subduction_rate(
    mixed_layer_depth_m: np.ndarray,
    velocity_east_m_per_s: np.ndarray,
    velocity_north_m_per_s: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    valid: np.ndarray,
    physical_ocean: np.ndarray,
) -> GeostrophicSubduction:
    """Return conservative negative divergence of depth-integrated transport.

    A single flux is constructed for every face shared by two usable cells.
    Physical coastlines have zero normal flux. Faces next to oceanic missing
    data and the north/south domain edges use the transport in the adjacent
    usable cell. Longitude is periodic.
    """
    depth = np.asarray(mixed_layer_depth_m, dtype=np.float64)
    east_velocity = np.asarray(velocity_east_m_per_s, dtype=np.float64)
    north_velocity = np.asarray(velocity_north_m_per_s, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool).copy()
    ocean = np.asarray(physical_ocean, dtype=bool)
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    shape = (latitude.size, longitude.size)

    if depth.shape != shape or east_velocity.shape != shape or north_velocity.shape != shape:
        raise ValueError("Subduction fields do not match their coordinates")
    if usable.shape != shape:
        raise ValueError("Subduction validity mask does not match its coordinates")
    if ocean.shape != shape:
        raise ValueError("Physical-ocean mask does not match its coordinates")
    if longitude.size < 2 or latitude.size < 2:
        raise ValueError("Subduction requires at least two coordinates in each direction")
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

    usable &= np.isfinite(depth) & np.isfinite(east_velocity) & np.isfinite(north_velocity)
    if np.any(usable & ~ocean):
        raise ValueError("Usable transport cells must be classified as physical ocean")

    latitude_edges = np.deg2rad(coordinate_edges(np.rad2deg(latitude)))
    if latitude_edges[0] < -0.5 * np.pi or latitude_edges[-1] > 0.5 * np.pi:
        raise ValueError("Latitude cell edges must remain between the poles")
    delta_longitude = float(longitude_spacing[0])
    delta_latitude = float(latitude_spacing[0])
    row_area = EARTH_RADIUS_METERS**2 * delta_longitude * (
        np.sin(latitude_edges[1:]) - np.sin(latitude_edges[:-1])
    )
    area = np.broadcast_to(row_area[:, None], shape).copy()

    east_transport = depth * east_velocity
    north_transport = depth * north_velocity

    # east_face[i, j] is positive eastward across the face between cell j and
    # cell j+1. The final longitude face wraps to cell zero.
    east_neighbour_usable = np.roll(usable, -1, axis=1)
    east_neighbour_ocean = np.roll(ocean, -1, axis=1)
    east_neighbour_transport = np.roll(east_transport, -1, axis=1)
    shared_east = usable & east_neighbour_usable
    west_wet_only = usable & ~east_neighbour_usable
    east_wet_only = ~usable & east_neighbour_usable
    east_coast = (west_wet_only & ~east_neighbour_ocean) | (
        east_wet_only & ~ocean
    )
    east_missing = (west_wet_only & east_neighbour_ocean) | (
        east_wet_only & ocean
    )
    meridional_face_length = EARTH_RADIUS_METERS * delta_latitude
    east_face = np.zeros(shape, dtype=np.float64)
    east_face[shared_east] = (
        0.5 * (east_transport + east_neighbour_transport) * meridional_face_length
    )[shared_east]
    east_face[west_wet_only & east_neighbour_ocean] = (
        east_transport * meridional_face_length
    )[west_wet_only & east_neighbour_ocean]
    east_face[east_wet_only & ocean] = (
        east_neighbour_transport * meridional_face_length
    )[east_wet_only & ocean]

    # north_face[k, j] is positive northward at latitude edge k. Interior wet
    # faces are shared; edge and missing-data faces use the wet-side value.
    north_face = np.zeros((latitude.size + 1, longitude.size), dtype=np.float64)
    south_usable = usable[:-1]
    north_usable = usable[1:]
    south_ocean = ocean[:-1]
    north_ocean = ocean[1:]
    shared_north = south_usable & north_usable
    south_only = south_usable & ~north_usable
    north_only = ~south_usable & north_usable
    north_coast = (south_only & ~north_ocean) | (north_only & ~south_ocean)
    north_missing = (south_only & north_ocean) | (north_only & south_ocean)
    interior_face_length = (
        EARTH_RADIUS_METERS * np.cos(latitude_edges[1:-1])[:, None] * delta_longitude
    )
    interior_north_face = north_face[1:-1]
    interior_north_face[shared_north] = (
        0.5 * (north_transport[:-1] + north_transport[1:]) * interior_face_length
    )[shared_north]
    interior_north_face[south_only & north_ocean] = (
        north_transport[:-1] * interior_face_length
    )[south_only & north_ocean]
    interior_north_face[north_only & south_ocean] = (
        north_transport[1:] * interior_face_length
    )[north_only & south_ocean]

    south_open = usable[0]
    north_open = usable[-1]
    south_face_length = EARTH_RADIUS_METERS * np.cos(latitude_edges[0]) * delta_longitude
    north_face_length = EARTH_RADIUS_METERS * np.cos(latitude_edges[-1]) * delta_longitude
    north_face[0, south_open] = north_transport[0, south_open] * south_face_length
    north_face[-1, north_open] = north_transport[-1, north_open] * north_face_length

    zonal_net_outward = east_face - np.roll(east_face, 1, axis=1)
    meridional_net_outward = north_face[1:] - north_face[:-1]
    zonal_rate = np.full(shape, np.nan)
    meridional_rate = np.full(shape, np.nan)
    zonal_rate[usable] = (-zonal_net_outward / area)[usable]
    meridional_rate[usable] = (-meridional_net_outward / area)[usable]
    rate = zonal_rate + meridional_rate

    east_missing_outward = float(
        np.sum(east_face[west_wet_only & east_neighbour_ocean])
        - np.sum(east_face[east_wet_only & ocean])
    )
    north_missing_outward = float(
        np.sum(interior_north_face[south_only & north_ocean])
        - np.sum(interior_north_face[north_only & south_ocean])
    )
    coastline_outward = float(
        np.sum(east_face[east_coast]) + np.sum(interior_north_face[north_coast])
    )
    missing_outward = east_missing_outward + north_missing_outward
    open_outward = float(
        np.sum(north_face[-1, north_open]) - np.sum(north_face[0, south_open])
    )
    total_outward = coastline_outward + missing_outward + open_outward
    area_integral = float(np.sum(rate[usable] * area[usable]))
    conservation = SubductionConservation(
        coastline_face_count=int(np.sum(east_coast) + np.sum(north_coast)),
        missing_data_face_count=int(np.sum(east_missing) + np.sum(north_missing)),
        open_domain_face_count=int(np.sum(south_open) + np.sum(north_open)),
        coastline_outward_m3_per_s=coastline_outward,
        missing_data_outward_m3_per_s=missing_outward,
        open_domain_outward_m3_per_s=open_outward,
        total_outward_m3_per_s=total_outward,
        area_integrated_subduction_m3_per_s=area_integral,
        closure_residual_m3_per_s=area_integral + total_outward,
    )
    return GeostrophicSubduction(rate, usable, zonal_rate, meridional_rate, area, conservation)
