"""Scientific ocean fields on the native dynamic-topography grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dyntopo import DynamicTopography
from app.mld import (
    MixedLayerDepth,
    RegriddedMixedLayerDepth,
    maximum_mld,
    regrid_maximum_mld,
    regrid_ocean_mask_by_overlap,
)
from app.subduction import GeostrophicSubduction, geostrophic_subduction_rate
from app.velocity import GeostrophicVelocity, reconstruct_geostrophic_velocity


@dataclass(frozen=True)
class OceanFields:
    """Unrounded scientific arrays, validity masks, and conservation budget."""

    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    physical_ocean: np.ndarray
    mld: RegriddedMixedLayerDepth
    velocity: GeostrophicVelocity
    subduction_geo: GeostrophicSubduction


def compute_ocean_fields(
    topography: DynamicTopography,
    mld_source: MixedLayerDepth,
) -> OceanFields:
    """Compute maximum MLD, geostrophic velocity, and conservative subduction.

    Missing scientific values remain NaN with their original validity masks.
    No display preparation or file output is performed.
    """
    mld_maximum = maximum_mld(mld_source.monthly_m)
    physical_ocean = regrid_ocean_mask_by_overlap(
        mld_source.source_ocean,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        topography.longitude_deg,
        topography.latitude_deg,
    )
    regridded_mld = regrid_maximum_mld(
        mld_maximum.maximum_m,
        mld_maximum.month_count,
        mld_source.longitude_deg,
        mld_source.latitude_deg,
        topography.longitude_deg,
        topography.latitude_deg,
        physical_ocean,
    )
    velocity = reconstruct_geostrophic_velocity(
        topography.height_m, topography.longitude_deg, topography.latitude_deg
    )
    subduction = geostrophic_subduction_rate(
        regridded_mld.maximum_m,
        velocity.east,
        velocity.north,
        topography.longitude_deg,
        topography.latitude_deg,
        regridded_mld.valid & velocity.valid,
        physical_ocean,
    )
    return OceanFields(
        longitude_deg=topography.longitude_deg,
        latitude_deg=topography.latitude_deg,
        physical_ocean=physical_ocean,
        mld=regridded_mld,
        velocity=velocity,
        subduction_geo=subduction,
    )
