"""Shared geostrophic-velocity reconstruction for production and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dyntopo import (
    EARTH_ROTATION_RAD_PER_S,
    GRAVITY_M_PER_S2,
    equatorial_beta_plane_velocity,
    spherical_gradient,
)


DEFAULT_FIT_HALF_WIDTH_DEG = 3.5
DEFAULT_SPLINE_SMOOTHING_M = 0.01
DEFAULT_BETA_PLANE_CORE_LATITUDE_DEG = 0.25
DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG = 3.0


@dataclass(frozen=True)
class GeostrophicVelocity:
    """Blended velocity plus the component solutions used by diagnostics."""

    east: np.ndarray
    north: np.ndarray
    valid: np.ndarray
    beta_east: np.ndarray
    beta_north: np.ndarray
    beta_valid: np.ndarray
    ordinary_east: np.ndarray
    ordinary_north: np.ndarray
    ordinary_valid: np.ndarray
    blend_weight: np.ndarray


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def reconstruct_geostrophic_velocity(
    height_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    fit_half_width_deg: float = DEFAULT_FIT_HALF_WIDTH_DEG,
    smoothing_m: float = DEFAULT_SPLINE_SMOOTHING_M,
    core_latitude_deg: float = DEFAULT_BETA_PLANE_CORE_LATITUDE_DEG,
    blend_half_width_deg: float = DEFAULT_BETA_PLANE_BLEND_LATITUDE_DEG,
) -> GeostrophicVelocity:
    """Reconstruct ordinary and equatorial geostrophic velocity.

    Ordinary f-plane geostrophy is used outside the equatorial transition.
    Cubic-spline beta-plane estimates are used in the core and blended into the
    ordinary solution with a cubic smoothstep through the transition.
    """
    if core_latitude_deg < 0.0:
        raise ValueError("Beta-plane core latitude must be non-negative")
    if blend_half_width_deg <= core_latitude_deg:
        raise ValueError("Blend half-width must exceed the beta-plane core latitude")

    grad_east, grad_north, magnitude = spherical_gradient(
        height_m,
        longitude_deg,
        latitude_deg,
    )
    coriolis = (
        2.0
        * EARTH_ROTATION_RAD_PER_S
        * np.sin(np.deg2rad(latitude_deg))[:, None]
    )
    ordinary_valid = np.isfinite(magnitude) & (np.abs(latitude_deg)[:, None] > 0.0)
    ordinary_east = np.full_like(grad_east, np.nan)
    ordinary_north = np.full_like(grad_north, np.nan)
    np.divide(
        -GRAVITY_M_PER_S2 * grad_north,
        coriolis,
        out=ordinary_east,
        where=ordinary_valid,
    )
    np.divide(
        GRAVITY_M_PER_S2 * grad_east,
        coriolis,
        out=ordinary_north,
        where=ordinary_valid,
    )
    ordinary_valid &= np.isfinite(ordinary_east) & np.isfinite(ordinary_north)

    beta_east, beta_north, beta_valid = equatorial_beta_plane_velocity(
        height_m,
        longitude_deg,
        latitude_deg,
        fit_half_width_deg=fit_half_width_deg,
        smoothing_m=smoothing_m,
    )
    latitude_abs = np.abs(latitude_deg)[:, None]
    blend_weight = np.broadcast_to(
        _smoothstep(
            (latitude_abs - core_latitude_deg)
            / (blend_half_width_deg - core_latitude_deg)
        ),
        np.shape(height_m),
    ).copy()
    beta_only = (latitude_abs <= core_latitude_deg) & beta_valid
    blended = (
        (latitude_abs > core_latitude_deg)
        & (latitude_abs < blend_half_width_deg)
        & beta_valid
        & ordinary_valid
    )

    east = ordinary_east.copy()
    north = ordinary_north.copy()
    east[beta_only] = beta_east[beta_only]
    north[beta_only] = beta_north[beta_only]
    east[blended] = (
        (1.0 - blend_weight[blended]) * beta_east[blended]
        + blend_weight[blended] * ordinary_east[blended]
    )
    north[blended] = (
        (1.0 - blend_weight[blended]) * beta_north[blended]
        + blend_weight[blended] * ordinary_north[blended]
    )
    valid = np.isfinite(east) & np.isfinite(north)
    return GeostrophicVelocity(
        east=east,
        north=north,
        valid=valid,
        beta_east=beta_east,
        beta_north=beta_north,
        beta_valid=beta_valid,
        ordinary_east=ordinary_east,
        ordinary_north=ordinary_north,
        ordinary_valid=ordinary_valid,
        blend_weight=blend_weight,
    )
