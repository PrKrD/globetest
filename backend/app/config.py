"""Runtime configuration for grid generation and export."""

from pathlib import Path
import os

# Single obvious control for the target global grid resolution.
GRID_RESOLUTION_DEG = float(os.getenv("GRID_RESOLUTION_DEG", "0.25"))

# Coarse product used by the first rendering prototype (Phase 2).
COARSE_RENDER_RESOLUTION_DEG = float(os.getenv("COARSE_RENDER_RESOLUTION_DEG", "2.0"))

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_ROOT / "data"
FRONTEND_PUBLIC_DATA_DIR = BACKEND_ROOT.parent / "frontend" / "public" / "data"

# Natural Earth (public domain) land polygons.
NATURAL_EARTH_LAND_ZIP_URL = (
    "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"
)
NATURAL_EARTH_LAND_SHP = os.getenv("NATURAL_EARTH_LAND_SHP")
