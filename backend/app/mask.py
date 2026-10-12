"""Land/sea mask generation."""

from __future__ import annotations

from pathlib import Path
import zipfile
import urllib.request

import geopandas as gpd
import numpy as np
import shapely

from .config import NATURAL_EARTH_LAND_SHP, NATURAL_EARTH_LAND_ZIP_URL
from .grid import GridDefinition


class CenterPointMaskClassifier:
    """Classify cells from center points.

    Kept as a dedicated class so it can later be replaced with
    fractional ocean coverage without changing calling code.
    """

    def classify(self, grid: GridDefinition, land_geometry) -> np.ndarray:
        lon2d, lat2d = np.meshgrid(grid.lon_centers, grid.lat_centers)
        is_land_flat = shapely.contains_xy(land_geometry, lon2d.ravel(), lat2d.ravel())
        return np.asarray(is_land_flat, dtype=bool).reshape(grid.n_lat, grid.n_lon)


def ensure_natural_earth_land(data_dir: Path) -> Path:
    if NATURAL_EARTH_LAND_SHP:
        local = Path(NATURAL_EARTH_LAND_SHP).expanduser().resolve()
        if local.exists():
            return local
        raise FileNotFoundError(f"NATURAL_EARTH_LAND_SHP path does not exist: {local}")

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ne_50m_land.zip"
    shp_path = data_dir / "ne_50m_land.shp"

    if shp_path.exists():
        return shp_path

    try:
        urllib.request.urlretrieve(NATURAL_EARTH_LAND_ZIP_URL, zip_path)
    except Exception as exc:  # pragma: no cover - network-dependent path
        raise RuntimeError(
            "Unable to download Natural Earth land polygons. "
            "Set NATURAL_EARTH_LAND_SHP to a local ne_50m_land.shp path "
            "or place Natural Earth files in backend/data."
        ) from exc
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    if not shp_path.exists():
        raise FileNotFoundError("Natural Earth land shapefile was not extracted correctly")

    return shp_path


def load_land_geometry(data_dir: Path):
    shp_path = ensure_natural_earth_land(data_dir)
    land = gpd.read_file(shp_path)
    return land.geometry.union_all()


def build_land_mask(grid: GridDefinition, data_dir: Path) -> np.ndarray:
    land_geometry = load_land_geometry(data_dir)
    classifier = CenterPointMaskClassifier()
    mask = classifier.classify(grid, land_geometry)

    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [False, True])):
        raise ValueError("Land/ocean mask contains invalid values")

    return mask
