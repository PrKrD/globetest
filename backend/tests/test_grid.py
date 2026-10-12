import numpy as np

from app.export import _coarsen_mask
from app.grid import build_global_grid


def test_global_grid_dimensions_and_ranges() -> None:
    grid = build_global_grid(0.25)
    assert grid.n_lon == 1440
    assert grid.n_lat == 720
    assert np.isclose(grid.lon_edges[0], -180.0)
    assert np.isclose(grid.lon_edges[-1], 180.0)
    assert np.isclose(grid.lat_edges[0], -90.0)
    assert np.isclose(grid.lat_edges[-1], 90.0)


def test_no_longitude_gaps_or_duplicates() -> None:
    grid = build_global_grid(1.0)
    diffs = np.diff(grid.lon_edges)
    assert np.allclose(diffs, 1.0)
    assert np.unique(grid.lon_edges).size == grid.lon_edges.size


def test_cell_id_uniqueness_and_row_major() -> None:
    grid = build_global_grid(2.0)
    assert int(grid.cell_ids[0, 0]) == 0
    assert int(grid.cell_ids[0, 1]) == 1
    assert int(grid.cell_ids[1, 0]) == grid.n_lon
    assert np.unique(grid.cell_ids).size == grid.cell_ids.size


def test_coarsen_mask_shape_and_values() -> None:
    mask = np.zeros((4, 8), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    coarse = _coarsen_mask(mask, factor=2)
    assert coarse.shape == (2, 4)
    assert set(np.unique(coarse)).issubset({0, 1})
    assert coarse[0, 0] == 1
