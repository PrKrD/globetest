"""Calendar alignment, physical signs and provenance for index validation."""

import hashlib
import json

import numpy as np
import pytest
import xarray as xr

from app.mslp_indices import compare_indices, read_monthly_index


def reference_tables(directory, tables):
    provenance = {}
    for name, (ids, values) in tables.items():
        if name in {"AO", "AAO"}:
            text = "\n".join(f"{key//12} {key%12+1} {value:.16g}" for key, value in zip(ids, values))
        else:
            lines = ["YEAR JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC"]
            for year in np.unique(ids//12):
                lines.append(str(year) + " " + " ".join(f"{v:.16g}" for v in values[ids//12 == year]))
            text = "\n".join(lines)
        raw = text.encode()
        (directory / f"{name}.txt").write_bytes(raw)
        provenance[name] = {"url": f"https://example.invalid/{name}.txt",
                            "retrieved_utc": "2026-11-06T00:00:00+00:00",
                            "sha256": hashlib.sha256(raw).hexdigest()}
    (directory / "provenance.json").write_text(json.dumps(provenance))


def pc_dataset():
    time = np.arange(np.datetime64("1979-01"), np.datetime64("1985-01")).astype("datetime64[ns]")
    rng = np.random.default_rng(51)
    pc = rng.normal(size=(len(time), 2))
    pc -= pc.mean(axis=0)
    pc[:, 1] -= np.dot(pc[:, 0], pc[:, 1])/np.dot(pc[:, 0], pc[:, 0])*pc[:, 0]
    pc /= pc.std(axis=0, ddof=1)
    maps = np.tile(np.array([-1, -1, -1, 1, 1, 1])[None, :, None], (2, 1, 4))
    return xr.Dataset({"standardized_pc": (("time", "mode"), pc),
                       "regression_map": (("mode", "latitude", "longitude"), maps)},
                      coords={"time": time, "mode": [1, 2],
                              "latitude": [20., 40., 50., 70., 80., 90.],
                              "longitude": [0., 90., 180., 270.]})


def test_cpc_and_bas_formats_retain_partial_final_year(tmp_path):
    ids = np.arange(1979*12, 1981*12+3)
    values = np.arange(len(ids))*.1
    reference_tables(tmp_path, {"AO": (ids, values), "SAM": (ids, values)})
    for name in ("AO", "SAM"):
        actual_ids, actual_values = read_monthly_index((tmp_path / f"{name}.txt").read_text(), name)
        np.testing.assert_array_equal(actual_ids, ids)
        np.testing.assert_allclose(actual_values, values)


@pytest.mark.parametrize("fault", ["gap", "duplicate", "sentinel", "sentinel_99", "bad_month"])
def test_bad_reference_samples_rejected(fault):
    lines = [f"{key//12} {key%12+1} 0.5" for key in range(1979*12, 1982*12)]
    if fault == "gap":
        del lines[10]
    elif fault == "duplicate":
        lines.append(lines[0])
    elif fault == "sentinel":
        lines[0] = "1979 1 -999"
    elif fault == "sentinel_99":
        lines[0] = "1979 1 -99.99"
    else:
        lines[0] = "1979 0 0.5"
    with pytest.raises(ValueError):
        read_monthly_index("\n".join(lines), "AO")


def test_calendar_join_and_physical_sign_are_invariant_to_eof_sign(tmp_path):
    original = pc_dataset()
    flipped = original.copy(deep=True)
    flipped["standardized_pc"] *= -1
    flipped["regression_map"] *= -1
    # Deliberately start reference one month later to catch positional alignment.
    ids = np.arange(1979*12+1, 1985*12)
    reference_tables(tmp_path, {"AO": (ids, -original.standardized_pc.values[1:, 0])})
    result = compare_indices({("north", 1940): original, ("north", 1979): flipped}, tmp_path)
    earlier, later = result["results"]
    assert earlier["start"] == "1979-02" and earlier["n"] == 71
    assert earlier["pc1_multiplier"] == -1
    assert later["pc1_multiplier"] == 1
    assert earlier["raw_pc1_r"] == pytest.approx(-1)
    assert later["raw_pc1_r"] == pytest.approx(1)
    assert earlier["aligned_pc1_r"] == later["aligned_pc1_r"] == pytest.approx(1)
    np.testing.assert_array_equal(original.standardized_pc, -flipped.standardized_pc)


def test_reference_checksum_mismatch_rejected(tmp_path):
    ds = pc_dataset()
    ids = np.arange(1979*12, 1985*12)
    reference_tables(tmp_path, {"AO": (ids, -ds.standardized_pc.values[:, 0])})
    with (tmp_path / "AO.txt").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        compare_indices({("north", 1979): ds}, tmp_path)


def test_station_pressure_reconstruction_checks_marshall_sam_sign(tmp_path):
    ids = np.arange(1971*12, 2001*12)
    rng = np.random.default_rng(8)
    p40 = 1010 + rng.normal(0, 3, len(ids))
    p65 = 985 + rng.normal(0, 6, len(ids))
    z40, z65 = np.empty_like(p40), np.empty_like(p65)
    for month in range(12):
        selected = ids % 12 == month
        for source, result in ((p40, z40), (p65, z65)):
            result[selected] = (source[selected]-source[selected].mean())/source[selected].std(ddof=0)
    sam = z40-z65
    reference_tables(tmp_path, {"SAM": (ids, sam), "P40": (ids, p40), "P65": (ids, p65)})
    ds = pc_dataset().assign_coords(latitude=[-20., -40., -50., -70., -80., -90.])
    sample = sam[(ids >= 1979*12) & (ids < 1985*12)]
    ds.standardized_pc.values[:, 0] = -(sample-sample.mean())/sample.std(ddof=1)
    result = compare_indices({("south", 1979): ds}, tmp_path)
    assert result["sam_formula_correlation"] == pytest.approx(1, abs=1e-12)
    assert result["results"][0]["aligned_pc1_r"] == pytest.approx(1)
