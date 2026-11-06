"""Build the globe-ready velocity and maximum mixed-layer-depth maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import DATA_DIR, FRONTEND_PUBLIC_DATA_DIR
from app.dyntopo import load_maximenko
from app.globe_export import export_globe_json
from app.mld import load_mld
from app.ocean_fields import compute_ocean_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the combined ocean velocity and maximum-MLD globe dataset"
    )
    parser.add_argument("--input", type=Path, default=DATA_DIR / "Maximenko_dyntopo.nc")
    parser.add_argument("--mld-input", type=Path, default=DATA_DIR / "mld_dr003_ref10m_v2023.nc")
    parser.add_argument(
        "--output",
        type=Path,
        default=FRONTEND_PUBLIC_DATA_DIR / "maximenko_slope.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_maximenko(str(args.input))
    mld_source = load_mld(str(args.mld_input))
    fields = compute_ocean_fields(source, mld_source)
    export_globe_json(fields, args.output)


if __name__ == "__main__":
    main()
