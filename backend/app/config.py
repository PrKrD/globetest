"""Paths for the Maximenko data transformation."""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_ROOT / "data"
FRONTEND_PUBLIC_DATA_DIR = BACKEND_ROOT.parent / "frontend" / "public" / "data"
