from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local temp dir to avoid host temp directory permission issues."""

    base = ROOT / "tmp_test"
    base.mkdir(parents=True, exist_ok=True)
    temp_dir = base / f"case_{uuid.uuid4().hex[:12]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
