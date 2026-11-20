"""Ensure importing the API does not initialize heavyweight ML runtimes."""
import os
import subprocess
import sys
from pathlib import Path


def test_api_startup_does_not_import_torch_or_transformers():
    backend = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(backend)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.main, sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        cwd=backend,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr or result.stdout
