"""Provenance gate: the pinned DIF oracle constants must be regenerable.

Runs ``scripts/oracles/gen_olr_oracle.py``, which reproduces ``_QUALITY`` from
its seed and refits the external-conditioner and rest-score DIF models with R
``MASS::polr``, comparing the result to the constants pinned in
``tests/test_dif.py``. R is not a project runtime or CI dependency (the engine
is scipy-only), so this test skips cleanly when R or the ``MASS`` package is
absent (the harness signals that with exit code 2).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracles" / "gen_olr_oracle.py"


def test_pinned_dif_constants_reproduced_by_polr() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 2:
        pytest.skip(f"R / MASS unavailable for the polr oracle:\n{completed.stdout}")
    assert completed.returncode == 0, (
        "DIF oracle provenance check failed; a pinned constant disagrees with "
        f"R MASS::polr.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
