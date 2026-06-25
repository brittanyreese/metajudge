"""Provenance gate: R MASS::polr must recover the DGP's planted b1/b2.

Runs scripts/oracles/gen_dgp_recovery.py, which regenerates a DGP sample from its seed and
refits it with MASS::polr. R is not a project runtime or CI dependency, so this test skips
cleanly when R or MASS is absent (exit code 2). Mirrors tests/test_oracle_provenance.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracles" / "gen_dgp_recovery.py"


def test_dgp_recovers_planted_coefficients_under_polr() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 2:
        pytest.skip(f"R / MASS unavailable for the polr recovery oracle:\n{completed.stdout}")
    assert completed.returncode == 0, (
        "DGP recovery provenance check failed; a planted coefficient was not recovered by "
        f"R MASS::polr.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
