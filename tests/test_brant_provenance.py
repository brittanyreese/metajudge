"""Provenance gate for the pinned Brant reference constants (SPEC R19).

``tests/test_brant.py`` checks the engine reproduces the pinned R ``brant`` literals. This
gate closes the other half: it re-runs ``sim/oracles/brant_reference.R`` and confirms the
frozen literals are still what R ``brant`` produces, so a stale or mistyped constant cannot
silently diverge from the reference. Mirrors ``tests/test_oracle_provenance.py`` for the
``MASS::polr`` DIF pins.

R is not a project runtime or CI dependency; this skips cleanly when ``Rscript`` or the R
``brant`` / ``MASS`` packages are absent, and runs in the weekly rigor and release workflows
where R is provisioned.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Sibling test module on sys.path under pytest's default (prepend) import mode. The pinned
# literals live there as the single source of truth; this gate reads them, never re-pins.
import test_brant as tb

_SCRIPT = Path(__file__).resolve().parents[1] / "sim" / "oracles" / "brant_reference.R"


def _parse_val_lines(stdout: str) -> dict[tuple[str, str], tuple[float, int, float]]:
    out: dict[tuple[str, str], tuple[float, int, float]] = {}
    for line in stdout.splitlines():
        if not line.startswith("VAL\t"):
            continue
        _, key, row, chi2, df, prob = line.split("\t")
        out[(key, row)] = (float(chi2), int(df), float(prob))
    return out


def _run_oracle() -> dict[tuple[str, str], tuple[float, int, float]]:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript unavailable for the R brant provenance oracle")
    completed = subprocess.run(
        ["Rscript", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0 and "there is no package called" in completed.stderr:
        pytest.skip(f"R brant / MASS package unavailable:\n{completed.stderr}")
    assert completed.returncode == 0, (
        f"brant_reference.R failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    vals = _parse_val_lines(completed.stdout)
    assert vals, f"no VAL lines parsed from oracle output:\n{completed.stdout}"
    return vals


def test_pinned_brant_constants_reproduced_by_r() -> None:
    vals = _run_oracle()
    # Both sides are R brant on the same frozen fixture, so the pinned literal must match the
    # fresh oracle value to R's print precision -- a tight gate that catches a stale or
    # mistyped constant. The engine-vs-literal agreement is checked in tests/test_brant.py.
    h_omni, h_df, h_p = vals[("brant_po_holds", "Omnibus")]
    assert h_omni == pytest.approx(tb.R_OMNIBUS_CHI2, abs=1e-5)
    assert h_df == tb.R_OMNIBUS_DF
    assert h_p == pytest.approx(tb.R_OMNIBUS_P, abs=1e-5)
    assert vals[("brant_po_holds", "x1")][0] == pytest.approx(tb.R_X1_CHI2, abs=1e-5)
    assert vals[("brant_po_holds", "x2")][0] == pytest.approx(tb.R_X2_CHI2, abs=1e-5)

    assert vals[("brant_po_violated", "Omnibus")][0] == pytest.approx(tb.RV_OMNIBUS_CHI2, abs=1e-5)
    assert vals[("brant_po_violated", "x1")][0] == pytest.approx(tb.RV_X1_CHI2, abs=1e-5)
    assert vals[("brant_po_violated", "x2")][0] == pytest.approx(tb.RV_X2_CHI2, abs=1e-5)
