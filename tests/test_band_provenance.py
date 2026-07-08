"""Provenance gate for the pinned Jodoin-Gierl ordinal-PO band constants (SPEC R19).

``src/metajudge/dif.py`` ships ``_JG_NEGLIGIBLE`` and ``_JG_LARGE`` as the derived
ordinal-PO A/B/C thresholds. Those literals must stay equal to the derivation summary
they were read from (``sim/results/band_derivation_summary.csv``), so a hand-edited
constant or a regenerated summary cannot silently diverge from the recorded provenance.

Mirrors ``tests/test_brant_provenance.py`` in intent, but reads the committed artifact
instead of re-running the study: the derivation is a ~90-minute Monte Carlo, so this gate
binds to its frozen output rather than reproducing it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from metajudge.dif import (
    _JG_LARGE,  # pyright: ignore[reportPrivateUsage]
    _JG_NEGLIGIBLE,  # pyright: ignore[reportPrivateUsage]
)

_SUMMARY = Path(__file__).resolve().parents[1] / "sim" / "results" / "band_derivation_summary.csv"


def _read_summary() -> dict[str, float]:
    with _SUMMARY.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    return {key: float(value) for key, value in row.items()}


def test_band_constants_match_derivation_summary() -> None:
    summary = _read_summary()
    # The shipped constants are the summary's derived ordinal bands rounded to four
    # decimals. Bind each to its column so a stale literal or a regenerated summary trips
    # here (abs=5e-5 is half the last retained digit).
    assert pytest.approx(summary["ordinal_band_ab"], abs=5e-5) == _JG_NEGLIGIBLE
    assert pytest.approx(summary["ordinal_band_bc"], abs=5e-5) == _JG_LARGE
