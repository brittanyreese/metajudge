"""Generate the shared Brant-test fixtures (one where PO holds, one where it fails).

Single source of truth for both oracles: writes the CSVs under sim/fixtures/, read
verbatim by sim/oracles/brant_reference.R (R brant) and by tests/test_diagnostics.py.
Generating once and sharing the CSV gives the python test and the R oracle exactly the
same rows, so the R brant output is the precise reference value for each fixture.

Run: uv run python sim/oracles/make_brant_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HOLDS = _FIXTURES / "brant_po_holds.csv"
VIOLATED = _FIXTURES / "brant_po_violated.csv"


def _write_holds() -> None:
    rng = np.random.default_rng(20260624)
    n = 600
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # PO holds: a single linear predictor with proportional (parallel) thresholds.
    eta = 0.8 * x1 - 0.5 * x2
    cuts = np.array([-1.2, -0.2, 0.9])
    probs = 1.0 / (1.0 + np.exp(-(cuts[None, :] - eta[:, None])))  # P(Y > k)
    u = rng.random(n)
    y = 1 + (u[:, None] > probs).sum(axis=1)  # categories 1..4
    pd.DataFrame({"y": y.astype(int), "x1": x1, "x2": x2}).to_csv(HOLDS, index=False)
    print(f"wrote {HOLDS} ({n} rows)")


def _write_violated() -> None:
    rng = np.random.default_rng(20260625)
    n = 1200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # PO fails on x1: its slope drifts across the three cutpoints (0.4 -> 0.9 -> 1.4),
    # so the parallel-lines assumption is broken. x2 stays proportional (slope -0.5).
    cuts = np.array([1.0, -0.2, -1.4])
    b1 = np.array([0.4, 0.9, 1.4])
    eta = cuts[None, :] + b1[None, :] * x1[:, None] - 0.5 * x2[:, None]  # P(Y > k) logit
    probs = 1.0 / (1.0 + np.exp(-eta))
    u = rng.random(n)
    y = 1 + (u[:, None] > probs).sum(axis=1)  # categories 1..4
    pd.DataFrame({"y": y.astype(int), "x1": x1, "x2": x2}).to_csv(VIOLATED, index=False)
    print(f"wrote {VIOLATED} ({n} rows)")


def main() -> None:
    _FIXTURES.mkdir(parents=True, exist_ok=True)
    _write_holds()
    _write_violated()


if __name__ == "__main__":
    main()
