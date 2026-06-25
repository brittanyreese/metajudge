"""Tests for sim.harness: run_cell and run_grid."""

import math

import numpy as np
import pandas as pd
from sim.dgp import DgpParams
from sim.harness import CellSummary, run_cell, run_grid, summarize_cell

_RESULT_COLS = [
    "rep",
    "seed",
    "p_total",
    "p_uniform",
    "p_nonuniform",
    "nagelkerke_r2_delta",
    "dif_class",
    "po_violation",
    "converged",
]


def test_run_cell_shape_and_columns() -> None:
    params = DgpParams(n_items_per_group=60, n_raters=3)
    df = run_cell(params, n_reps=4, base_seed=0)
    assert list(df.columns) == _RESULT_COLS
    assert len(df) == 4
    assert df["rep"].tolist() == [0, 1, 2, 3]
    assert df["converged"].dtype == bool


def test_run_cell_is_reproducible() -> None:
    params = DgpParams(n_items_per_group=60, n_raters=3)
    a = run_cell(params, n_reps=3, base_seed=42)
    b = run_cell(params, n_reps=3, base_seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_run_grid_concatenates_cells_with_params() -> None:
    cells = [
        DgpParams(n_items_per_group=60, n_raters=3, dif_uniform=0.0),
        DgpParams(n_items_per_group=60, n_raters=3, dif_uniform=0.8),
    ]
    grid = run_grid(cells, n_reps=2, base_seed=0)
    assert len(grid) == 4  # 2 cells x 2 reps
    assert set(grid["cell"].tolist()) == {0, 1}
    assert "dif_uniform" in grid.columns
    assert sorted(grid["dif_uniform"].unique().tolist()) == [0.0, 0.8]
    # Cells are seeded independently: their per-rep seeds do not collide.
    assert grid[grid["cell"] == 0]["seed"].tolist() != grid[grid["cell"] == 1]["seed"].tolist()


# ---------------------------------------------------------------------------
# CellSummary / summarize_cell tests
# ---------------------------------------------------------------------------


def _canned(
    p_total: list[float],
    converged: list[bool],
    r2: list[float] | None = None,
    p_nonuniform: list[float] | None = None,
    po: list[bool] | None = None,
) -> pd.DataFrame:
    n = len(p_total)
    return pd.DataFrame(
        {
            "rep": list(range(n)),
            "seed": list(range(n)),
            "p_total": p_total,
            "p_uniform": p_total,
            "p_nonuniform": p_nonuniform if p_nonuniform is not None else p_total,
            "nagelkerke_r2_delta": r2 if r2 is not None else [0.0] * n,
            "dif_class": ["A"] * n,
            "po_violation": po if po is not None else [False] * n,
            "converged": converged,
        }
    )


def test_summarize_rejection_rate_and_mc_se() -> None:
    # 10 converged reps, 2 with p_total < 0.05 -> rate 0.2, SE = sqrt(0.2*0.8/10).
    p_total = [0.01, 0.02, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.55, 0.65]
    summary = summarize_cell(_canned(p_total, [True] * 10), alpha=0.05)
    assert isinstance(summary, CellSummary)
    assert summary.n_converged == 10
    assert summary.reject_total_rate == 0.2
    assert math.isclose(summary.mc_se_total, math.sqrt(0.2 * 0.8 / 10))


def test_summarize_ignores_nonconverged() -> None:
    # 12 rows, 2 non-converged: rates are over the 10 converged reps.
    p_total = [0.01, 0.02, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.55, 0.65, 0.01, 0.01]
    converged = [True] * 10 + [False, False]
    summary = summarize_cell(_canned(p_total, converged), alpha=0.05)
    assert summary.n_reps == 12
    assert summary.n_converged == 10
    assert summary.reject_total_rate == 0.2  # the two non-converged 0.01s are excluded


def test_summarize_power_effect_and_po_rate() -> None:
    p_nonuniform = [0.001, 0.002, 0.003, 0.2, 0.3]  # 3 of 5 reject at 0.05
    r2 = [0.10, 0.12, 0.08, 0.02, 0.03]
    po = [True, True, False, False, False]
    summary = summarize_cell(
        _canned([0.5] * 5, [True] * 5, r2=r2, p_nonuniform=p_nonuniform, po=po), alpha=0.05
    )
    assert summary.reject_nonuniform_rate == 0.6
    assert math.isclose(summary.mean_r2_delta, float(np.mean(r2)))
    assert summary.po_flag_rate == 0.4
