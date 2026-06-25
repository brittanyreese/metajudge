"""Tests for sim.harness: run_cell and run_grid."""

import pandas as pd
from sim.dgp import DgpParams
from sim.harness import run_cell, run_grid

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
