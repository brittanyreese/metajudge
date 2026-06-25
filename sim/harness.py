# sim/harness.py
"""Replication grid harness over the DGP and the shipped logistic_dif engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd

from metajudge import logistic_dif
from sim.dgp import FOCAL, REFERENCE, DgpParams, simulate

_CELL_SEED_STRIDE = 100_000  # keeps each cell's per-rep seed block disjoint

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


def run_cell(
    params: DgpParams,
    *,
    n_reps: int,
    base_seed: int,
    conditioner: str = "external",
) -> pd.DataFrame:
    """Replicate one cell ``n_reps`` times through ``logistic_dif``.

    Each replication draws a fresh :class:`~sim.dgp.SimSample` with seed
    ``base_seed + rep`` and passes it through :func:`~metajudge.logistic_dif`.

    When the engine raises :exc:`ValueError` (non-identifiable draw) the row is
    recorded with ``converged=False`` and ``NaN`` for every float statistic rather
    than aborting the cell.

    Parameters
    ----------
    params:
        DGP cell specification.
    n_reps:
        Number of Monte Carlo replications.
    base_seed:
        Seed for the first replication; subsequent reps use ``base_seed + rep``.
    conditioner:
        ``"external"`` passes the simulated external score; ``"rest_score"`` passes
        ``None`` so ``logistic_dif`` uses its own rest-score fallback.

    Returns
    -------
    pd.DataFrame
        One row per replication with columns in :data:`_RESULT_COLS`.
    """
    if conditioner not in ("external", "rest_score"):
        raise ValueError("conditioner must be 'external' or 'rest_score'")
    rows: list[dict[str, object]] = []
    for rep in range(n_reps):
        seed = base_seed + rep
        sample = simulate(params, seed=seed)
        cond = sample.conditioner if conditioner == "external" else None
        try:
            res = logistic_dif(
                sample.ratings, focal=FOCAL, reference=REFERENCE, conditioner=cond
            )
        except ValueError:
            rows.append(
                {
                    "rep": rep,
                    "seed": seed,
                    "p_total": np.nan,
                    "p_uniform": np.nan,
                    "p_nonuniform": np.nan,
                    "nagelkerke_r2_delta": np.nan,
                    "dif_class": "NA",
                    "po_violation": False,
                    "converged": False,
                }
            )
        else:
            rows.append(
                {
                    "rep": rep,
                    "seed": seed,
                    "p_total": res.p_total,
                    "p_uniform": res.p_uniform,
                    "p_nonuniform": res.p_nonuniform,
                    "nagelkerke_r2_delta": res.nagelkerke_r2_delta,
                    "dif_class": res.dif_class,
                    "po_violation": res.po_violation,
                    "converged": res.converged,
                }
            )
    return pd.DataFrame(rows, columns=_RESULT_COLS)


def run_grid(
    cells: Sequence[DgpParams],
    *,
    n_reps: int,
    base_seed: int,
    conditioner: str = "external",
) -> pd.DataFrame:
    """Run every cell and return a tidy frame: one row per (cell, rep) with cell params.

    Each cell receives a disjoint seed block: cell ``idx`` uses
    ``base_seed + idx * _CELL_SEED_STRIDE`` as its ``base_seed``.

    Parameters
    ----------
    cells:
        Sequence of :class:`~sim.dgp.DgpParams` cells to sweep.
    n_reps:
        Number of replications per cell.
    base_seed:
        Starting seed for cell 0.
    conditioner:
        Passed through to :func:`run_cell`.

    Returns
    -------
    pd.DataFrame
        All cells concatenated with a leading ``cell`` index column and one
        column per :class:`~sim.dgp.DgpParams` field appended after the result
        columns.
    """
    parts: list[pd.DataFrame] = []
    for idx, params in enumerate(cells):
        cell_df = run_cell(
            params,
            n_reps=n_reps,
            base_seed=base_seed + idx * _CELL_SEED_STRIDE,
            conditioner=conditioner,
        )
        cell_df.insert(0, "cell", idx)
        for field, value in asdict(params).items():
            cell_df[field] = [value] * len(cell_df)
        parts.append(cell_df)
    return pd.concat(parts, ignore_index=True)
