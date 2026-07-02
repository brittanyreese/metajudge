# sim/harness.py
"""Replication grid harness over the DGP and the shipped logistic_dif engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from metajudge import ClusterBootstrapDif, cluster_bootstrap_dif, logistic_dif
from sim.dgp import FOCAL, REFERENCE, DgpParams, simulate

_CELL_SEED_STRIDE = 100_000  # keeps each cell's per-rep seed block disjoint


@dataclass(frozen=True)
class CellSummary:
    """Operating characteristics of one cell, computed over converged replications."""

    n_reps: int
    n_converged: int
    reject_total_rate: float
    reject_uniform_rate: float
    reject_nonuniform_rate: float
    mc_se_total: float
    mean_r2_delta: float
    po_flag_rate: float
    alpha: float


def summarize_cell(results: pd.DataFrame, *, alpha: float = 0.05) -> CellSummary:
    """Rejection rates, power, Monte-Carlo SE, and mean effect size for one cell.

    Rates are over converged replications only; a cell with no converged draw returns
    NaN rates. ``mc_se_total`` is the binomial SE of the total-DIF rejection rate.
    """
    n_reps = len(results)
    conv = results[results["converged"]]
    n_conv = len(conv)
    if n_conv == 0:
        return CellSummary(
            n_reps=n_reps,
            n_converged=0,
            reject_total_rate=float("nan"),
            reject_uniform_rate=float("nan"),
            reject_nonuniform_rate=float("nan"),
            mc_se_total=float("nan"),
            mean_r2_delta=float("nan"),
            po_flag_rate=float("nan"),
            alpha=alpha,
        )
    reject_total = float((conv["p_total"] < alpha).mean())
    reject_uniform = float((conv["p_uniform"] < alpha).mean())
    reject_nonuniform = float((conv["p_nonuniform"] < alpha).mean())
    mc_se = float(np.sqrt(reject_total * (1.0 - reject_total) / n_conv))
    return CellSummary(
        n_reps=n_reps,
        n_converged=n_conv,
        reject_total_rate=reject_total,
        reject_uniform_rate=reject_uniform,
        reject_nonuniform_rate=reject_nonuniform,
        mc_se_total=mc_se,
        mean_r2_delta=float(conv["nagelkerke_r2_delta"].mean()),
        po_flag_rate=float(conv["brant_po_flag"].mean()),
        alpha=alpha,
    )


_RESULT_COLS = [
    "rep",
    "seed",
    "p_total",
    "p_uniform",
    "p_nonuniform",
    "nagelkerke_r2_delta",
    "dif_class",
    "brant_po_flag",
    "converged",
    "conditioner_group_corr",
    "conditioner_common_support",
    "conditioner_overlap_weak",
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
            res = logistic_dif(sample.ratings, focal=FOCAL, reference=REFERENCE, conditioner=cond)
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
                    "brant_po_flag": False,
                    "converged": False,
                    "conditioner_group_corr": np.nan,
                    "conditioner_common_support": np.nan,
                    "conditioner_overlap_weak": False,
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
                    "brant_po_flag": res.po_violation,
                    "converged": res.converged,
                    "conditioner_group_corr": res.conditioner_group_corr,
                    "conditioner_common_support": res.conditioner_common_support,
                    "conditioner_overlap_weak": res.conditioner_overlap_weak,
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


_BOOTSTRAP_COLS = [
    "rep",
    "seed",
    "r2_ci_low",
    "r2_ci_high",
    "chi2_ci_low",
    "chi2_ci_high",
    "ci_reliable",
    "n_effective",
    "base_converged",
]


def run_cell_bootstrap(
    params: DgpParams,
    *,
    n_reps: int,
    n_boot: int,
    base_seed: int,
    ci_level: float = 0.95,
    conditioner: str = "external",
) -> pd.DataFrame:
    """Replicate one cell through ``cluster_bootstrap_dif``.

    Each replication draws a fresh :class:`~sim.dgp.SimSample` and calls
    :func:`~metajudge.cluster_bootstrap_dif`. Non-identifiable draws (``ValueError``) are
    recorded with ``NaN`` CI bounds and ``ci_reliable=False`` rather than aborting.

    Parameters
    ----------
    params:
        DGP cell specification.
    n_reps:
        Number of outer Monte Carlo replications.
    n_boot:
        Bootstrap draws per replication, passed to ``cluster_bootstrap_dif``.
    base_seed:
        Seed for the first replication; subsequent reps use ``base_seed + rep``.
    ci_level:
        Confidence level for the percentile CI (default 0.95).
    conditioner:
        ``"external"`` passes the simulated external score; ``"rest_score"`` passes
        ``None`` so ``cluster_bootstrap_dif`` uses its own rest-score fallback.

    Returns
    -------
    pd.DataFrame
        One row per replication with columns in ``_BOOTSTRAP_COLS``.
    """
    if conditioner not in ("external", "rest_score"):
        raise ValueError("conditioner must be 'external' or 'rest_score'")
    rows: list[dict[str, object]] = []
    for rep in range(n_reps):
        seed = base_seed + rep
        sample = simulate(params, seed=seed)
        cond = sample.conditioner if conditioner == "external" else None
        try:
            result: ClusterBootstrapDif = cluster_bootstrap_dif(
                sample.ratings,
                focal=FOCAL,
                reference=REFERENCE,
                conditioner=cond,
                n_boot=n_boot,
                seed=seed + 1_000_000,
                ci=ci_level,
            )
        except ValueError:
            rows.append(
                {
                    "rep": rep,
                    "seed": seed,
                    "r2_ci_low": float("nan"),
                    "r2_ci_high": float("nan"),
                    "chi2_ci_low": float("nan"),
                    "chi2_ci_high": float("nan"),
                    "ci_reliable": False,
                    "n_effective": 0,
                    "base_converged": False,
                }
            )
        else:
            rows.append(
                {
                    "rep": rep,
                    "seed": seed,
                    "r2_ci_low": result.r2_delta_ci_low,
                    "r2_ci_high": result.r2_delta_ci_high,
                    "chi2_ci_low": result.chi2_total_ci_low,
                    "chi2_ci_high": result.chi2_total_ci_high,
                    "ci_reliable": result.ci_reliable,
                    "n_effective": result.n_effective,
                    "base_converged": result.base.converged,
                }
            )
    return pd.DataFrame(rows, columns=_BOOTSTRAP_COLS)
