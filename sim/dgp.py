# sim/dgp.py
"""Data-generating process for ordinal rater-facet DIF simulation.

Cumulative-logit (McCullagh 1980; Zumbo framework) response model:

    logit P(X > level_k) = thresholds[k] + slope_k * theta + b2 * G + b3 * (theta * G) + u_rater

with a separable impact mean (focal theta ~ N(mu_focal, 1)), a random rater intercept
u_rater, and a per-cutpoint trait slope that breaks proportional odds when po_violation
!= 0. See research/2026-06-24-e07-dif-sim-validation-methods-grounding.md sections 1, 3.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from metajudge.data import Ratings

REFERENCE = "reference"
FOCAL = "focal"


@dataclass(frozen=True)
class DgpParams:
    """One cell of the simulation design. n_items_per_group counts judged outputs."""

    n_items_per_group: int
    n_raters: int
    n_categories: int = 5
    thresholds: tuple[float, ...] = (2.0, 0.8, -0.8, -2.0)
    trait_slope: float = 1.0
    rater_sd: float = 0.5
    mu_focal: float = 0.0
    dif_uniform: float = 0.0
    dif_nonuniform: float = 0.0
    po_violation: float = 0.0
    conditioner_reliability: float = 1.0


@dataclass(frozen=True)
class SimSample:
    """A simulated draw plus the ground truth Plan 3 needs to score the engine."""

    ratings: Ratings
    theta: dict[Hashable, float]
    conditioner: dict[Hashable, float]
    params: DgpParams


def _validate(p: DgpParams) -> None:
    if len(p.thresholds) != p.n_categories - 1:
        raise ValueError(
            f"thresholds must have n_categories-1={p.n_categories - 1} entries, "
            f"got {len(p.thresholds)}"
        )
    if any(a <= b for a, b in zip(p.thresholds, p.thresholds[1:], strict=False)):
        raise ValueError("thresholds must be strictly decreasing (P(X>k) ordering)")
    if not 0.0 < p.conditioner_reliability <= 1.0:
        raise ValueError("conditioner_reliability must be in (0, 1]")
    if p.n_raters < 1:
        raise ValueError("n_raters must be >= 1")
    if p.n_items_per_group < 1:
        raise ValueError("n_items_per_group must be >= 1")


def _category_probs(eta_by_cut: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map per-cutpoint linear predictors (..., K-1) to category probabilities (..., K)."""
    p_gt: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-eta_by_cut))  # P(X > level_k)
    # Enforce P(X>k) non-increasing in k so category probs stay non-negative even when a
    # per-cutpoint slope (PO violation) would otherwise cross the cumulative curves.
    p_gt = np.minimum.accumulate(p_gt, axis=-1)  # type: ignore[reportUnknownMemberType]
    lead = np.ones((*p_gt.shape[:-1], 1))
    tail = np.zeros((*p_gt.shape[:-1], 1))
    upper: NDArray[np.float64] = np.concatenate([lead, p_gt], axis=-1)  # type: ignore[reportUnknownMemberType]
    lower: NDArray[np.float64] = np.concatenate([p_gt, tail], axis=-1)  # type: ignore[reportUnknownMemberType]
    return upper - lower


def simulate(params: DgpParams, *, seed: int) -> SimSample:
    """Generate one ordinal rater-facet sample at ``params`` and ``seed``."""
    _validate(params)
    rng = np.random.default_rng(seed)
    k_cuts = params.n_categories - 1
    thr: NDArray[np.float64] = np.asarray(params.thresholds, dtype=float)
    centered: NDArray[np.float64] = np.linspace(1.0, -1.0, k_cuts)
    slope_by_cut: NDArray[np.float64] = params.trait_slope + params.po_violation * centered

    n = params.n_items_per_group
    theta_ref: NDArray[np.float64] = rng.normal(0.0, 1.0, n)
    theta_foc: NDArray[np.float64] = rng.normal(params.mu_focal, 1.0, n)
    u: NDArray[np.float64] = rng.normal(0.0, params.rater_sd, params.n_raters)

    item_theta: list[tuple[str, float, int]] = []
    for i in range(n):
        item_theta.append((f"ref_{i}", float(theta_ref[i]), 0))
        item_theta.append((f"foc_{i}", float(theta_foc[i]), 1))

    theta_map: dict[Hashable, float] = {}
    rows: list[dict[str, object]] = []
    for item_id, th, g in item_theta:
        theta_map[item_id] = th
        # eta per cutpoint per rater: shape (K-1, R).
        eta = (
            thr[:, None]
            + slope_by_cut[:, None] * th
            + params.dif_uniform * g
            + params.dif_nonuniform * th * g
            + u[None, :]
        )
        probs = _category_probs(eta.T)  # (R, K)
        cum: NDArray[np.float64] = np.cumsum(probs, axis=1)  # type: ignore[reportUnknownMemberType]
        draws: NDArray[np.float64] = rng.random(params.n_raters)
        cat = (draws[:, None] >= cum).sum(axis=1)  # 0..K-1
        cat = np.minimum(cat, params.n_categories - 1)  # guard float cum[-1] < draw
        stratum = FOCAL if g == 1 else REFERENCE
        for r in range(params.n_raters):
            rows.append(
                {"item": item_id, "rater": f"r{r}", "score": int(cat[r]) + 1, "stratum": stratum}
            )

    long = pd.DataFrame(rows)
    ratings = Ratings.from_long(
        long, item="item", rater="rater", score="score", stratum="stratum"
    )

    items = list(theta_map.keys())
    th_vec: NDArray[np.float64] = np.asarray([theta_map[it] for it in items], dtype=float)
    rel = params.conditioner_reliability
    var_theta = float(np.var(th_vec))
    sigma_e = float(np.sqrt(var_theta * (1.0 - rel) / rel)) if rel < 1.0 else 0.0
    noise: NDArray[np.float64] = (
        rng.normal(0.0, sigma_e, len(items)) if sigma_e > 0.0 else np.zeros(len(items))
    )
    conditioner: dict[Hashable, float] = {
        it: float(th_vec[j] + noise[j]) for j, it in enumerate(items)
    }
    return SimSample(ratings=ratings, theta=theta_map, conditioner=conditioner, params=params)
