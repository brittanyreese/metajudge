"""Proportional-odds (parallel-lines) assumption diagnostics.

Implements the Brant (1990) Wald-type test. Fit the J-1 separate binary logits
P(Y > k) underlying the proportional-odds model and test whether their slope
vectors are equal across cutpoints. A significant omnibus result means the
proportional-odds assumption is violated for at least one predictor; the
per-predictor tests locate it.

The covariance of the stacked cutpoint slope estimators uses the Brant
cross-cutpoint form: the J-1 binary models are fit on the same observations, so
their estimators are correlated. For cutpoints k < l (so pi_k >= pi_l),

    Cov(beta_k, beta_l) = (X' W_k X)^-1 (X' W_kl X) (X' W_l X)^-1,

with W_k = diag(pi_k (1 - pi_k)) and W_kl = diag(pi_l (1 - pi_k)). Ignoring the
cross term (treating the fits as independent) inflates the variance and shrinks
the statistic, so it is kept.

Reference: Brant, R. (1990). Assessing proportionality in the proportional odds
model for ordinal logistic regression. Biometrics, 46(4), 1171-1178.
doi:10.2307/2532457.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import chi2 as _chi2dist  # type: ignore[import-untyped]


@dataclass(frozen=True)
class BrantResult:
    """Outcome of a Brant proportional-odds test.

    ``per_predictor`` maps each predictor name to ``(chi2, df, p)``.
    """

    omnibus_chi2: float
    omnibus_df: int
    omnibus_p: float
    per_predictor: dict[str, tuple[float, int, float]]
    converged: bool


def _fit_binary_logit(
    y: NDArray[np.float64], design: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Fit a binary logit on a design that already includes the intercept column.

    Returns ``(beta, pi, converged)`` where ``beta`` has length ``design.shape[1]``
    (intercept first) and ``pi`` is the fitted P(y = 1) vector.
    """
    q = design.shape[1]

    def nll(b: NDArray[np.float64]) -> float:
        eta = design @ b
        return -float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    res = minimize(nll, np.zeros(q), method="BFGS")  # type: ignore[reportUnknownVariableType]
    beta: NDArray[np.float64] = np.asarray(res.x, dtype=np.float64)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    pi: NDArray[np.float64] = np.asarray(1.0 / (1.0 + np.exp(-(design @ beta))), dtype=np.float64)
    return beta, pi, bool(res.success)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def brant_test(
    endog: NDArray[np.int_],
    exog: NDArray[np.float64],
    *,
    names: list[str] | None = None,
) -> BrantResult:
    """Brant test of the proportional-odds assumption.

    Args:
        endog: ordinal categories, shape ``(n,)``, with J distinct ordered levels
            (any integer coding; the J-1 cutpoints are taken at the sorted uniques).
        exog: design matrix without intercept, shape ``(n, p)``.
        names: optional predictor names (length p); defaults to ``x0..x{p-1}``.

    Returns:
        BrantResult with the omnibus and per-predictor Wald chi-squares. Omnibus df
        is ``(J - 2) * p``; each per-predictor df is ``J - 2``.
    """
    y = np.asarray(endog, dtype=int)
    x = np.asarray(exog, dtype=float)
    n, p = x.shape
    levels: NDArray[np.int_] = np.unique(y)  # type: ignore[reportUnknownMemberType]
    m = int(levels.size) - 1  # number of cutpoints J-1
    if names is None:
        names = [f"x{i}" for i in range(p)]

    design: NDArray[np.float64] = np.column_stack([np.ones(n), x])  # type: ignore[reportUnknownMemberType]
    q = p + 1

    betas: list[NDArray[np.float64]] = []
    pis: list[NDArray[np.float64]] = []
    converged = True
    for k in range(m):
        zk: NDArray[np.float64] = (y > levels[k]).astype(float)
        beta, pi, ok = _fit_binary_logit(zk, design)
        betas.append(beta)
        pis.append(pi)
        converged = converged and ok

    # Per-cutpoint inverse information.
    inv_info: list[NDArray[np.float64]] = []
    for k in range(m):
        w = pis[k] * (1.0 - pis[k])
        info_k = design.T @ (design * w[:, None])
        inv_info.append(np.asarray(np.linalg.pinv(info_k), dtype=np.float64))

    # Full covariance of the stacked slope-and-intercept estimator (m*q square).
    big = m * q
    vmat: NDArray[np.float64] = np.zeros((big, big))
    for k in range(m):
        vmat[k * q : (k + 1) * q, k * q : (k + 1) * q] = inv_info[k]
    for k in range(m):
        for j in range(k + 1, m):
            # k < j  =>  pi_k >= pi_j ; W_kj = diag(pi_j (1 - pi_k)).
            wkj = pis[j] * (1.0 - pis[k])
            cross = inv_info[k] @ (design.T @ (design * wkj[:, None])) @ inv_info[j]
            vmat[k * q : (k + 1) * q, j * q : (j + 1) * q] = cross
            vmat[j * q : (j + 1) * q, k * q : (k + 1) * q] = cross.T

    beta_stack: NDArray[np.float64] = np.concatenate(betas)  # type: ignore[reportUnknownMemberType]

    def _wald(predictor_cols: list[int]) -> tuple[float, int]:
        # Contrast beta_r - beta_0 over the given slope columns, for r = 1..m-1.
        rows = (m - 1) * len(predictor_cols)
        dmat: NDArray[np.float64] = np.zeros((rows, big))
        row = 0
        for r in range(1, m):
            for col in predictor_cols:
                dmat[row, 0 * q + col] = -1.0
                dmat[row, r * q + col] = 1.0
                row += 1
        d = dmat @ beta_stack
        cov = dmat @ vmat @ dmat.T
        stat = float(d @ np.linalg.pinv(cov) @ d)
        return stat, rows

    # Slope columns are 1..p (column 0 is the intercept).
    all_slopes = list(range(1, q))
    omnibus_chi2, omnibus_df = _wald(all_slopes)
    omnibus_p = float(_chi2dist.sf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0  # type: ignore[reportUnknownMemberType]

    per_predictor: dict[str, tuple[float, int, float]] = {}
    for idx in range(p):
        stat, df = _wald([1 + idx])
        pval = float(_chi2dist.sf(stat, df)) if df > 0 else 1.0  # type: ignore[reportUnknownMemberType]
        per_predictor[names[idx]] = (stat, df, pval)

    return BrantResult(
        omnibus_chi2=omnibus_chi2,
        omnibus_df=omnibus_df,
        omnibus_p=omnibus_p,
        per_predictor=per_predictor,
        converged=converged,
    )
