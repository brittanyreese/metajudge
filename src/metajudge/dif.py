"""DIF pillar: ordinal (proportional-odds) logistic-regression DIF.

Implements the Zumbo (1999) logistic-regression DIF framework for ordinal scores,
as used by the lordif package (Choi, Gibbons & Crane, 2011). Three nested
proportional-odds (cumulative logit) models per comparison yield likelihood-ratio
chi-square tests for total (2 df), uniform (1 df), and nonuniform (1 df) DIF; effect
size is the Nagelkerke pseudo-R-squared change classified on the Jodoin & Gierl (2001)
thresholds. See docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md.

A DIF analysis needs a quality conditioner independent of the studied response. The
default conditioner is a leave-one-rater-out rest score; callers may pass an explicit
external conditioner instead. With a single rater and no external conditioner there is
no independent conditioner and the analysis refuses to run.

statsmodels is *not* imported here; it is only an oracle in the test suite. The runtime
fit is scipy alone.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from scipy.stats import chi2  # type: ignore[import-untyped]

from metajudge.data import Ratings

# Jodoin & Gierl (2001) Nagelkerke R-squared change thresholds.
_JG_NEGLIGIBLE = 0.035
_JG_LARGE = 0.070


@dataclass(frozen=True)
class DifResult:
    """Ordinal logistic-regression DIF outcome for one focal-vs-reference comparison."""

    chi2_total: float
    chi2_uniform: float
    chi2_nonuniform: float
    p_total: float
    p_uniform: float
    p_nonuniform: float
    nagelkerke_r2_delta: float
    dif_class: str
    conditioner_source: str
    n_obs: int
    reference_level: str
    focal_level: str
    converged: bool


def _classify_jodoin_gierl(r2_delta: float) -> str:
    """Map a Nagelkerke R-squared change to an A/B/C DIF magnitude class.

    Jodoin & Gierl (2001): negligible (A) below 0.035, moderate (B) in
    ``[0.035, 0.070)``, large (C) at or above 0.070. These are an R-squared magnitude
    rule, not the ETS Mantel-Haenszel delta classification.
    """
    if r2_delta < _JG_NEGLIGIBLE:
        return "A"
    if r2_delta < _JG_LARGE:
        return "B"
    return "C"


def _fit_proportional_odds(endog: NDArray[np.int_], x: NDArray[np.float64]) -> tuple[float, bool]:
    """Fit a cumulative-logit (proportional-odds) model by ML.

    Returns ``(log_likelihood, converged)``. ``endog`` holds contiguous category indices
    ``0..K-1`` (``K >= 2``). ``x`` is the ``(n, p)`` design matrix (no intercept; absorbed
    into the thresholds). The model is ``P(Y <= k) = sigmoid(t_k - x.beta)`` with strictly
    increasing thresholds ``t_k``, parameterized as ``t_0`` plus softplus increments to
    keep them ordered. This reproduces R ``MASS::polr`` (method "logistic") and, in the
    two-category limit, the ``statsmodels`` ``Logit`` MLE.
    """
    n, p = x.shape
    k = int(endog.max()) + 1
    n_cut = k - 1
    counts: NDArray[np.float64] = np.bincount(endog, minlength=k).astype(float)  # type: ignore[reportUnknownMemberType]

    # Initialize thresholds from the marginal cumulative logits, betas at zero.
    cumulative: NDArray[np.float64] = np.cumsum(counts)[:-1] / n  # type: ignore[reportUnknownMemberType]
    cum: NDArray[np.float64] = np.clip(cumulative, 1e-6, 1 - 1e-6)  # type: ignore[reportUnknownMemberType]
    t_init: NDArray[np.float64] = np.log(cum / (1.0 - cum))
    theta_init: NDArray[np.float64] = np.empty(n_cut)
    theta_init[0] = t_init[0]
    diffs: NDArray[np.float64] = np.clip(np.diff(t_init), 1e-6, None)  # type: ignore[reportUnknownMemberType]
    theta_init[1:] = np.log(np.expm1(diffs))  # inverse softplus
    x0: NDArray[np.float64] = np.concatenate([np.zeros(p), theta_init])  # type: ignore[reportUnknownMemberType]
    rows: NDArray[np.intp] = np.arange(n)  # type: ignore[reportUnknownMemberType]

    def neg_log_likelihood(params: NDArray[np.float64]) -> float:
        beta: NDArray[np.float64] = params[:p]
        theta: NDArray[np.float64] = params[p:]
        thresh: NDArray[np.float64] = np.empty(n_cut)
        thresh[0] = theta[0]
        if n_cut > 1:
            thresh[1:] = thresh[0] + np.cumsum(np.logaddexp(0.0, theta[1:]))  # type: ignore[reportUnknownMemberType]
        xb: NDArray[np.float64] = x @ beta
        # cumulative P(Y <= k) for k = 0..K-2, shape (n, K-1)
        cdf: NDArray[np.float64] = expit(thresh[np.newaxis, :] - xb[:, np.newaxis])
        full: NDArray[np.float64] = np.empty((n, k))
        full[:, 0] = cdf[:, 0]
        if k > 2:
            full[:, 1:-1] = np.diff(cdf, axis=1)  # type: ignore[reportUnknownMemberType]
        full[:, -1] = 1.0 - cdf[:, -1]
        probs: NDArray[np.float64] = full[rows, endog]
        return float(-np.sum(np.log(np.clip(probs, 1e-12, None))))  # type: ignore[reportUnknownMemberType]

    res = minimize(  # type: ignore[reportUnknownVariableType]
        neg_log_likelihood, x0, method="BFGS", options={"maxiter": 2000}
    )
    converged = bool(res.success)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    if not converged:  # pragma: no cover - retry path for hard fits
        res = minimize(  # type: ignore[reportUnknownVariableType]
            neg_log_likelihood, x0, method="Nelder-Mead", options={"maxiter": 20000}
        )
        converged = bool(res.success)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return float(-res.fun), converged  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _nagelkerke(ll_model: float, ll_null: float, n: int) -> float:
    """Nagelkerke pseudo-R-squared for a model log-likelihood vs the null."""
    cox_snell = 1.0 - np.exp(2.0 * (ll_null - ll_model) / n)
    denom = 1.0 - np.exp(2.0 * ll_null / n)
    return float(cox_snell / denom)


def logistic_dif(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    conditioner: Mapping[Hashable, float] | None = None,
) -> DifResult:
    """Ordinal logistic-regression DIF of ``focal`` vs ``reference``.

    Fits three nested proportional-odds models (conditioner; + group; + group x
    conditioner) and returns the likelihood-ratio chi-square tests for total, uniform,
    and nonuniform DIF, the Nagelkerke R-squared change, and its A/B/C class.

    The conditioner is the quality axis the responses are matched on, and it must be
    independent of the studied response. If ``conditioner`` is given it maps each item to
    an external quality score (the stronger path). Otherwise a leave-one-rater-out rest
    score is used, which needs at least two raters per item. The conditioner is
    standardized internally; this affine rescaling does not change the likelihood-ratio
    tests or the R-squared change.

    The rest-score default detects differential functioning relative to the rater panel:
    it removes the studied rater's own response but not a bias shared across the panel.
    When the bias is a property of the instrument applied to a group (every rater shares
    it), the rest score is contaminated by that shared bias and will understate the DIF.
    Use an external, independent conditioner (a gold quality score, or a leave-one-
    criterion-out mean across rubric dimensions) for instrument-level bias.

    Raises:
        ValueError: if ``focal`` or ``reference`` is not a stratum level; if an explicit
            conditioner omits an included item; if no independent conditioner can be
            formed (single rating per item and no explicit conditioner); if all responses
            fall in one category; or if the conditioner is constant or near-perfectly
            collinear with the group, so DIF is not identifiable.
    """
    strata = ratings.strata()
    for level in (focal, reference):
        if level not in strata:
            raise ValueError(f"stratum level not found: {level}")

    focal_items = set(strata[focal])
    reference_items = set(strata[reference])
    included = focal_items | reference_items

    mat: NDArray[np.float64] = ratings.wide().to_numpy(dtype=float)  # items x raters, NaN unrated
    items_list: list[Hashable] = list(ratings.items)

    scores: list[float] = []
    groups: list[float] = []
    cond_rows: list[float] = []
    source = "external" if conditioner is not None else "rest_score"

    for row_idx, item in enumerate(items_list):
        if item not in included:
            continue
        row_scores: NDArray[np.float64] = mat[row_idx]
        rated: NDArray[np.float64] = row_scores[~np.isnan(row_scores)]
        is_focal_item = item in focal_items
        if conditioner is not None:
            if item not in conditioner:
                raise ValueError(f"conditioner missing item: {item!r}")
            item_cond = float(conditioner[item])
            for value in rated.tolist():
                cond_rows.append(item_cond)
                scores.append(float(value))
                groups.append(1.0 if is_focal_item else 0.0)
        else:
            count = int(rated.size)
            if count < 2:
                raise ValueError(
                    "cannot form an independent conditioner: item "
                    f"{item!r} has a single rating and no explicit conditioner was given. "
                    "Provide >=2 raters per item or pass conditioner=."
                )
            total = float(rated.sum())
            for value in rated.tolist():
                # leave-one-rater-out rest score: mean of the other raters
                cond_rows.append((total - float(value)) / (count - 1))
                scores.append(float(value))
                groups.append(1.0 if is_focal_item else 0.0)

    n = len(scores)
    g: NDArray[np.float64] = np.asarray(groups, dtype=float)
    cond: NDArray[np.float64] = np.asarray(cond_rows, dtype=float)

    # Map distinct response values to contiguous category indices 0..K-1.
    y: NDArray[np.float64] = np.asarray(scores, dtype=float)
    levels: NDArray[np.float64] = np.unique(y)  # type: ignore[reportUnknownMemberType]
    if len(levels) < 2:
        raise ValueError(
            "cannot fit DIF: all responses fall in a single category, so there is no "
            "ordinal variation to model."
        )

    # The conditioner must carry quality information independent of the group. With no
    # conditioner variance, or a conditioner (near-)perfectly collinear with the group,
    # quality and group membership cannot be separated and DIF is not identifiable. This
    # happens, for example, under perfect within-item rater agreement.
    std = float(cond.std(ddof=0))
    if std == 0.0:
        raise ValueError(
            "cannot identify DIF: the conditioner has no variance, so there is nothing to "
            "match on. Provide a conditioner that varies across items."
        )
    cond_z: NDArray[np.float64] = (cond - cond.mean()) / std
    correlation = float(np.corrcoef(cond_z, g)[0, 1])  # type: ignore[reportUnknownMemberType]
    if abs(correlation) > 0.999:
        raise ValueError(
            "cannot identify DIF: the conditioner is near-perfectly collinear with the "
            f"group (|r|={abs(correlation):.4f}), so group membership and quality cannot "
            "be separated. With the rest-score conditioner this signals perfect "
            "within-item rater agreement; supply an independent external conditioner."
        )

    endog: NDArray[np.int_] = np.searchsorted(levels, y).astype(np.int_)  # type: ignore[reportUnknownMemberType]
    cat_counts: NDArray[np.float64] = np.bincount(endog, minlength=len(levels)).astype(float)  # type: ignore[reportUnknownMemberType]
    ll_null = float(np.sum(cat_counts * np.log(cat_counts / n)))

    x1: NDArray[np.float64] = cond_z.reshape(-1, 1)
    x2: NDArray[np.float64] = np.column_stack([cond_z, g])  # type: ignore[reportUnknownMemberType]
    x3: NDArray[np.float64] = np.column_stack([cond_z, g, cond_z * g])  # type: ignore[reportUnknownMemberType]
    ll1, c1 = _fit_proportional_odds(endog, x1)
    ll2, c2 = _fit_proportional_odds(endog, x2)
    ll3, c3 = _fit_proportional_odds(endog, x3)

    # Optimizer noise can make a nested log-likelihood rise by ~1e-10 under no DIF; a
    # likelihood-ratio chi-square cannot be negative, so clamp at zero.
    chi2_total = max(0.0, -2.0 * (ll1 - ll3))
    chi2_uniform = max(0.0, -2.0 * (ll1 - ll2))
    chi2_nonuniform = max(0.0, -2.0 * (ll2 - ll3))
    r2_delta = max(0.0, _nagelkerke(ll3, ll_null, n) - _nagelkerke(ll1, ll_null, n))

    return DifResult(
        chi2_total=chi2_total,
        chi2_uniform=chi2_uniform,
        chi2_nonuniform=chi2_nonuniform,
        p_total=float(chi2.sf(chi2_total, df=2)),  # type: ignore[reportUnknownMemberType]
        p_uniform=float(chi2.sf(chi2_uniform, df=1)),  # type: ignore[reportUnknownMemberType]
        p_nonuniform=float(chi2.sf(chi2_nonuniform, df=1)),  # type: ignore[reportUnknownMemberType]
        nagelkerke_r2_delta=r2_delta,
        dif_class=_classify_jodoin_gierl(r2_delta),
        conditioner_source=source,
        n_obs=n,
        reference_level=reference,
        focal_level=focal,
        converged=c1 and c2 and c3,
    )
