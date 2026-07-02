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

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from scipy.stats import chi2, norm  # type: ignore[import-untyped]

from metajudge._constants import MIN_EFFECTIVE
from metajudge.data import Ratings

# Jodoin & Gierl (2001) Nagelkerke R-squared change thresholds.
_JG_NEGLIGIBLE = 0.035
_JG_LARGE = 0.070


# ── Brant proportional-odds diagnostic (private; surfaced via DifResult.po_violation) ──


@dataclass(frozen=True)
class _BrantResult:
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
    pi: NDArray[np.float64] = np.asarray(expit(design @ beta), dtype=np.float64)
    return beta, pi, bool(res.success)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _brant_test(
    endog: NDArray[np.int_],
    exog: NDArray[np.float64],
    *,
    names: list[str] | None = None,
) -> _BrantResult:
    """Brant (1990) Wald-type test of the proportional-odds assumption.

    Args:
        endog: ordinal categories, shape ``(n,)``.
        exog: design matrix without intercept, shape ``(n, p)``.
        names: optional predictor names (length p); defaults to ``x0..x{p-1}``.
    """
    y = np.asarray(endog, dtype=int)
    x = np.asarray(exog, dtype=float)
    n, p = x.shape
    levels: NDArray[np.int_] = np.unique(y)  # type: ignore[reportUnknownMemberType]
    m = int(levels.size) - 1
    if m < 2:
        # Proportional-odds requires at least 3 ordinal levels; test is not applicable.
        return _BrantResult(
            omnibus_chi2=0.0, omnibus_df=0, omnibus_p=1.0, per_predictor={}, converged=True
        )
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

    inv_info: list[NDArray[np.float64]] = []
    for k in range(m):
        w = pis[k] * (1.0 - pis[k])
        info_k = design.T @ (design * w[:, None])
        if np.linalg.cond(info_k) > 1e10:
            # Near-perfect separation: pinv would return a large pseudoinverse and inflate
            # chi2, producing a spurious po_violation signal. Bail early with converged=False
            # so the caller treats the Brant result as unreliable (advisory-only test).
            return _BrantResult(
                omnibus_chi2=0.0, omnibus_df=0, omnibus_p=1.0, per_predictor={}, converged=False
            )
        inv_info.append(np.asarray(np.linalg.pinv(info_k), dtype=np.float64))

    big = m * q
    vmat: NDArray[np.float64] = np.zeros((big, big))
    for k in range(m):
        vmat[k * q : (k + 1) * q, k * q : (k + 1) * q] = inv_info[k]
    for k in range(m):
        for j in range(k + 1, m):
            wkj = pis[j] * (1.0 - pis[k])
            cross = inv_info[k] @ (design.T @ (design * wkj[:, None])) @ inv_info[j]
            vmat[k * q : (k + 1) * q, j * q : (j + 1) * q] = cross
            vmat[j * q : (j + 1) * q, k * q : (k + 1) * q] = cross.T

    beta_stack: NDArray[np.float64] = np.concatenate(betas)  # type: ignore[reportUnknownMemberType]

    def _wald(predictor_cols: list[int]) -> tuple[float, int, bool]:
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
        if np.linalg.cond(cov) > 1e10:
            return 0.0, rows, False
        stat = float(d @ np.linalg.pinv(cov) @ d)
        return stat, rows, True

    all_slopes = list(range(1, q))
    omnibus_chi2, omnibus_df, ok_wald = _wald(all_slopes)
    if not ok_wald:
        return _BrantResult(
            omnibus_chi2=0.0, omnibus_df=0, omnibus_p=1.0, per_predictor={}, converged=False
        )
    omnibus_p = float(chi2.sf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0  # type: ignore[reportUnknownMemberType]

    per_predictor: dict[str, tuple[float, int, float]] = {}
    for idx in range(p):
        stat, df, ok_wald = _wald([1 + idx])
        if not ok_wald:
            return _BrantResult(
                omnibus_chi2=0.0, omnibus_df=0, omnibus_p=1.0, per_predictor={}, converged=False
            )
        pval = float(chi2.sf(stat, df)) if df > 0 else 1.0  # type: ignore[reportUnknownMemberType]
        per_predictor[names[idx]] = (stat, df, pval)

    return _BrantResult(
        omnibus_chi2=omnibus_chi2,
        omnibus_df=omnibus_df,
        omnibus_p=omnibus_p,
        per_predictor=per_predictor,
        converged=converged,
    )


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
    po_violation: bool
    conditioner_group_corr: float

    @property
    def conditioner_is_external(self) -> bool:
        """Whether the conditioner came from an external source rather than the panel rest score."""
        return self.conditioner_source == "external"


def _classify_jodoin_gierl(r2_delta: float) -> str:
    """Map a Nagelkerke R-squared change to an A/B/C DIF magnitude class.

    Jodoin & Gierl (2001): negligible (A) below 0.035, moderate (B) in
    ``[0.035, 0.070)``, large (C) at or above 0.070. These are an R-squared magnitude
    rule, not the ETS Mantel-Haenszel delta classification.

    Returns ``"?"`` when ``r2_delta`` is NaN (signals an optimization failure upstream).
    """
    if np.isnan(r2_delta):  # type: ignore[reportUnknownMemberType]
        return "?"
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


# Optimizer noise can make a nested log-likelihood rise by ~1e-10 under no DIF; a
# likelihood-ratio chi-square cannot be negative, so clamp at zero. But a *meaningful*
# negative -- the fuller model fitting worse than the model it nests -- is impossible
# unless the fit failed, and the bare clamp would mask such a divergence as a clean null.
_LR_NOISE_TOL = 1e-6


def _lr_chi2(ll_reduced: float, ll_full: float, tol: float = _LR_NOISE_TOL) -> tuple[float, bool]:
    """Nested likelihood-ratio chi-square ``-2(ll_reduced - ll_full)`` with a fit guard.

    Returns ``(chi2, ok)``. ``chi2`` is clamped at zero. ``ok`` is ``False`` when the raw
    statistic is negative beyond ``tol`` (the fuller model fit worse than the model it
    nests), which signals an optimization failure rather than the absence of an effect.
    """
    raw = -2.0 * (ll_reduced - ll_full)
    if raw < -tol:
        return 0.0, False
    return max(0.0, raw), True


@dataclass(frozen=True)
class _DifStats:
    """Numeric DIF statistics shared by the analytic fit and the bootstrap refits."""

    chi2_total: float
    chi2_uniform: float
    chi2_nonuniform: float
    nagelkerke_r2_delta: float
    n_obs: int
    converged: bool
    po_violation: bool
    conditioner_group_corr: float


def _emit_item_rows(
    rated: NDArray[np.float64],
    *,
    is_focal: bool,
    item_cond: float | None,
    scores: list[float],
    groups: list[float],
    cond_rows: list[float],
) -> None:
    """Append one (score, group, conditioner) row per rating of a single item.

    ``item_cond`` is the external item conditioner, or ``None`` for the leave-one-rater-out
    rest score (which needs at least two ratings on the item). Shared by :func:`logistic_dif`
    and the bootstrap so both assemble identical rows without a DataFrame round-trip.
    """
    group = 1.0 if is_focal else 0.0
    if item_cond is not None:
        for value in rated.tolist():
            cond_rows.append(item_cond)
            scores.append(float(value))
            groups.append(group)
        return
    count = int(rated.size)
    if count < 2:
        raise ValueError(
            f"cannot form an independent conditioner: an item has {count} rating(s) and no "
            "explicit conditioner was given. Provide >=2 raters per item or pass conditioner=."
        )
    total = float(rated.sum())
    for value in rated.tolist():
        # leave-one-rater-out rest score: mean of the other raters
        cond_rows.append((total - float(value)) / (count - 1))
        scores.append(float(value))
        groups.append(group)


def _dif_stats(
    scores: list[float],
    groups: list[float],
    cond_rows: list[float],
    *,
    want_split: bool,
    po_alpha: float = 1e-3,
) -> _DifStats:
    """Fit the nested proportional-odds models and return the DIF statistics.

    ``want_split`` controls whether the uniform/nonuniform decomposition is computed: the
    total chi-square and the Nagelkerke R-squared change need only the conditioner-only and
    full models, so the bootstrap (which reports a CI for the total only) skips the middle
    fit -- two fits per resample instead of three. When ``False`` the uniform and nonuniform
    statistics are ``nan`` and ``converged`` reflects the two fits and the total guard.

    Raises:
        ValueError: if all responses fall in one category, or the conditioner has no variance
            or is near-perfectly collinear with the group, so DIF is not identifiable.
    """
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
    x3: NDArray[np.float64] = np.column_stack([cond_z, g, cond_z * g])  # type: ignore[reportUnknownMemberType]
    ll1, c1 = _fit_proportional_odds(endog, x1)
    ll3, c3 = _fit_proportional_odds(endog, x3)

    chi2_total, ok_total = _lr_chi2(ll1, ll3)
    r2_raw = _nagelkerke(ll3, ll_null, n) - _nagelkerke(ll1, ll_null, n)
    r2_delta = max(0.0, r2_raw) if ok_total else float("nan")

    if not want_split:
        return _DifStats(
            chi2_total=chi2_total,
            chi2_uniform=float("nan"),
            chi2_nonuniform=float("nan"),
            nagelkerke_r2_delta=r2_delta,
            n_obs=n,
            converged=c1 and c3 and ok_total,
            po_violation=False,
            conditioner_group_corr=correlation,
        )

    # The three statistics are a telescoping nested decomposition: pre-clamp,
    # chi2_total == chi2_uniform + chi2_nonuniform exactly (ll2 cancels). The per-test
    # clamp is the only non-additivity source, and it doubles as a divergence guard.
    x2: NDArray[np.float64] = np.column_stack([cond_z, g])  # type: ignore[reportUnknownMemberType]
    ll2, c2 = _fit_proportional_odds(endog, x2)
    chi2_uniform, ok_uniform = _lr_chi2(ll1, ll2)
    chi2_nonuniform, ok_nonuniform = _lr_chi2(ll2, ll3)

    # PO diagnostic on the full M3 design. A proportional-odds violation can be absorbed
    # into the group x conditioner interaction and misread as nonuniform DIF, so it is
    # flagged. Advisory only: the Brant test is oversensitive at large N (Harrell 2015,
    # Ch. 13), so po_alpha defaults strict (1e-3). Failures to fit leave the flag off.
    try:
        brant = _brant_test(endog, x3, names=["cond", "group", "cond_group"])
        po_violation = bool(brant.converged and brant.omnibus_p < po_alpha)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        po_violation = False

    return _DifStats(
        chi2_total=chi2_total,
        chi2_uniform=chi2_uniform,
        chi2_nonuniform=chi2_nonuniform,
        nagelkerke_r2_delta=r2_delta,
        n_obs=n,
        converged=c1 and c2 and c3 and ok_total and ok_uniform and ok_nonuniform,
        po_violation=po_violation,
        conditioner_group_corr=correlation,
    )


def logistic_dif(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    conditioner: Mapping[Hashable, float] | None = None,
    po_alpha: float = 1e-3,
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
    criterion-out mean across rubric dimensions) when you need a stronger instrument-level
    analysis; the interpretation is only as good as that conditioner.

    Two scope limitations to keep in mind. First, the conditioner enters the model as a
    single linear term, so the match assumes a linear quality-to-response relationship; a
    strongly nonlinear one leaves residual confounding (DIF impurity) that this screen does
    not remove. Second, this tests one focal-vs-reference pair. Sweeping many stratum pairs
    inflates the familywise error rate, and no multiplicity correction is applied here;
    adjust across the pairs you run (e.g. Holm) when screening more than one.

    Raises:
        ValueError: if ``focal`` or ``reference`` is not a stratum level; if an explicit
            conditioner omits an included item; if no independent conditioner can be
            formed (single rating per item and no explicit conditioner); if all responses
            fall in one category; or if the conditioner is constant or near-perfectly
            collinear with the group, so DIF is not identifiable.
    """
    if conditioner is None and len(ratings.raters) == 2:
        warnings.warn(
            "With only 2 raters the leave-one-rater-out rest score equals the other rater's "
            "score on each item, making the conditioner dependent on the studied response. "
            "Provide an external conditioner for a valid DIF analysis.",
            UserWarning,
            stacklevel=2,
        )
    strata = ratings.strata()
    for level in (focal, reference):
        if level not in strata:
            # Stratum keys are coerced to str (see Ratings.strata), so an integer label 1
            # is the key "1"; list the available levels so the mismatch is obvious.
            available = ", ".join(repr(k) for k in sorted(strata))
            raise ValueError(
                f"stratum level not found: {level!r}. Available levels (as strings): "
                f"{available}. Pass the label as a string."
            )

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
        item_cond: float | None
        if conditioner is not None:
            if item not in conditioner:
                raise ValueError(f"conditioner missing item: {item!r}")
            item_cond = float(conditioner[item])
        else:
            item_cond = None
        _emit_item_rows(
            rated,
            is_focal=item in focal_items,
            item_cond=item_cond,
            scores=scores,
            groups=groups,
            cond_rows=cond_rows,
        )

    stats = _dif_stats(scores, groups, cond_rows, want_split=True, po_alpha=po_alpha)

    if stats.n_obs < 500:
        if stats.n_obs < 200:
            _n_obs_msg = (
                f"n_obs={stats.n_obs}: the Jodoin-Gierl A/B/C classification was calibrated on "
                "educational testing datasets with ≥500 examinees; at this sample size the "
                "R²-change thresholds overlap and the class should be treated as indicative only."
            )
        else:
            _n_obs_msg = (
                f"n_obs={stats.n_obs}: the Jodoin-Gierl A/B/C classification was calibrated on "
                ">=500 examinees; between 200-499 the thresholds have moderate calibration "
                "support and the class should be treated as indicative rather than definitive."
            )
        warnings.warn(_n_obs_msg, UserWarning, stacklevel=2)

    return DifResult(
        chi2_total=stats.chi2_total,
        chi2_uniform=stats.chi2_uniform,
        chi2_nonuniform=stats.chi2_nonuniform,
        p_total=float(chi2.sf(stats.chi2_total, df=2)),  # type: ignore[reportUnknownMemberType]
        p_uniform=float(chi2.sf(stats.chi2_uniform, df=1)),  # type: ignore[reportUnknownMemberType]
        p_nonuniform=float(chi2.sf(stats.chi2_nonuniform, df=1)),  # type: ignore[reportUnknownMemberType]
        nagelkerke_r2_delta=stats.nagelkerke_r2_delta,
        dif_class=_classify_jodoin_gierl(stats.nagelkerke_r2_delta),
        conditioner_source=source,
        n_obs=stats.n_obs,
        reference_level=reference,
        focal_level=focal,
        converged=stats.converged,
        po_violation=stats.po_violation,
        conditioner_group_corr=stats.conditioner_group_corr,
    )


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down familywise-error correction for a family of DIF tests.

    When you screen more than one focal-vs-reference stratum pair, the per-pair p-values
    (:attr:`DifResult.p_total`, or the uniform/nonuniform p-values) form a family and their
    uncorrected significance inflates the familywise error rate. Pass them here to get
    adjusted p-values that control it. The returned list aligns with the input order, and the
    adjustment is monotone in the sorted p-values. Reproduces
    ``statsmodels.stats.multitest.multipletests(method="holm")``.
    """
    p: NDArray[np.float64] = np.asarray(pvalues, dtype=float)
    m = int(p.size)
    if m == 0:
        return []
    order: NDArray[np.intp] = np.argsort(p)  # type: ignore[reportUnknownMemberType]
    adjusted: NDArray[np.float64] = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        # Step-down: the rank-th smallest p is scaled by the number of hypotheses not yet
        # rejected, then made monotone non-decreasing via a running max, and capped at 1.
        running = max(running, (m - rank) * float(p[idx]))
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


@dataclass(frozen=True)
class DifSweep:
    """A family of DIF comparisons with Holm-corrected familywise p-values.

    ``results`` holds one :class:`DifResult` per requested pair, aligned with ``pairs``
    (each a ``(focal, reference)`` tuple). ``p_total_holm``/``p_uniform_holm``/
    ``p_nonuniform_holm`` are the per-pair p-values after a Holm-Bonferroni step-down over
    the family, one correction per test type, aligned with ``results``. Read the corrected
    p-values, not the raw ones, when deciding significance across more than one pair.
    """

    results: list[DifResult]
    pairs: list[tuple[str, str]]
    p_total_holm: list[float]
    p_uniform_holm: list[float]
    p_nonuniform_holm: list[float]


def sweep(
    ratings: Ratings,
    *,
    pairs: Sequence[tuple[str, str]],
    conditioner: Mapping[Hashable, float] | None = None,
    po_alpha: float = 1e-3,
) -> DifSweep:
    """Run :func:`logistic_dif` over several focal-vs-reference pairs with Holm correction.

    Screening more than one stratum pair makes the per-pair p-values a family whose
    uncorrected significance inflates the familywise error rate. This runs each pair through
    the same engine and applies :func:`holm_adjust` across the family, separately for the
    total, uniform, and nonuniform tests, so the caller reads significance off corrected
    p-values rather than reassembling the correction by hand. Each ``(focal, reference)``
    tuple is passed straight to :func:`logistic_dif`; the same ``conditioner`` and
    ``po_alpha`` apply to every pair.

    Raises:
        ValueError: if ``pairs`` is empty, or (via :func:`logistic_dif`) for any per-pair
            identifiability or stratum-lookup failure.
    """
    pair_list = [(str(f), str(r)) for f, r in pairs]
    if not pair_list:
        raise ValueError("sweep needs at least one (focal, reference) pair")
    results = [
        logistic_dif(
            ratings, focal=focal, reference=reference, conditioner=conditioner, po_alpha=po_alpha
        )
        for focal, reference in pair_list
    ]
    return DifSweep(
        results=results,
        pairs=pair_list,
        p_total_holm=holm_adjust([r.p_total for r in results]),
        p_uniform_holm=holm_adjust([r.p_uniform for r in results]),
        p_nonuniform_holm=holm_adjust([r.p_nonuniform for r in results]),
    )


def _bca_bounds(
    boot: NDArray[np.float64],
    theta_hat: float,
    jackknife: NDArray[np.float64],
    pct_low: float,
    pct_high: float,
) -> tuple[float, float, bool]:
    """Bias-corrected and accelerated (BCa) percentile bounds (Efron 1987).

    ``pct_low``/``pct_high`` are the target percentiles (e.g. 2.5 and 97.5). ``jackknife``
    holds the leave-one-cluster-out point estimates for the acceleration term. Returns
    ``(low, high, ok)``; ``ok`` is ``False`` when the bias correction or acceleration is
    non-finite (a boundary-degenerate statistic), and the caller should fall back to the
    plain percentile interval. Matches ``scipy.stats.bootstrap(method="BCa")``.
    """
    b: NDArray[np.float64] = np.asarray(boot, dtype=float)
    jack: NDArray[np.float64] = np.asarray(jackknife, dtype=float)
    prop = float(np.mean(b < theta_hat))
    if prop <= 0.0 or prop >= 1.0 or jack.size < 2:
        return float("nan"), float("nan"), False
    z0 = float(norm.ppf(prop))  # type: ignore[reportUnknownMemberType]
    diff = jack.mean() - jack
    denom = 6.0 * float(np.sum(diff**2)) ** 1.5
    a = float(np.sum(diff**3) / denom) if denom > 0.0 else 0.0

    def _adjust(pct: float) -> float:
        z = float(norm.ppf(pct / 100.0))  # type: ignore[reportUnknownMemberType]
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))))  # type: ignore[reportUnknownMemberType]

    adj_low, adj_high = _adjust(pct_low), _adjust(pct_high)
    if not (np.isfinite(adj_low) and np.isfinite(adj_high)) or adj_low >= adj_high:
        return float("nan"), float("nan"), False
    lo, hi = (float(x) for x in np.percentile(b, [100.0 * adj_low, 100.0 * adj_high]))  # type: ignore[reportUnknownMemberType]
    return lo, hi, True


# With fewer items per group the item-cluster bootstrap draws from too small a space of
# distinct resample compositions for the percentile CI to be stable, even if n_effective
# clears the convergence floor. The number of distinct resamples (multisets) of n items is
# C(2n-1, n): only 126 at n=5 and 35 at n=4 — sparse enough that the 2.5/97.5 percentiles
# jump between a handful of discrete values. Below 5 the CI should be treated as unstable.
_MIN_CLUSTER_SIZE = 5


@dataclass(frozen=True)
class ClusterBootstrapDif:
    """Cluster-bootstrap robustness layer over an analytic :class:`DifResult`.

    The analytic likelihood-ratio test pools every (item, rater) cell as i.i.d.; under the
    clustering of a crossed rater-by-item design that makes it anti-conservative. This
    resamples whole item blocks (all of an item's rater scores together), stratified within
    the focal and reference groups, and reports percentile confidence intervals from refits
    of the validated engine. ``ci_level`` is the confidence level the bounds describe.
    ``n_effective`` is the count of non-degenerate resamples; when it falls below the
    reliability floor the bounds are indicative only and ``ci_reliable`` is ``False``. See
    docs/decisions/2026-06-23-e07-dif-cluster-bootstrap.md.

    ``ci_method`` records how the bounds were formed: ``"bca"`` is the bias-corrected and
    accelerated interval (Efron 1987), with the acceleration from a leave-one-cluster-out
    jackknife; ``"percentile"`` is the plain percentile fallback used when the statistic is
    boundary-degenerate (the Nagelkerke R-squared change is bounded below at 0, where the
    bias correction is undefined). Near that boundary a plain-percentile bound is fragile, so
    a negligible-vs-nonnegligible verdict from one is indicative, not decisive; read the
    point estimate alongside it. BCa corrects the bias but the bound is still an estimate.
    """

    base: DifResult
    r2_delta_ci_low: float
    r2_delta_ci_high: float
    chi2_total_ci_low: float
    chi2_total_ci_high: float
    cluster: str
    ci_level: float
    n_boot: int
    n_effective: int
    ci_method: str

    @property
    def ci_reliable(self) -> bool:
        """Whether enough resamples survived for a trustworthy percentile CI.

        ``False`` when fewer than ``MIN_EFFECTIVE`` resamples cleared the engine's
        identifiability guards: with so few draws the percentile bounds are noise dressed as
        precision, and the point estimate (``base``) is the honest summary.
        """
        return self.n_effective >= MIN_EFFECTIVE


def cluster_bootstrap_dif(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    conditioner: Mapping[Hashable, float] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
    po_alpha: float = 1e-3,
) -> ClusterBootstrapDif:
    """Cluster bootstrap of :func:`logistic_dif` resampling item blocks.

    Keeps the analytic point estimate (``base``) and adds confidence intervals at level
    ``ci`` (default 0.95 -> 2.5/97.5) for the Nagelkerke R-squared change and the total
    chi-square, computed by resampling items with replacement, stratified within the focal
    and reference groups, with each item's full rater block carried intact. This makes the
    effective sample the number of independent item clusters per group rather than the pooled
    cell count, and preserves the within-item rater correlation the analytic i.i.d. model
    ignores.

    The intervals are bias-corrected and accelerated (BCa, Efron 1987) when computable, with
    the acceleration from a leave-one-cluster-out jackknife; ``ClusterBootstrapDif.ci_method``
    reports ``"bca"``. It falls back to the plain percentile interval (``"percentile"``) in
    two cases: when the statistic is boundary-degenerate (the R-squared change piles at its 0
    lower bound, where the bias correction is undefined), and when the panel is large enough
    that the jackknife (one refit per item cluster) would add more fits than the bootstrap
    itself. That gate keeps a robust run at ``O(n_boot)`` fits: BCa is applied where it both
    helps most (small samples, where the percentile interval undercovers) and is cheap.

    Degenerate resamples (a draw with no ordinal variation, or one that fails the engine's
    identifiability guards) are dropped; the realized count is ``n_effective``. Below the
    reliability floor (100 surviving resamples) the bounds are indicative only and
    ``ci_reliable`` is ``False`` -- read the point estimate instead.

    Raises:
        ValueError: if ``ci`` is not strictly inside ``(0, 1)``.
        ValueError: if ``ratings`` has no stratum column (pass ``stratum=`` to
            :meth:`Ratings.from_long <metajudge.data.Ratings.from_long>`).
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}")
    tail = 100.0 * (1.0 - ci) / 2.0
    pct_low, pct_high = tail, 100.0 - tail
    base = logistic_dif(
        ratings, focal=focal, reference=reference, conditioner=conditioner, po_alpha=po_alpha
    )
    strata = ratings.strata()
    focal_items: list[Hashable] = list(strata[focal])
    reference_items: list[Hashable] = list(strata[reference])

    # Each item's block is its multiset of rated values; rater identity does not enter the
    # engine, so the bootstrap feeds these straight into the shared assembly/fit helpers and
    # skips the DataFrame -> Ratings -> wide round-trip the analytic call needs.
    mat: NDArray[np.float64] = ratings.wide().to_numpy(dtype=float)  # items x raters
    item_pos: dict[Hashable, int] = {item: i for i, item in enumerate(ratings.items)}
    block_values: dict[Hashable, NDArray[np.float64]] = {}
    for item in focal_items + reference_items:
        row: NDArray[np.float64] = mat[item_pos[item]]
        block_values[item] = row[~np.isnan(row)]

    rng = np.random.default_rng(seed)
    n_focal = len(focal_items)
    n_reference = len(reference_items)
    if n_focal < _MIN_CLUSTER_SIZE or n_reference < _MIN_CLUSTER_SIZE:
        warnings.warn(
            f"cluster_bootstrap_dif: focal group has {n_focal} item(s), reference has "
            f"{n_reference}; with fewer than {_MIN_CLUSTER_SIZE} items per group the bootstrap "
            "draws from too small a space of distinct resamples for a stable percentile CI "
            "(see ci_reliable). Treat the CI bounds as indicative only.",
            UserWarning,
            stacklevel=2,
        )
    chi2_totals: list[float] = []
    r2_deltas: list[float] = []

    for _ in range(n_boot):
        draw_focal = rng.integers(0, n_focal, size=n_focal)
        draw_reference = rng.integers(0, n_reference, size=n_reference)
        scores: list[float] = []
        groups: list[float] = []
        cond_rows: list[float] = []
        try:
            for j in range(n_focal):
                item = focal_items[int(draw_focal[j])]
                _emit_item_rows(
                    block_values[item],
                    is_focal=True,
                    item_cond=float(conditioner[item]) if conditioner is not None else None,
                    scores=scores,
                    groups=groups,
                    cond_rows=cond_rows,
                )
            for j in range(n_reference):
                item = reference_items[int(draw_reference[j])]
                _emit_item_rows(
                    block_values[item],
                    is_focal=False,
                    item_cond=float(conditioner[item]) if conditioner is not None else None,
                    scores=scores,
                    groups=groups,
                    cond_rows=cond_rows,
                )
            # Only the total statistic gets a bootstrap CI, so the middle (uniform) fit is
            # skipped: two proportional-odds fits per resample instead of three.
            stats = _dif_stats(scores, groups, cond_rows, want_split=False)
        except ValueError:
            continue
        if not stats.converged:
            continue
        chi2_totals.append(stats.chi2_total)
        r2_deltas.append(stats.nagelkerke_r2_delta)

    if not r2_deltas:
        warnings.warn(
            f"cluster_bootstrap_dif: all {n_boot} resamples were degenerate or failed to "
            "converge; CI bounds are NaN. Inspect the base result for identifiability issues "
            "or provide an external conditioner.",
            UserWarning,
            stacklevel=2,
        )
        return ClusterBootstrapDif(
            base=base,
            r2_delta_ci_low=float("nan"),
            r2_delta_ci_high=float("nan"),
            chi2_total_ci_low=float("nan"),
            chi2_total_ci_high=float("nan"),
            cluster="item",
            ci_level=ci,
            n_boot=n_boot,
            n_effective=0,
            ci_method="percentile",
        )

    r2_arr: NDArray[np.float64] = np.asarray(r2_deltas, dtype=float)
    c2_arr: NDArray[np.float64] = np.asarray(chi2_totals, dtype=float)

    # BCa needs a leave-one-cluster-out jackknife: one analytic refit per item cluster. That
    # is worth it when the sample is small (where the percentile interval undercovers and the
    # jackknife is cheap), but for a large panel the jackknife dominates the whole run and the
    # percentile interval is already decent. Gate on cost: run BCa only when the jackknife
    # would not add more fits than the bootstrap itself; otherwise keep the percentile CI.
    ci_method = "percentile"
    r2_low = r2_high = c2_low = c2_high = float("nan")
    combined: list[tuple[Hashable, bool]] = [(it, True) for it in focal_items] + [
        (it, False) for it in reference_items
    ]
    if len(combined) <= n_boot:
        r2_jack: list[float] = []
        c2_jack: list[float] = []
        for leave in range(len(combined)):
            j_scores: list[float] = []
            j_groups: list[float] = []
            j_cond: list[float] = []
            n_f = n_r = 0
            for pos, (item, is_focal) in enumerate(combined):
                if pos == leave:
                    continue
                _emit_item_rows(
                    block_values[item],
                    is_focal=is_focal,
                    item_cond=float(conditioner[item]) if conditioner is not None else None,
                    scores=j_scores,
                    groups=j_groups,
                    cond_rows=j_cond,
                )
                n_f += is_focal
                n_r += not is_focal
            if n_f == 0 or n_r == 0:
                continue
            try:
                j_stats = _dif_stats(j_scores, j_groups, j_cond, want_split=False)
            except ValueError:
                continue
            if not j_stats.converged:
                continue
            r2_jack.append(j_stats.nagelkerke_r2_delta)
            c2_jack.append(j_stats.chi2_total)

        r2_jack_arr: NDArray[np.float64] = np.asarray(r2_jack, dtype=float)
        c2_jack_arr: NDArray[np.float64] = np.asarray(c2_jack, dtype=float)
        r2_low, r2_high, ok_r2 = _bca_bounds(
            r2_arr, base.nagelkerke_r2_delta, r2_jack_arr, pct_low, pct_high
        )
        c2_low, c2_high, ok_c2 = _bca_bounds(
            c2_arr, base.chi2_total, c2_jack_arr, pct_low, pct_high
        )
        if ok_r2 and ok_c2:
            ci_method = "bca"

    if ci_method == "percentile":
        # Either the panel was too large to justify the jackknife, or BCa was undefined for a
        # boundary-degenerate statistic (the R2-change piles at 0); use the percentile CI.
        r2_low, r2_high = (float(x) for x in np.percentile(r2_arr, [pct_low, pct_high]))  # type: ignore[reportUnknownMemberType]
        c2_low, c2_high = (float(x) for x in np.percentile(c2_arr, [pct_low, pct_high]))  # type: ignore[reportUnknownMemberType]

    return ClusterBootstrapDif(
        base=base,
        r2_delta_ci_low=r2_low,
        r2_delta_ci_high=r2_high,
        chi2_total_ci_low=c2_low,
        chi2_total_ci_high=c2_high,
        cluster="item",
        ci_level=ci,
        n_boot=n_boot,
        n_effective=len(r2_deltas),
        ci_method=ci_method,
    )
