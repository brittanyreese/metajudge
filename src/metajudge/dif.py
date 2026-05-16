"""DIF pillar: Mantel-Haenszel across one stratum with ETS A/B/C classification.

The signature contribution of this library. ``mantel_haenszel_dif`` dichotomizes
each item's score, matches items on binned judge ability, and pools the per-bin
2x2 (group x correctness) tables into the Mantel-Haenszel common odds ratio,
then maps that onto the ETS (Holland & Thayer) A/B/C effect-size scale.

statsmodels is *not* imported here; it is only an oracle in the test suite. The
sole runtime statistics dependency is ``scipy.stats.chi2`` for the p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2  # type: ignore[import-untyped]

from metajudge.data import Ratings


@dataclass(frozen=True)
class DifResult:
    """Mantel-Haenszel DIF outcome for one focal-vs-reference comparison."""

    common_odds_ratio: float
    mh_chi_square: float
    p_value: float
    mh_delta: float
    ets_class: str
    reference_level: str
    focal_level: str


def _mh_from_tables(tables: list[NDArray[np.float64]]) -> tuple[float, float]:
    """Pool a list of 2x2 stratum tables into the MH OR and MH chi-square.

    Each table is ``[[a, b], [c, d]]`` where row 0 is the reference group and
    row 1 the focal group, column 0 is "correct" and column 1 "incorrect".

    Returns ``(common_odds_ratio, mh_chi_square)`` where:

    - ``OR = sum(a_k * d_k / n_k) / sum(b_k * c_k / n_k)``
    - ``chi_square = (|sum a_k - sum E(a_k)| - 0.5)^2 / sum Var(a_k)``
      with ``E(a_k) = (a_k + b_k)(a_k + c_k) / n_k`` and
      ``Var(a_k) = (a+b)(c+d)(a+c)(b+d) / (n_k^2 (n_k - 1))``.

    Empty strata (``n_k == 0``) and single-observation strata contribute nothing
    to their respective sums. If the OR denominator is zero the odds ratio is
    ``inf``; if the variance sum is zero the chi-square is ``0.0``. This matches
    ``statsmodels.stats.contingency_tables.StratifiedTable`` to floating-point
    precision on well-posed tables.
    """
    num_or = 0.0
    den_or = 0.0
    sum_a = 0.0
    sum_ea = 0.0
    sum_var = 0.0
    for table in tables:
        a, b = float(table[0][0]), float(table[0][1])
        c, d = float(table[1][0]), float(table[1][1])
        n = a + b + c + d
        if n == 0:
            continue
        num_or += a * d / n
        den_or += b * c / n
        sum_a += a
        sum_ea += (a + b) * (a + c) / n
        if n > 1:
            sum_var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))

    common_or = num_or / den_or if den_or > 0 else math.inf
    chi_sq = (abs(sum_a - sum_ea) - 0.5) ** 2 / sum_var if sum_var > 0 else 0.0
    return common_or, chi_sq


def _classify(common_or: float, p_value: float) -> tuple[float, str]:
    """Map a common odds ratio + p-value to ETS delta and A/B/C class.

    ``mh_delta = -2.35 * ln(OR)``. Class A (negligible) if ``p >= 0.05`` or
    ``|delta| < 1.0``; class C (large) if ``p < 0.05`` and ``|delta| >= 1.5``;
    class B otherwise. A non-positive or non-finite odds ratio yields delta 0.0.
    """
    delta = -2.35 * math.log(common_or) if common_or > 0 and math.isfinite(common_or) else 0.0
    if p_value >= 0.05 or abs(delta) < 1.0:
        return delta, "A"
    if abs(delta) >= 1.5:
        return delta, "C"
    return delta, "B"


def mantel_haenszel_dif(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    n_match_bins: int = 4,
    score_threshold: float | None = None,
) -> DifResult:
    """Mantel-Haenszel DIF of ``focal`` vs ``reference`` with ETS classification.

    Each item's score (mean across raters) is dichotomized into "correct" /
    "incorrect" at ``score_threshold``. The default threshold is the median of
    the per-item mean scores; "correct" is ``score > threshold`` (falling back to
    ``score >= threshold`` when the strict split leaves no variation, e.g. a
    boundary-valued median). Items are matched on ``n_match_bins`` quantile bins
    of judge ability (the per-item mean), and the per-bin 2x2 group x correctness
    tables are pooled via :func:`_mh_from_tables`.

    When the matching variable and the dichotomized response are collinear (for
    example single-rater binary scores, where ability *is* the response), every
    ability bin separates the outcome and the matched strata carry no information.
    In that case the analysis falls back to the unmatched single pooled 2x2
    table, which is the correct degenerate-stratification limit of MH.

    Raises:
        ValueError: if ``focal`` or ``reference`` is not a stratum level.
    """
    strata = ratings.strata()
    for level in (focal, reference):
        if level not in strata:
            raise ValueError(f"stratum level not found: {level}")

    items = list(ratings.items)
    per_item = ratings.wide().mean(axis=1).reindex(items)

    scores: NDArray[np.float64] = per_item.to_numpy(dtype=float)
    threshold = float(per_item.median()) if score_threshold is None else score_threshold
    correct: NDArray[np.bool_] = scores > threshold
    if int(correct.sum()) in (0, len(items)):
        # Strict split collapsed (e.g. boundary-valued threshold); use >=.
        correct = scores >= threshold

    focal_items = set(strata[focal])
    reference_items = set(strata[reference])
    is_focal: NDArray[np.bool_] = np.array([item in focal_items for item in items])
    is_reference: NDArray[np.bool_] = np.array([item in reference_items for item in items])

    def _table(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        ref = is_reference & mask
        foc = is_focal & mask
        a = float((ref & correct).sum())
        b = float((ref & ~correct).sum())
        cc = float((foc & correct).sum())
        d = float((foc & ~correct).sum())
        return np.array([[a, b], [cc, d]], dtype=float)

    ranks = per_item.rank(method="first")
    bin_labels: NDArray[np.float64] = pd.qcut(
        ranks, q=n_match_bins, labels=False, duplicates="drop"
    ).to_numpy(dtype=float)

    all_items: NDArray[np.bool_] = np.ones(len(items), dtype=bool)
    labels: list[float] = sorted(
        {float(x) for x in bin_labels.tolist() if not math.isnan(float(x))}
    )
    tables: list[NDArray[np.float64]] = [_table(bin_labels == label) for label in labels]

    common_or, chi_sq = _mh_from_tables(tables)
    if common_or == 0.0 or not math.isfinite(common_or):
        # Matched stratification was uninformative (collinear ability/response).
        # Fall back to the unmatched single pooled 2x2 table.
        tables = [_table(all_items)]
        common_or, chi_sq = _mh_from_tables(tables)

    if common_or == 0.0 or not math.isfinite(common_or):
        # No outcome variation at all (e.g. a constant response). The MH odds
        # ratio is genuinely undefined; the only defensible DIF conclusion is
        # "no differential functioning": OR = 1, no significance, class A.
        common_or, chi_sq = 1.0, 0.0

    p_value = float(chi2.sf(chi_sq, df=1))  # type: ignore[reportUnknownMemberType]
    delta, ets = _classify(common_or, p_value)
    return DifResult(
        common_odds_ratio=float(common_or),
        mh_chi_square=float(chi_sq),
        p_value=p_value,
        mh_delta=float(delta),
        ets_class=ets,
        reference_level=reference,
        focal_level=focal,
    )
