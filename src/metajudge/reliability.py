"""Reliability pillar: Krippendorff's alpha (commodity) with a bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import krippendorff as kd  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as _f_dist  # type: ignore[import-untyped]

from metajudge._constants import MIN_EFFECTIVE
from metajudge.data import Ratings

_LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio"]


@dataclass(frozen=True)
class AlphaResult:
    """Krippendorff's alpha with a percentile bootstrap CI.

    ``n_bootstrap`` is the requested number of resamples; ``n_effective`` is how
    many were actually realized. They differ when a resample is degenerate (a
    column draw with no ratable variation makes alpha undefined and the resample
    is dropped), in which case the CI rests on ``n_effective`` replicates, not
    ``n_bootstrap`` -- read a materially smaller ``n_effective`` as a low-precision
    CI rather than a full-strength one, or read ``ci_reliable``.
    """

    alpha: float
    ci_low: float
    ci_high: float
    level: str
    n_bootstrap: int
    n_effective: int

    @property
    def ci_reliable(self) -> bool:
        """Whether enough resamples were realized for a trustworthy percentile CI.

        ``False`` when fewer than ``MIN_EFFECTIVE`` replicates survived, whether
        because few were requested or many were dropped as degenerate: the bounds are then
        indicative only and the point estimate ``alpha`` is the honest summary. Mirrors
        :attr:`metajudge.dif.ClusterBootstrapDif.ci_reliable`.
        """
        return self.n_effective >= MIN_EFFECTIVE


def krippendorff_alpha(
    ratings: Ratings,
    *,
    level: str = "ordinal",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> AlphaResult:
    """Krippendorff's alpha with a percentile bootstrap confidence interval.

    The CI is the 2.5/97.5 percentile interval of the bootstrap distribution.
    This is the simple percentile method without bias correction or
    acceleration; for alpha it is known to undercover (the interval is too
    narrow) in small samples or near the boundaries (Hayes & Krippendorff, 2007,
    *Communication Methods and Measures*). Degenerate resamples (no ratable
    variation) are dropped; the realized replicate count is reported as
    ``AlphaResult.n_effective``, which is below ``n_bootstrap`` when that happens.

    The resample draws units (columns) with replacement and holds the rater panel
    fixed, so the CI reflects sampling of judged items, not sampling of raters. With
    a small panel that you intend to generalize over, this understates uncertainty
    about the panel itself.
    """
    matrix = ratings.coder_unit_matrix()
    lom = cast(_LevelOfMeasurement, level)
    point = float(kd.alpha(reliability_data=matrix, level_of_measurement=lom))  # type: ignore[reportUnknownMemberType]
    rng = np.random.default_rng(seed)
    n_units = matrix.shape[1]
    boot: list[float] = []
    for _ in range(n_bootstrap):
        # numpy types Generator.integers as Unknown under some interpreter/stub
        # combinations (e.g. CPython 3.11), so pyright cannot infer the index
        # dtype; it is a plain integer index array either way.
        cols = rng.integers(0, n_units, size=n_units)  # type: ignore[reportUnknownVariableType]
        sample: NDArray[np.float64] = matrix[:, cols]
        try:
            boot.append(float(kd.alpha(reliability_data=sample, level_of_measurement=lom)))  # type: ignore[reportUnknownMemberType]
        except (ValueError, ZeroDivisionError):
            continue
    if boot:
        ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))  # type: ignore[reportUnknownMemberType]
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return AlphaResult(
        alpha=point,
        ci_low=ci_low,
        ci_high=ci_high,
        level=level,
        n_bootstrap=n_bootstrap,
        n_effective=len(boot),
    )


@dataclass(frozen=True)
class IccResult:
    """ICC(2,1)/(2,k) with McGraw & Wong (1996) exact F-based confidence intervals.

    ``icc1_ci_*``/``icck_ci_*`` are the 95% absolute-agreement limits (McGraw & Wong,
    1996, *Psychological Methods*, Case 2). The ICC(2,k) interval is the Spearman-Brown
    step-up of the ICC(2,1) bounds, matching pingouin's ``ICC(A,1)``/``ICC(A,k)``.
    """

    icc1: float
    icck: float
    n_targets: int
    n_raters: int
    icc1_ci_low: float
    icc1_ci_high: float
    icck_ci_low: float
    icck_ci_high: float


def icc(ratings: Ratings) -> IccResult:
    wide = ratings.wide()
    if bool(wide.isna().to_numpy().any()):
        raise ValueError(
            "ICC(2,1)/(2,k) is defined on a complete crossed targets x raters matrix; "
            "missing cells found. Incomplete or partially-crossed designs require a "
            "variance-components estimator (ten Hove et al. 2024, Psychological Methods), "
            "not listwise deletion, which is biased and discards information. That "
            "estimator is outside this two-pillar screening package; metajudge refuses "
            "here rather than report a wrong reliability coefficient."
        )
    data = wide.to_numpy(dtype=float)  # targets x raters
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError(
            "ICC(2,1)/(2,k) is defined on a crossed design with at least 2 targets and 2 "
            f"raters; got {n} targets x {k} raters. With a single target or rater the "
            "between-rater and error mean squares are undefined (0/0), so no ICC exists to "
            "report."
        )

    grand = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((data - grand) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc1 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icck = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)

    # McGraw & Wong (1996) exact CI for the two-way random absolute-agreement ICC (Case 2).
    # The estimated denominator df v makes this an approximation to an exact interval; it is
    # the interval pingouin and the R irr package report for ICC(A,1)/ICC(A,k).
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    fj = ms_cols / ms_error
    # McGraw & Wong denominator df v: the numerator carries the k*icc*fj term, the
    # denominator's second summand does not (the two expressions differ, see the paper).
    base = n * (1.0 + (k - 1) * icc1) - k * icc1
    num = k * icc1 * fj + base
    v = df2 * num**2 / (df1 * (k * icc1 * fj) ** 2 + base**2)
    f_u = float(_f_dist.ppf(0.975, df1, v))  # type: ignore[reportUnknownMemberType]
    f_l = float(_f_dist.ppf(0.975, v, df1))  # type: ignore[reportUnknownMemberType]
    mixed = k * ms_cols + (k * n - k - n) * ms_error
    ci1_low = n * (ms_rows - f_u * ms_error) / (f_u * mixed + n * ms_rows)
    ci1_high = n * (f_l * ms_rows - ms_error) / (mixed + n * f_l * ms_rows)
    # ICC(2,k) limits are the Spearman-Brown step-up of the ICC(2,1) limits.
    cik_low = ci1_low * k / (1.0 + ci1_low * (k - 1))
    cik_high = ci1_high * k / (1.0 + ci1_high * (k - 1))
    return IccResult(
        icc1=float(icc1),
        icck=float(icck),
        n_targets=n,
        n_raters=k,
        icc1_ci_low=float(ci1_low),
        icc1_ci_high=float(ci1_high),
        icck_ci_low=float(cik_low),
        icck_ci_high=float(cik_high),
    )
