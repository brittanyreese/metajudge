"""Reliability pillar: Krippendorff's alpha (commodity) with a bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import krippendorff as kd  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from metajudge.data import Ratings

_LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio"]

# Below this many realized resamples the percentile CI is too thin to trust. Kept in
# sync with dif._MIN_EFFECTIVE so the alpha and DIF pillars apply the same reliability
# floor and expose the same ``ci_reliable`` contract.
_MIN_EFFECTIVE = 100


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

        ``False`` when fewer than ``_MIN_EFFECTIVE`` (100) replicates survived, whether
        because few were requested or many were dropped as degenerate: the bounds are then
        indicative only and the point estimate ``alpha`` is the honest summary. Mirrors
        :attr:`metajudge.dif.ClusterBootstrapDif.ci_reliable`.
        """
        return self.n_effective >= _MIN_EFFECTIVE


def _alpha(matrix: NDArray[np.float64], level: str) -> float:
    lom = cast(_LevelOfMeasurement, level)
    return float(kd.alpha(reliability_data=matrix, level_of_measurement=lom))  # type: ignore[reportUnknownMemberType]


def krippendorff_alpha(
    ratings: Ratings,
    *,
    level: str = "nominal",
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
    """
    matrix = ratings.coder_unit_matrix()
    point = _alpha(matrix, level)
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
            boot.append(_alpha(sample, level))
        except (ValueError, ZeroDivisionError):
            continue
    if boot:
        ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))  # type: ignore[reportUnknownMemberType]
    else:
        ci_low, ci_high = point, point
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
    icc1: float
    icck: float
    n_targets: int
    n_raters: int


def icc(ratings: Ratings) -> IccResult:
    wide = ratings.wide()
    if bool(wide.isna().to_numpy().any()):
        raise ValueError(
            "ICC(2,1)/(2,k) is defined on a complete crossed targets x raters matrix; "
            "missing cells found. Incomplete or partially-crossed designs require a "
            "variance-components estimator (ten Hove et al. 2024, Psychological Methods), "
            "not listwise deletion, which is biased and discards information. That "
            "estimator is the deferred E04 variance pillar; E07 refuses here rather than "
            "report a wrong reliability coefficient."
        )
    data = wide.to_numpy(dtype=float)  # targets x raters
    n, k = data.shape

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
    return IccResult(icc1=float(icc1), icck=float(icck), n_targets=n, n_raters=k)
