"""Reliability pillar: Krippendorff's alpha (commodity) with a bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import krippendorff as kd  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from metajudge.data import Ratings

_LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio"]


@dataclass(frozen=True)
class AlphaResult:
    alpha: float
    ci_low: float
    ci_high: float
    level: str
    n_bootstrap: int


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
    matrix = ratings.coder_unit_matrix()
    point = _alpha(matrix, level)
    rng = np.random.default_rng(seed)
    n_units = matrix.shape[1]
    boot: list[float] = []
    for _ in range(n_bootstrap):
        cols = rng.integers(0, n_units, size=n_units)
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
        raise ValueError("ICC requires a complete matrix; missing cells found")
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
