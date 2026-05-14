# tests/test_alpha.py
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.reliability import AlphaResult, krippendorff_alpha


def _ratings_from_matrix(
    matrix: Sequence[Sequence[float]],
    stratum: list[str] | None = None,
) -> Ratings:
    # matrix is raters x items; build long form
    rows: list[dict[str, object]] = []
    for r_idx, row in enumerate(matrix):
        for i_idx, val in enumerate(row):
            if isinstance(val, float) and np.isnan(val):
                continue
            rec: dict[str, object] = {
                "item": f"i{i_idx}",
                "rater": f"r{r_idx}",
                "score": val,
            }
            if stratum is not None:
                rec["group"] = stratum[i_idx]
            rows.append(rec)
    df = pd.DataFrame(rows)
    return Ratings.from_long(
        df,
        item="item",
        rater="rater",
        score="score",
        stratum="group" if stratum else None,
    )


def test_perfect_agreement_alpha_is_one() -> None:
    r = _ratings_from_matrix([[1, 0, 1, 0], [1, 0, 1, 0]])
    res = krippendorff_alpha(r, level="nominal", n_bootstrap=200, seed=1)
    assert isinstance(res, AlphaResult)
    assert res.alpha == pytest.approx(1.0, abs=1e-9)


def test_matches_reference_library() -> None:
    import krippendorff as kd  # type: ignore[import-untyped]

    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2],
        [1, 2, 3, 3, 2, 2, 4, 1, 2],
        [1, 2, 3, 3, 2, 1, 4, 1, 2],
    ]
    r = _ratings_from_matrix(matrix)
    expected = kd.alpha(  # type: ignore[reportUnknownMemberType]
        reliability_data=np.array(matrix, dtype=float),
        level_of_measurement="ordinal",
    )
    res = krippendorff_alpha(r, level="ordinal", n_bootstrap=200, seed=1)
    assert res.alpha == pytest.approx(expected, abs=1e-9)


def test_ci_brackets_point_estimate() -> None:
    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2],
        [1, 2, 3, 3, 2, 2, 4, 1, 3],
        [1, 1, 3, 3, 2, 1, 4, 2, 2],
    ]
    r = _ratings_from_matrix(matrix)
    res = krippendorff_alpha(r, level="ordinal", n_bootstrap=500, seed=42)
    assert res.ci_low <= res.alpha <= res.ci_high
    assert res.n_bootstrap == 500
