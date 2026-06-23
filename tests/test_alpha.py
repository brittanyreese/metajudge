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


def test_n_effective_equals_n_bootstrap_on_clean_run() -> None:
    # A well-varied 9-unit matrix yields no degenerate resamples, so every
    # requested replicate is realized and the CI rests on the full count.
    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2],
        [1, 2, 3, 3, 2, 2, 4, 1, 3],
        [1, 1, 3, 3, 2, 1, 4, 2, 2],
    ]
    r = _ratings_from_matrix(matrix)
    res = krippendorff_alpha(r, level="ordinal", n_bootstrap=500, seed=42)
    assert res.n_effective == res.n_bootstrap == 500


def test_n_effective_counts_only_realized_resamples() -> None:
    # With two units, many bootstrap resamples draw the same unit twice, leaving
    # no ratable variation; kd.alpha raises and the resample is dropped. The CI is
    # then computed on fewer-than-requested replicates, and n_effective must report
    # that realized count (strictly below n_bootstrap) so the gap is visible.
    r = _ratings_from_matrix([[1, 0], [1, 0]])
    res = krippendorff_alpha(r, level="nominal", n_bootstrap=200, seed=1)
    assert 0 < res.n_effective < res.n_bootstrap
    assert res.n_bootstrap == 200


def test_ci_reliable_tracks_effective_count() -> None:
    # Parity with ClusterBootstrapDif.ci_reliable: True only when enough resamples
    # survive for a trustworthy percentile CI (the shared 100-replicate floor).
    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2],
        [1, 2, 3, 3, 2, 2, 4, 1, 3],
        [1, 1, 3, 3, 2, 1, 4, 2, 2],
    ]
    r = _ratings_from_matrix(matrix)
    ok = krippendorff_alpha(r, level="ordinal", n_bootstrap=500, seed=42)
    assert ok.n_effective >= 100
    assert ok.ci_reliable is True
    # too few resamples for a trustworthy CI even with none dropped
    thin = krippendorff_alpha(r, level="ordinal", n_bootstrap=50, seed=42)
    assert thin.n_effective < 100
    assert thin.ci_reliable is False
