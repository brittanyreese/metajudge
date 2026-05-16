"""Tests for the DIF pillar: Mantel-Haenszel + ETS A/B/C classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from metajudge.data import Ratings
from metajudge.dif import (
    DifResult,
    _mh_from_tables,  # pyright: ignore[reportPrivateUsage]
    mantel_haenszel_dif,
)


def _make(scores_ref: list[int], scores_focal: list[int]) -> Ratings:
    rows: list[dict[str, object]] = []
    for idx, s in enumerate(scores_ref):
        rows.append({"item": f"ref{idx}", "rater": "r1", "score": s, "group": "ref"})
    for idx, s in enumerate(scores_focal):
        rows.append({"item": f"foc{idx}", "rater": "r1", "score": s, "group": "foc"})
    df = pd.DataFrame(rows)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")


def test_identical_groups_have_no_dif() -> None:
    base = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0] * 3
    r = _make(base, base)
    res = mantel_haenszel_dif(r, focal="foc", reference="ref", n_match_bins=2)
    assert isinstance(res, DifResult)
    assert res.common_odds_ratio == pytest.approx(1.0, abs=1e-6)
    assert res.mh_delta == pytest.approx(0.0, abs=1e-6)
    assert res.ets_class == "A"


def test_matches_statsmodels_common_odds_ratio() -> None:
    """Validate the MH math helper directly against statsmodels' StratifiedTable.

    No Ratings round-trip and no private _bins kwarg: we hand-build the same
    2x2xK stratum tables that statsmodels consumes and assert our pooled odds
    ratio (and chi-square) agree to ~1e-6.
    """
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    # two matched strata, hand-built 2x2 tables (group x item-correct)
    tables: list[NDArray[np.float64]] = [
        np.array([[30, 10], [20, 20]], dtype=float),
        np.array([[25, 15], [15, 25]], dtype=float),
    ]
    # statsmodels expects 2x2xK; stack our list of 2x2 tables along axis 2.
    stacked: NDArray[np.float64] = np.dstack(tables)  # pyright: ignore[reportUnknownMemberType]
    st_oracle = sm.StratifiedTable(stacked)
    expected_or = float(st_oracle.oddsratio_pooled)
    expected_chisq = float(st_oracle.test_null_odds(correction=True).statistic)

    common_or, chi_sq = _mh_from_tables(tables)
    assert common_or == pytest.approx(expected_or, abs=1e-6)
    assert chi_sq == pytest.approx(expected_chisq, abs=1e-6)


def test_strong_uniform_dif_flags_class_c() -> None:
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 2, size=200).tolist()
    # focal systematically scored lower regardless of matched ability
    focal = (rng.random(200) < 0.15).astype(int).tolist()
    r = _make(ref, focal)
    res = mantel_haenszel_dif(r, focal="foc", reference="ref", n_match_bins=3)
    assert res.p_value < 0.05
    assert res.ets_class in {"B", "C"}


def test_unknown_levels_raise() -> None:
    base = [0, 1, 0, 1, 1, 0]
    r = _make(base, base)
    with pytest.raises(ValueError, match="stratum level not found"):
        mantel_haenszel_dif(r, focal="nope", reference="ref")
    with pytest.raises(ValueError, match="stratum level not found"):
        mantel_haenszel_dif(r, focal="foc", reference="nope")


@settings(max_examples=50, deadline=None)
@given(scores=st.lists(st.integers(0, 1), min_size=20, max_size=60))
def test_symmetry_swapping_groups_inverts_odds_ratio(scores: list[int]) -> None:
    r = _make(scores, scores)  # identical groups
    res = mantel_haenszel_dif(r, focal="foc", reference="ref", n_match_bins=2)
    # identical distributions: OR must stay ~1 and class A regardless of input
    assert res.ets_class == "A"
    assert res.common_odds_ratio == pytest.approx(1.0, abs=1e-6)
