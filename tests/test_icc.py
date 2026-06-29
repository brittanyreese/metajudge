# tests/test_icc.py
import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.reliability import icc


def _wide_to_ratings(wide: pd.DataFrame) -> Ratings:
    long = wide.reset_index(names="item").melt(id_vars="item", var_name="rater", value_name="score")
    return Ratings.from_long(long, item="item", rater="rater", score="score")


def _shrout_fleiss_table() -> pd.DataFrame:
    # Shrout & Fleiss (1979) Table 2 — canonical worked example.
    return pd.DataFrame(
        {
            "j1": [9, 6, 8, 7, 10, 6],
            "j2": [2, 1, 4, 1, 5, 2],
            "j3": [5, 3, 6, 2, 6, 4],
            "j4": [8, 2, 8, 6, 9, 7],
        },
        index=["t1", "t2", "t3", "t4", "t5", "t6"],
    )


def test_icc_matches_known_shrout_fleiss_value() -> None:
    r = _wide_to_ratings(_shrout_fleiss_table())
    res = icc(r)
    # Shrout & Fleiss ICC(2,1) = 0.290 for this table.
    assert res.icc1 == pytest.approx(0.290, abs=1e-3)
    assert res.n_targets == 6
    assert res.n_raters == 4


def test_icc_matches_pingouin_oracle() -> None:
    pg = pytest.importorskip("pingouin")
    wide = _shrout_fleiss_table()
    r = _wide_to_ratings(wide)
    long = wide.reset_index(names="targets").melt(
        id_vars="targets", var_name="raters", value_name="scores"
    )
    table = pg.intraclass_corr(
        data=long, targets="targets", raters="raters", ratings="scores"
    ).set_index("Type")
    res = icc(r)
    # Two-way random-effects, absolute agreement = pingouin's ICC(A,1)/ICC(A,k)
    # (older pingouin labelled these ICC2/ICC2k).
    assert res.icc1 == pytest.approx(float(table.loc["ICC(A,1)", "ICC"]), abs=1e-6)
    assert res.icck == pytest.approx(float(table.loc["ICC(A,k)", "ICC"]), abs=1e-6)


def test_icc_rejects_missing_cells_with_cited_guidance() -> None:
    # Shrout-Fleiss ICC is defined on a complete crossed design. For incomplete /
    # partially-crossed data the correct estimand is a variance-components estimator
    # (ten Hove et al. 2024), not listwise deletion (biased). That machinery is outside
    # this two-pillar package, so metajudge refuses rather than ship a wrong number.
    wide = _shrout_fleiss_table()
    wide.loc["t1", "j1"] = None
    r = _wide_to_ratings(wide)
    with pytest.raises(ValueError) as exc:
        icc(r)
    msg = str(exc.value)
    assert "missing" in msg  # still names the proximate cause
    assert "variance-components" in msg  # the correct estimator for incomplete data
    assert "listwise" in msg  # names the biased fallback it is refusing to use
    assert "outside this two-pillar" in msg
