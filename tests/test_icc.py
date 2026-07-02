# tests/test_icc.py
import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.reliability import icc


def _wide_to_ratings(wide: pd.DataFrame) -> Ratings:
    long = wide.reset_index(names="item").melt(id_vars="item", var_name="rater", value_name="score")
    return Ratings.from_long(long, item="item", rater="rater", score="score")


def _shrout_fleiss_table() -> pd.DataFrame:
    # Shrout & Fleiss (1979) Table 2, the canonical worked example.
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
    # McGraw & Wong (1996) exact F-based CI, absolute agreement. pingouin rounds the
    # reported CI to 2 dp, so match at that tolerance; the literal pins below carry the
    # full-precision regression anchor.
    ci1 = table.loc["ICC(A,1)", "CI95"]
    cik = table.loc["ICC(A,k)", "CI95"]
    assert res.icc1_ci_low == pytest.approx(float(ci1[0]), abs=1e-2)
    assert res.icc1_ci_high == pytest.approx(float(ci1[1]), abs=1e-2)
    assert res.icck_ci_low == pytest.approx(float(cik[0]), abs=1e-2)
    assert res.icck_ci_high == pytest.approx(float(cik[1]), abs=1e-2)


def test_icc_ci_matches_pinned_mcgraw_wong_values() -> None:
    # Full-precision pin of the McGraw & Wong (1996) Case-2 (absolute-agreement) CI for
    # the Shrout-Fleiss table, cross-checked against pingouin's 2-dp report [0.02, 0.76]
    # and [0.07, 0.93]. Reference wins over any literal drift.
    res = icc(_wide_to_ratings(_shrout_fleiss_table()))
    assert res.icc1_ci_low == pytest.approx(0.018787, abs=1e-5)
    assert res.icc1_ci_high == pytest.approx(0.761084, abs=1e-5)
    assert res.icck_ci_low == pytest.approx(0.071137, abs=1e-5)
    assert res.icck_ci_high == pytest.approx(0.927232, abs=1e-5)
    # ICC(A,k) CI is the Spearman-Brown step-up of the ICC(A,1) CI bounds.
    assert res.icc1_ci_low < res.icc1 < res.icc1_ci_high
    assert res.icck_ci_low < res.icck < res.icck_ci_high


def test_icc_rejects_degenerate_dimensions() -> None:
    # ICC(2,1)/(2,k) needs >=2 targets and >=2 raters; a single rater leaves the
    # between-rater and error mean squares undefined (0/0). Refuse clearly rather than
    # raise a bare ZeroDivisionError or emit nan.
    wide = _shrout_fleiss_table()[["j1"]]  # one rater
    r = _wide_to_ratings(wide)
    with pytest.raises(ValueError) as exc:
        icc(r)
    assert "at least 2" in str(exc.value)


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
