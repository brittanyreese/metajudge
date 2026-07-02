"""Tests for the DIF pillar: ordinal logistic-regression DIF (Zumbo / lordif).

The numerical oracle is R ``MASS::polr`` (the canonical proportional-odds fit). The
frozen fixture below (24 items x 6 raters, a strong uniform DIF) was scored once, its
three nested models fit with polr, and the resulting statistics pinned as literal
constants. The engine is independently cross-checked against statsmodels ``Logit`` in
the binary (two-category) limit, where proportional-odds reduces to ordinary logistic
regression. statsmodels ``OrderedModel`` is deliberately not used: it does not
reproduce the canonical proportional-odds likelihood (its fit disagrees with both polr
and the binary-limit Logit MLE). When a reference value and a literal disagree, the
reference wins and the literal is corrected.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import metajudge.dif as dif_module
from metajudge.data import Ratings
from metajudge.dif import (
    ClusterBootstrapDif,
    DifResult,
    DifSweep,
    _bca_bounds,  # pyright: ignore[reportPrivateUsage]
    _classify_jodoin_gierl,  # pyright: ignore[reportPrivateUsage]
    _common_support,  # pyright: ignore[reportPrivateUsage]
    _DifStats,  # pyright: ignore[reportPrivateUsage]
    _fit_proportional_odds,  # pyright: ignore[reportPrivateUsage]
    _lr_chi2,  # pyright: ignore[reportPrivateUsage]
    cluster_bootstrap_dif,
    holm_adjust,
    logistic_dif,
    sweep,
)

# --- frozen DIF-bearing fixture (see scratch gen_olr_oracle.py, seed 20260622) ---
# items it00..it11 are reference, it12..it23 are focal; 6 raters each (rt0..rt5).
_QUALITY = [
    1.160389,
    -1.364047,
    -1.746011,
    0.138741,
    -1.48227,
    -0.378891,
    -0.892256,
    0.228572,
    -0.87114,
    0.616882,
    -0.297235,
    -0.077086,
    0.899641,
    1.183881,
    -0.341555,
    1.022717,
    -1.069796,
    0.31503,
    -1.503547,
    0.077358,
    -2.209854,
    -1.400007,
    0.170668,
    -0.163181,
]
_SCORES = [
    5,
    4,
    5,
    5,
    5,
    4,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    3,
    4,
    4,
    3,
    3,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    2,
    1,
    1,
    2,
    1,
    4,
    2,
    3,
    3,
    5,
    5,
    3,
    4,
    1,
    1,
    2,
    1,
    3,
    1,
    4,
    1,
    4,
    4,
    3,
    3,
    2,
    2,
    3,
    2,
    3,
    4,
    1,
    4,
    5,
    4,
    2,
    2,
    2,
    3,
    2,
    3,
    2,
    4,
    5,
    3,
    3,
    4,
    5,
    5,
    1,
    1,
    2,
    2,
    3,
    1,
    5,
    2,
    3,
    4,
    4,
    3,
    3,
    2,
    1,
    1,
    1,
    2,
    4,
    1,
    4,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    3,
    4,
    1,
    1,
    3,
    1,
    1,
    4,
    1,
]
_N_ITEMS = 24
_N_RATERS = 6

# Pinned R MASS::polr oracle for the frozen fixture (external conditioner).
_OR_CHI2_TOTAL = 20.283392
_OR_CHI2_UNIFORM = 19.445739
_OR_CHI2_NONUNIFORM = 0.837652
_OR_P_TOTAL = 0.00003940
_OR_P_UNIFORM = 0.00001035
_OR_P_NONUNIFORM = 0.36006908
_OR_NAGELKERKE_R2_DELTA = 0.074462

# Pinned R MASS::polr oracle for the same fixture via the leave-one-rater-out rest score.
# The rest score is contaminated by the group-shared bias, so the same data that is class C
# under the independent conditioner above is class A here. This pins the default path.
_RS_CHI2_UNIFORM = 0.458909
_RS_CHI2_NONUNIFORM = 0.010651
_RS_NAGELKERKE_R2_DELTA = 0.001982


def _frozen() -> tuple[Ratings, dict[Hashable, float]]:
    rows: list[dict[str, object]] = []
    for i in range(_N_ITEMS):
        item = f"it{i:02d}"
        group = "ref" if i < _N_ITEMS // 2 else "foc"
        for r in range(_N_RATERS):
            rows.append(
                {
                    "item": item,
                    "rater": f"rt{r}",
                    "score": _SCORES[i * _N_RATERS + r],
                    "group": group,
                }
            )
    df = pd.DataFrame(rows)
    ratings = Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")
    conditioner: dict[Hashable, float] = {f"it{i:02d}": _QUALITY[i] for i in range(_N_ITEMS)}
    return ratings, conditioner


def _make(
    scores_ref: list[list[int]],
    scores_focal: list[list[int]],
) -> Ratings:
    """Build Ratings from per-item lists of rater scores (one inner list per item)."""
    rows: list[dict[str, object]] = []
    for idx, item_scores in enumerate(scores_ref):
        for r, s in enumerate(item_scores):
            rows.append({"item": f"ref{idx}", "rater": f"rt{r}", "score": s, "group": "ref"})
    for idx, item_scores in enumerate(scores_focal):
        for r, s in enumerate(item_scores):
            rows.append({"item": f"foc{idx}", "rater": f"rt{r}", "score": s, "group": "foc"})
    df = pd.DataFrame(rows)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")


def test_matches_pinned_oracle_with_external_conditioner() -> None:
    ratings, conditioner = _frozen()
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert isinstance(res, DifResult)
    assert res.n_obs == _N_ITEMS * _N_RATERS
    assert res.conditioner_source == "external"
    assert res.chi2_total == pytest.approx(_OR_CHI2_TOTAL, abs=1e-3)
    assert res.chi2_uniform == pytest.approx(_OR_CHI2_UNIFORM, abs=1e-3)
    assert res.chi2_nonuniform == pytest.approx(_OR_CHI2_NONUNIFORM, abs=1e-3)
    assert res.p_total == pytest.approx(_OR_P_TOTAL, abs=1e-5)
    assert res.p_uniform == pytest.approx(_OR_P_UNIFORM, abs=1e-5)
    assert res.p_nonuniform == pytest.approx(_OR_P_NONUNIFORM, abs=1e-4)
    assert res.nagelkerke_r2_delta == pytest.approx(_OR_NAGELKERKE_R2_DELTA, abs=1e-3)
    # uniform DIF strong, nonuniform absent; R2 delta just over the JG 0.070 cutoff.
    assert res.p_uniform < 0.05
    assert res.p_nonuniform > 0.05
    assert res.dif_class == "C"


def test_conditioner_group_corr_matches_corrcoef() -> None:
    ratings, conditioner = _frozen()
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    # Independently rebuild the per-row conditioner and group vectors the engine feeds
    # into _dif_stats: one row per rating, first half of items are "ref", second half "foc".
    cond_rows: NDArray[np.float64] = np.repeat(  # type: ignore[reportUnknownMemberType]
        np.asarray(_QUALITY, dtype=float), _N_RATERS
    )
    group: NDArray[np.float64] = np.repeat(  # type: ignore[reportUnknownMemberType]
        np.array([0.0] * (_N_ITEMS // 2) + [1.0] * (_N_ITEMS // 2)), _N_RATERS
    )
    standardized_cond = (cond_rows - cond_rows.mean()) / cond_rows.std(ddof=0)
    expected = float(np.corrcoef(standardized_cond, group)[0, 1])
    assert res.conditioner_group_corr == pytest.approx(expected, abs=1e-9)


def test_engine_matches_statsmodels_logit_in_binary_limit() -> None:
    """Two-category proportional odds is ordinary logistic regression.

    The fitter's log-likelihood must equal statsmodels ``Logit`` (with an intercept,
    which is the single ordinal threshold) to floating-point precision. This validates
    the engine against the canonical logistic-regression MLE.
    """
    sm = pytest.importorskip("statsmodels.api")
    from scipy.special import expit  # type: ignore[import-untyped]

    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    yb = (rng.random(n) < expit(0.8 * x1 - 0.5 * x2)).astype(np.int_)
    design: NDArray[np.float64] = np.column_stack([x1, x2])  # type: ignore[reportUnknownMemberType]

    ll_engine, converged = _fit_proportional_odds(yb, design)
    ll_logit = float(sm.Logit(yb, sm.add_constant(design)).fit(disp=False).llf)
    assert ll_engine == pytest.approx(ll_logit, abs=1e-6)
    assert converged is True


def test_no_dif_when_groups_identical() -> None:
    # focal items mirror reference items exactly (same quality, same scores).
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4], [3, 2, 3]]
    res = logistic_dif(_make(ref, ref), focal="foc", reference="ref")
    assert res.p_uniform > 0.05
    assert res.dif_class == "A"


def test_strong_uniform_dif_flags_significant() -> None:
    # reference scores high, focal scores low at every matched quality level.
    ref = [[4, 5, 5], [5, 4, 5], [4, 4, 5], [5, 5, 4], [4, 5, 4], [5, 4, 4]]
    foc = [[1, 2, 1], [2, 1, 1], [1, 1, 2], [2, 2, 1], [1, 2, 2], [2, 1, 1]]
    res = logistic_dif(_make(ref, foc), focal="foc", reference="ref")
    assert res.p_uniform < 0.05
    assert res.dif_class in {"B", "C"}


def test_rest_score_default_equals_explicit_same_conditioner() -> None:
    ratings, _ = _frozen()
    # Leave-one-rater-out rest score, computed independently, as an explicit conditioner
    # per row would differ per rater; the default path is validated by its own source tag
    # and by producing a finite, classified result on multi-rater data.
    res = logistic_dif(ratings, focal="foc", reference="ref")
    assert res.conditioner_source == "rest_score"
    assert np.isfinite(res.chi2_total)
    assert res.dif_class in {"A", "B", "C"}
    assert res.n_obs == _N_ITEMS * _N_RATERS


def test_single_rater_without_conditioner_raises() -> None:
    ref = [[3], [4], [2], [5], [1], [3]]
    foc = [[2], [1], [3], [2], [1], [2]]
    with pytest.raises(ValueError, match="conditioner"):
        logistic_dif(_make(ref, foc), focal="foc", reference="ref")


def test_items_outside_focal_and_reference_are_ignored() -> None:
    # A third stratum's items must not enter the focal-vs-reference comparison.
    rows: list[dict[str, object]] = []
    for grp, prefix in (("ref", "a"), ("foc", "b"), ("other", "c")):
        for idx in range(4):
            for r in range(3):
                rows.append(
                    {
                        "item": f"{prefix}{idx}",
                        "rater": f"rt{r}",
                        "score": (idx + r) % 5 + 1,
                        "group": grp,
                    }
                )
    df = pd.DataFrame(rows)
    ratings = Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")
    res = logistic_dif(ratings, focal="foc", reference="ref")
    assert res.n_obs == 4 * 3 * 2  # only ref + foc rows, the 'other' stratum dropped


def test_external_conditioner_missing_item_raises() -> None:
    ratings, conditioner = _frozen()
    conditioner.pop("it00")
    with pytest.raises(ValueError, match="conditioner missing item"):
        logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)


def test_unknown_levels_raise() -> None:
    ref = [[1, 2], [3, 4], [2, 1]]
    r = _make(ref, ref)
    with pytest.raises(ValueError, match="stratum level not found"):
        logistic_dif(r, focal="nope", reference="ref")
    with pytest.raises(ValueError, match="stratum level not found"):
        logistic_dif(r, focal="foc", reference="nope")


def test_unknown_level_error_lists_available_levels() -> None:
    # Integer stratum labels become string keys; a caller passing focal=1 must see the
    # available (stringified) levels and the coercion hint, not a bare "not found".
    ref = [[1, 2], [3, 4], [2, 1]]
    r = _make(ref, ref)
    with pytest.raises(ValueError) as exc:
        logistic_dif(r, focal="1", reference="ref")
    msg = str(exc.value)
    assert "Available levels" in msg
    assert "'foc'" in msg and "'ref'" in msg
    assert "as a string" in msg


def test_jodoin_gierl_classification_boundaries() -> None:
    assert _classify_jodoin_gierl(0.02) == "A"
    assert _classify_jodoin_gierl(0.05) == "B"
    assert _classify_jodoin_gierl(0.10) == "C"
    # boundary handling (a project convention; Jodoin & Gierl state the bands, not the
    # strict-vs-closed endpoints): < 0.035 is A, [0.035, 0.070) is B, >= 0.070 is C
    assert _classify_jodoin_gierl(0.0349) == "A"
    assert _classify_jodoin_gierl(0.035) == "B"
    assert _classify_jodoin_gierl(0.070) == "C"


def test_rest_score_path_matches_pinned_oracle() -> None:
    # Pins the leave-one-rater-out default path (what report/demo call) to R polr.
    ratings, _ = _frozen()
    res = logistic_dif(ratings, focal="foc", reference="ref")
    assert res.conditioner_source == "rest_score"
    assert res.chi2_uniform == pytest.approx(_RS_CHI2_UNIFORM, abs=1e-3)
    assert res.chi2_nonuniform == pytest.approx(_RS_CHI2_NONUNIFORM, abs=1e-3)
    assert res.nagelkerke_r2_delta == pytest.approx(_RS_NAGELKERKE_R2_DELTA, abs=1e-3)
    assert res.dif_class == "A"
    assert res.converged is True


def test_constant_scores_raise() -> None:
    const = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    with pytest.raises(ValueError, match="single category"):
        logistic_dif(_make(const, const), focal="foc", reference="ref")


def test_collinear_conditioner_raises() -> None:
    # Perfect within-item agreement, group-separated: the conditioner is collinear with
    # the group, so DIF is not identifiable. Refuse rather than report a false class A.
    ref = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    foc = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    with pytest.raises(ValueError, match="collinear"):
        logistic_dif(_make(ref, foc), focal="foc", reference="ref")


def _corr_band_ratings(
    ref_cond: list[float], foc_cond: list[float]
) -> tuple[Ratings, dict[Hashable, float]]:
    """5 reference + 5 focal items with varied (multi-category) scores and the given
    per-item external conditioner values, so ``conditioner_group_corr`` is driven purely
    by ``ref_cond``/``foc_cond`` (5 items, 3 raters each, matching ``_partial_overlap_ratings``)."""
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4]]
    foc = [[3, 2, 3], [4, 3, 4], [2, 1, 2], [5, 4, 5], [1, 2, 1]]
    ratings = _make(ref, foc)
    conditioner: dict[Hashable, float] = {
        **{f"ref{i}": v for i, v in enumerate(ref_cond)},
        **{f"foc{i}": v for i, v in enumerate(foc_cond)},
    }
    return ratings, conditioner


def _expected_corr(ref_cond: list[float], foc_cond: list[float], n_raters: int = 3) -> float:
    """Independently rebuild the per-row conditioner/group vectors and correlate them."""
    ref_rows: NDArray[np.float64] = np.repeat(np.array(ref_cond, dtype=float), n_raters)  # type: ignore[reportUnknownMemberType]
    foc_rows: NDArray[np.float64] = np.repeat(np.array(foc_cond, dtype=float), n_raters)  # type: ignore[reportUnknownMemberType]
    cond: NDArray[np.float64] = np.concatenate([ref_rows, foc_rows])  # type: ignore[reportUnknownMemberType]
    group: NDArray[np.float64] = np.concatenate(  # type: ignore[reportUnknownMemberType]
        [np.zeros_like(ref_rows), np.ones_like(foc_rows)]
    )
    z = (cond - cond.mean()) / cond.std(ddof=0)
    return float(np.corrcoef(z, group)[0, 1])  # type: ignore[reportUnknownMemberType]


def test_conditioner_overlap_weak_true_in_weak_band() -> None:
    # Focal conditioner values sit mostly above reference, with overlap at 5.
    ref_cond, foc_cond = [1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 6.0, 7.0, 8.0, 9.0]
    expected = _expected_corr(ref_cond, foc_cond)
    assert 0.2 <= expected < 0.999
    ratings, conditioner = _corr_band_ratings(ref_cond, foc_cond)
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_group_corr == pytest.approx(expected, abs=1e-9)
    assert res.conditioner_overlap_weak is True


def test_conditioner_overlap_weak_true_in_calibrated_band() -> None:
    # A moderate group shift the pre-calibration 0.7 convention ignored: the S3
    # operating-characteristics study (docs/sim-operating-characteristics.md) measured
    # a 17% false B/C classification rate under H0 already at |corr| in (0.2, 0.4],
    # so the calibrated flag must fire here.
    ref_cond, foc_cond = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]
    expected = _expected_corr(ref_cond, foc_cond)
    assert 0.2 <= expected < 0.7
    ratings, conditioner = _corr_band_ratings(ref_cond, foc_cond)
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_group_corr == pytest.approx(expected, abs=1e-9)
    assert res.conditioner_overlap_weak is True


def test_conditioner_overlap_weak_false_below_band() -> None:
    # Nearly identical conditioner distributions across groups (|corr| < 0.2, the
    # calibrated safe band: 0.5% false B/C rate under H0 in the S3 study).
    ref_cond, foc_cond = [1.0, 2.0, 3.0, 4.0, 5.0], [1.25, 2.25, 3.25, 4.25, 5.25]
    expected = _expected_corr(ref_cond, foc_cond)
    assert expected < 0.2
    ratings, conditioner = _corr_band_ratings(ref_cond, foc_cond)
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_group_corr == pytest.approx(expected, abs=1e-9)
    assert res.conditioner_overlap_weak is False


def test_conditioner_overlap_weak_true_near_999_boundary_not_refused() -> None:
    # Strong separation but still identifiable (|corr| < 0.999): the engine must not
    # raise, and the advisory flag must still fire since 0.999 is an inclusive bound.
    ref_cond, foc_cond = [0.0, 1.0, 2.0, 3.0, 4.0], [40.0, 41.0, 42.0, 43.0, 44.0]
    expected = _expected_corr(ref_cond, foc_cond)
    assert 0.7 <= expected < 0.999
    ratings, conditioner = _corr_band_ratings(ref_cond, foc_cond)
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_group_corr == pytest.approx(expected, abs=1e-9)
    assert res.conditioner_overlap_weak is True


def test_conditioner_overlap_weak_raises_above_999_threshold() -> None:
    # Mirrors test_collinear_conditioner_raises: |corr| > 0.999 is still refused by the
    # existing identifiability guard, unchanged by this flag.
    ref = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    foc = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    with pytest.raises(ValueError, match="collinear"):
        logistic_dif(_make(ref, foc), focal="foc", reference="ref")


def test_constant_conditioner_raises() -> None:
    ratings, conditioner = _frozen()
    flat: dict[Hashable, float] = {item: 1.0 for item in conditioner}
    with pytest.raises(ValueError, match="no variance"):
        logistic_dif(ratings, focal="foc", reference="ref", conditioner=flat)


def test_no_dif_chi2_statistics_are_nonnegative() -> None:
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4], [3, 2, 3]]
    res = logistic_dif(_make(ref, ref), focal="foc", reference="ref")
    assert res.chi2_total >= 0.0
    assert res.chi2_uniform >= 0.0
    assert res.chi2_nonuniform >= 0.0


def test_lr_chi2_normal_positive_is_ok() -> None:
    # Full model fits better than reduced (the nested-DIF normal case): positive chi2, ok.
    value, ok = _lr_chi2(-200.0, -190.0)
    assert ok is True
    assert value == pytest.approx(20.0)


def test_lr_chi2_clamps_optimizer_noise_to_zero_but_stays_ok() -> None:
    # Full model infinitesimally worse than reduced (BFGS noise under true no-DIF):
    # clamp the tiny negative chi2 to 0 but do not flag a fit failure.
    value, ok = _lr_chi2(-200.0, -200.0 - 1e-9)
    assert ok is True
    assert value == 0.0


def test_lr_chi2_meaningful_negative_flags_not_converged() -> None:
    # Full model meaningfully worse than the nested reduced model is impossible unless the
    # optimizer failed; the bare max(0,.) clamp would mask this as a clean "no DIF" null.
    value, ok = _lr_chi2(-200.0, -205.0)
    assert ok is False
    assert value == 0.0


def test_cluster_bootstrap_returns_bracketed_ci() -> None:
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=120, seed=0
    )
    assert isinstance(res, ClusterBootstrapDif)
    assert isinstance(res.base, DifResult)
    assert res.cluster == "item"
    assert res.r2_delta_ci_low <= res.r2_delta_ci_high
    assert res.r2_delta_ci_low >= 0.0
    assert res.chi2_total_ci_low <= res.chi2_total_ci_high
    assert 0 < res.n_effective <= res.n_boot == 120


def test_cluster_bootstrap_drops_nonconverged_refits(monkeypatch: pytest.MonkeyPatch) -> None:
    ratings, cond = _frozen()
    bootstrap_calls = 0

    def fake_dif_stats(
        scores: list[float],
        groups: list[float],
        cond_rows: list[float],
        *,
        want_split: bool,
        po_alpha: float = 1e-3,
    ) -> _DifStats:
        nonlocal bootstrap_calls
        del scores, groups, cond_rows, po_alpha
        if want_split:
            return _DifStats(
                chi2_total=1.0,
                chi2_uniform=0.5,
                chi2_nonuniform=0.5,
                nagelkerke_r2_delta=0.01,
                n_obs=_N_ITEMS * _N_RATERS,
                converged=True,
                po_violation=False,
                conditioner_group_corr=0.0,
            )
        bootstrap_calls += 1
        if bootstrap_calls <= 3:
            # Bootstrap phase (n_boot=3): the first refit does not converge and is dropped.
            return _DifStats(
                chi2_total=float(bootstrap_calls),
                chi2_uniform=float("nan"),
                chi2_nonuniform=float("nan"),
                nagelkerke_r2_delta=0.01 * bootstrap_calls,
                n_obs=_N_ITEMS * _N_RATERS,
                converged=bootstrap_calls != 1,
                po_violation=False,
                conditioner_group_corr=0.0,
            )
        # Jackknife phase (leave-one-cluster-out): converged constants; equal values give
        # zero acceleration, and the boundary check drives BCa to the percentile fallback.
        return _DifStats(
            chi2_total=2.5,
            chi2_uniform=float("nan"),
            chi2_nonuniform=float("nan"),
            nagelkerke_r2_delta=0.02,
            n_obs=_N_ITEMS * _N_RATERS,
            converged=True,
            po_violation=False,
            conditioner_group_corr=0.0,
        )

    monkeypatch.setattr(dif_module, "_dif_stats", fake_dif_stats)
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=3, seed=0
    )
    assert bootstrap_calls >= 3  # 3 bootstrap refits, plus the leave-one-cluster-out jackknife
    assert res.n_effective == 2
    assert res.chi2_total_ci_low > 1.0


def test_cluster_bootstrap_is_reproducible() -> None:
    ratings, cond = _frozen()
    kw = {"focal": "foc", "reference": "ref", "conditioner": cond, "n_boot": 120, "seed": 7}
    a = cluster_bootstrap_dif(ratings, **kw)  # type: ignore[arg-type]
    b = cluster_bootstrap_dif(ratings, **kw)  # type: ignore[arg-type]
    assert a.r2_delta_ci_low == b.r2_delta_ci_low
    assert a.r2_delta_ci_high == b.r2_delta_ci_high
    assert a.chi2_total_ci_low == b.chi2_total_ci_low
    assert a.chi2_total_ci_high == b.chi2_total_ci_high


def test_cluster_bootstrap_preserves_point_estimate_and_detects_strong_dif() -> None:
    ratings, cond = _frozen()
    direct = logistic_dif(ratings, focal="foc", reference="ref", conditioner=cond)
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=150, seed=0
    )
    # the analytic point estimate is preserved unchanged as the base
    assert res.base.nagelkerke_r2_delta == pytest.approx(direct.nagelkerke_r2_delta)
    assert res.base.chi2_total == pytest.approx(direct.chi2_total)
    # the frozen fixture is strong uniform DIF: the effect-size interval stays above zero
    assert res.r2_delta_ci_low > 0.0


def test_cluster_bootstrap_is_honest_under_no_dif() -> None:
    # focal and reference drawn from the SAME score pattern: no real DIF. The bootstrap
    # must not manufacture an effect; its effect-size lower bound stays in the class-A band.
    pattern = [
        [1, 2, 3],
        [3, 4, 2],
        [5, 4, 5],
        [2, 1, 2],
        [4, 5, 4],
        [3, 2, 3],
        [2, 3, 1],
        [4, 3, 5],
    ]
    ratings = _make(pattern, pattern)
    cond: dict[Hashable, float] = {f"ref{i}": float(i % 4) for i in range(len(pattern))}
    cond.update({f"foc{i}": float(i % 4) for i in range(len(pattern))})
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=150, seed=0
    )
    assert res.r2_delta_ci_low < 0.035


def test_cluster_bootstrap_rest_score_path() -> None:
    # No external conditioner: each resample rebuilds its own leave-one-rater-out rest
    # score. Moderate, non-collinear DIF so the engine does not refuse.
    rows: list[dict[str, object]] = []
    groups = ["foc", "ref"]
    for i in range(16):
        grp = groups[i % 2]
        for r in range(3):
            rows.append({"item": f"i{i}", "rater": f"r{r}", "score": (i + r) % 5, "group": grp})
    df = pd.DataFrame(rows)
    ratings = Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")
    res = cluster_bootstrap_dif(ratings, focal="foc", reference="ref", n_boot=120, seed=0)
    assert res.base.conditioner_source == "rest_score"
    assert res.n_effective > 0
    assert res.r2_delta_ci_low <= res.r2_delta_ci_high


def test_cluster_bootstrap_zero_boot_returns_nan_ci() -> None:
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(ratings, focal="foc", reference="ref", conditioner=cond, n_boot=0)
    assert res.n_effective == 0
    assert math.isnan(res.r2_delta_ci_low)
    assert math.isnan(res.r2_delta_ci_high)
    assert math.isnan(res.chi2_total_ci_low)
    assert math.isnan(res.chi2_total_ci_high)


def test_cluster_bootstrap_ci_reliable_tracks_effective_count() -> None:
    ratings, cond = _frozen()
    # 200 resamples all survive on the frozen fixture: above the reliability floor.
    ok = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=200, seed=0
    )
    assert ok.n_effective >= 100
    assert ok.ci_reliable is True
    # zero / very thin resampling: the percentile CI is not trustworthy.
    none = cluster_bootstrap_dif(ratings, focal="foc", reference="ref", conditioner=cond, n_boot=0)
    assert none.ci_reliable is False
    thin = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=5, seed=0
    )
    assert thin.n_effective < 100
    assert thin.ci_reliable is False


def test_cluster_bootstrap_ci_level_is_configurable_and_recorded() -> None:
    ratings, cond = _frozen()
    narrow = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=300, seed=0, ci=0.90
    )
    wide = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=300, seed=0, ci=0.99
    )
    # the chosen level is recorded for honest, self-describing provenance
    assert narrow.ci_level == 0.90
    assert wide.ci_level == 0.99
    # default stays the 95% interval
    default = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=300, seed=0
    )
    assert default.ci_level == 0.95
    # a lower confidence level yields a narrower effect-size interval
    width_90 = narrow.r2_delta_ci_high - narrow.r2_delta_ci_low
    width_99 = wide.r2_delta_ci_high - wide.r2_delta_ci_low
    assert width_90 < width_99


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_cluster_bootstrap_invalid_ci_raises(bad: float) -> None:
    ratings, cond = _frozen()
    with pytest.raises(ValueError, match="ci must be in"):
        cluster_bootstrap_dif(
            ratings, focal="foc", reference="ref", conditioner=cond, n_boot=10, ci=bad
        )


def test_cluster_bootstrap_bounds_are_stable() -> None:
    # Characterization lock for the engine: the per-resample DIF statistics are invariant to
    # observation order and do not need the ll2 (uniform/nonuniform) fit. The frozen fixture
    # has DIF away from the 0 boundary, so the CI is the bias-corrected accelerated (BCa)
    # interval (validated against scipy in test_bca_bounds_match_scipy_oracle). The tolerance
    # is the engine's optimizer-noise scale (_LR_NOISE_TOL, 1e-6): a BFGS-derived statistic is
    # not bitwise reproducible across platforms/BLAS, but a real regression moves these by
    # orders of magnitude, not parts per million.
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=200, seed=0
    )
    assert res.n_effective == 200
    assert res.ci_method == "bca"
    assert res.r2_delta_ci_low == pytest.approx(0.031220102947280143, abs=1e-6)
    assert res.r2_delta_ci_high == pytest.approx(0.13546044457360618, abs=1e-6)
    assert res.chi2_total_ci_low == pytest.approx(8.096802268528158, abs=1e-6)
    assert res.chi2_total_ci_high == pytest.approx(35.86937198633991, abs=1e-6)


def test_logistic_dif_po_violation_responds_to_alpha() -> None:
    """The flag reflects the Brant computation and the alpha knob, not a constant.

    po_alpha=1.0 flags any fitted model (omnibus_p < 1.0); po_alpha=0.0 never flags
    (no p-value is below 0). A flag that moved with neither would be hardcoded.
    """
    import numpy as np
    import pandas as pd

    from metajudge.data import Ratings
    from metajudge.dif import logistic_dif

    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for item in range(120):
        stratum = "focal" if item % 2 == 0 else "reference"
        q = rng.standard_normal()
        for rater in range(3):
            p_lo = 1.0 / (1.0 + np.exp(-(1.2 - 0.1 * q)))
            p_hi = 1.0 / (1.0 + np.exp(-(-1.1 - 2.0 * q)))
            u = rng.random()
            score = 1 + int(u > p_lo) + int(u > p_hi)
            rows.append({"item": item, "rater": rater, "score": score, "stratum": stratum})
    ratings = Ratings.from_long(
        pd.DataFrame(rows), item="item", rater="rater", score="score", stratum="stratum"
    )

    strict = logistic_dif(ratings, focal="focal", reference="reference", po_alpha=1.0)
    never = logistic_dif(ratings, focal="focal", reference="reference", po_alpha=0.0)
    assert isinstance(strict.po_violation, bool)
    assert strict.po_violation is True
    assert never.po_violation is False


def test_brant_test_m_lt_2_early_return_with_binary_scores() -> None:
    """Binary ordinal (2 levels → m=1 < 2) triggers the _brant_test early-return path.

    With m < 2 the Brant test is inapplicable; the implementation returns converged=True
    and p=1.0, so po_violation is always False regardless of po_alpha.
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for item in range(40):
        stratum = "foc" if item < 20 else "ref"
        for rater in range(3):
            rows.append(
                {"item": item, "rater": rater, "score": int(rng.random() > 0.5), "stratum": stratum}
            )
    ratings = Ratings.from_long(
        pd.DataFrame(rows), item="item", rater="rater", score="score", stratum="stratum"
    )
    # po_alpha=1.0 would flag any real Brant result; binary data short-circuits before that.
    result = logistic_dif(ratings, focal="foc", reference="ref", po_alpha=1.0)
    assert result.po_violation is False


def test_brant_test_uses_default_names_when_none_passed() -> None:
    """Calling _brant_test with names=None exercises the default-names branch."""
    from metajudge.dif import _brant_test  # pyright: ignore[reportPrivateUsage]

    rng = np.random.default_rng(0)
    endog = rng.integers(0, 3, size=60)
    exog = rng.standard_normal((60, 2))
    result = _brant_test(endog, exog)  # names=None triggers the default x0..x{p-1} path
    assert "x0" in result.per_predictor
    assert "x1" in result.per_predictor


def test_dif_stats_brant_runtime_error_sets_po_violation_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeError from _brant_test is caught; po_violation falls back to False."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced from test")

    monkeypatch.setattr(dif_module, "_brant_test", _raise)
    ratings, _ = _frozen()
    result = logistic_dif(ratings, focal="foc", reference="ref")
    assert result.po_violation is False


def test_bootstrap_value_error_resamples_give_nan_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """ValueError inside the bootstrap try-block is caught; all resamples drop → NaN CI."""
    original = dif_module._dif_stats  # pyright: ignore[reportPrivateUsage]

    def _fail_in_bootstrap(*args: object, want_split: bool, **kwargs: object) -> object:
        if not want_split:
            raise ValueError("forced bootstrap failure")
        return original(*args, want_split=want_split, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dif_module, "_dif_stats", _fail_in_bootstrap)
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=3, seed=0
    )
    assert res.n_effective == 0
    assert math.isnan(res.r2_delta_ci_low)
    assert math.isnan(res.chi2_total_ci_low)


def test_bca_bounds_match_scipy_oracle() -> None:
    # Oracle: scipy.stats.bootstrap(method="BCa"). Feed _bca_bounds the exact replicate
    # distribution scipy drew and the leave-one-out jackknife; the BCa math (bias correction
    # z0 + jackknife acceleration + adjusted percentiles) must reproduce scipy's interval.
    from scipy.stats import bootstrap  # type: ignore[import-untyped]

    rng = np.random.default_rng(0)
    data: NDArray[np.float64] = rng.gamma(2.0, 1.0, size=40)  # skewed, so BCa != percentile
    # random_state via **kwargs: the installed scipy accepts it at runtime, but its type stub
    # only names the newer `rng` alias, so a literal kwarg trips reportCallIssue.
    boot_kwargs = {
        "method": "BCa",
        "n_resamples": 3000,
        "random_state": 7,
        "confidence_level": 0.90,
    }
    res = bootstrap((data,), np.mean, **boot_kwargs)  # type: ignore[reportUnknownVariableType]
    theta_hat = float(data.mean())
    jack: NDArray[np.float64] = np.array(  # type: ignore[reportUnknownMemberType]
        [float(np.delete(data, i).mean()) for i in range(int(data.size))]  # type: ignore[reportUnknownArgumentType]
    )
    boot_dist: NDArray[np.float64] = np.asarray(res.bootstrap_distribution, dtype=float)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    lo, hi, ok = _bca_bounds(boot_dist, theta_hat, jack, 5.0, 95.0)
    assert ok is True
    assert lo == pytest.approx(float(res.confidence_interval.low), abs=1e-9)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert hi == pytest.approx(float(res.confidence_interval.high), abs=1e-9)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_bca_bounds_reduce_to_percentile_without_bias_or_acceleration() -> None:
    # Exactly half the replicates below theta_hat (z0=0) and constant jackknife (a=0) =>
    # BCa collapses to the plain percentile interval.
    half: NDArray[np.float64] = np.linspace(0.001, 1.0, 500)
    boot: NDArray[np.float64] = np.concatenate([-half, half])  # type: ignore[reportUnknownMemberType]
    jack: NDArray[np.float64] = np.zeros(10)
    lo, hi, ok = _bca_bounds(boot, 0.0, jack, 2.5, 97.5)
    assert ok is True
    pctl: NDArray[np.float64] = np.percentile(boot, [2.5, 97.5])  # type: ignore[reportUnknownMemberType]
    assert lo == pytest.approx(float(pctl[0]), abs=1e-9)
    assert hi == pytest.approx(float(pctl[1]), abs=1e-9)


def test_bca_bounds_flag_degenerate_boundary_statistic() -> None:
    # All replicates below theta_hat (a 0-bounded statistic piled at the boundary) makes the
    # bias correction infinite; _bca_bounds signals ok=False so the caller uses percentile.
    boot: NDArray[np.float64] = np.zeros(500)
    jack: NDArray[np.float64] = np.zeros(10)
    lo, hi, ok = _bca_bounds(boot, 0.5, jack, 2.5, 97.5)
    assert ok is False
    assert math.isnan(lo) and math.isnan(hi)


def test_cluster_bootstrap_reports_ci_method() -> None:
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=200, seed=0
    )
    assert res.ci_method in {"bca", "percentile"}
    assert res.r2_delta_ci_low <= res.r2_delta_ci_high


def test_cluster_bootstrap_skips_bca_when_panel_exceeds_n_boot() -> None:
    # 24 item clusters with n_boot=10: the leave-one-cluster-out jackknife would add more
    # fits than the bootstrap, so BCa is gated off and the percentile CI is used.
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=10, seed=0
    )
    assert res.ci_method == "percentile"


def test_holm_adjust_matches_statsmodels_oracle() -> None:
    mt = pytest.importorskip("statsmodels.stats.multitest")
    pvals = [0.001, 0.013, 0.021, 0.04, 0.6, 0.99]
    expected = list(mt.multipletests(pvals, method="holm")[1])
    got = holm_adjust(pvals)
    assert got == pytest.approx(expected, abs=1e-12)


def test_holm_adjust_unsorted_input_matches_statsmodels() -> None:
    mt = pytest.importorskip("statsmodels.stats.multitest")
    pvals = [0.04, 0.001, 0.6, 0.021, 0.99, 0.013]  # deliberately unordered
    expected = list(mt.multipletests(pvals, method="holm")[1])
    assert holm_adjust(pvals) == pytest.approx(expected, abs=1e-12)


def test_holm_adjust_edge_cases() -> None:
    assert holm_adjust([]) == []
    assert holm_adjust([0.3]) == pytest.approx([0.3])
    # Every adjusted p is capped at 1.0.
    assert all(p <= 1.0 for p in holm_adjust([0.9, 0.95, 0.99]))


# --- multi-stratum sweep ---------------------------------------------------


def _three_stratum() -> tuple[Ratings, dict[Hashable, float]]:
    """Three strata (A/B/C), 9 items each, 4 raters, with a varying external conditioner.

    Group B carries a uniform downward shift so at least one pair shows DIF and the Holm
    family spans significant and non-significant p-values.
    """
    rng = np.random.default_rng(20260701)
    rows: list[dict[str, object]] = []
    conditioner: dict[Hashable, float] = {}
    shifts = {"A": 0.0, "B": -1.4, "C": 0.05}
    for group, shift in shifts.items():
        for i in range(9):
            item = f"{group}{i:02d}"
            quality = float(rng.normal())
            conditioner[item] = quality
            latent = quality + shift
            for r in range(4):
                score = min(5, max(1, round(3.0 + latent + float(rng.normal(scale=0.6)))))
                rows.append({"item": item, "rater": f"rt{r}", "score": score, "group": group})
    df = pd.DataFrame(rows)
    ratings = Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")
    return ratings, conditioner


def test_sweep_holm_matches_statsmodels_oracle() -> None:
    mt = pytest.importorskip("statsmodels.stats.multitest")
    ratings, cond = _three_stratum()
    pairs = [("B", "A"), ("C", "A"), ("B", "C")]
    result = sweep(ratings, pairs=pairs, conditioner=cond)
    assert isinstance(result, DifSweep)

    raw_total = [r.p_total for r in result.results]
    raw_uniform = [r.p_uniform for r in result.results]
    raw_nonuniform = [r.p_nonuniform for r in result.results]
    assert result.p_total_holm == pytest.approx(
        list(mt.multipletests(raw_total, method="holm")[1]), abs=1e-12
    )
    assert result.p_uniform_holm == pytest.approx(
        list(mt.multipletests(raw_uniform, method="holm")[1]), abs=1e-12
    )
    assert result.p_nonuniform_holm == pytest.approx(
        list(mt.multipletests(raw_nonuniform, method="holm")[1]), abs=1e-12
    )


def test_sweep_results_match_individual_logistic_dif() -> None:
    ratings, cond = _three_stratum()
    pairs = [("B", "A"), ("C", "A")]
    result = sweep(ratings, pairs=pairs, conditioner=cond)
    assert result.pairs == pairs
    for (focal, reference), res in zip(pairs, result.results, strict=True):
        solo = logistic_dif(ratings, focal=focal, reference=reference, conditioner=cond)
        assert res.focal_level == focal
        assert res.reference_level == reference
        assert res.chi2_total == pytest.approx(solo.chi2_total, abs=1e-9)
        assert res.p_total == pytest.approx(solo.p_total, abs=1e-12)


def test_sweep_adjusted_p_never_below_raw() -> None:
    ratings, cond = _three_stratum()
    result = sweep(ratings, pairs=[("B", "A"), ("C", "A"), ("B", "C")], conditioner=cond)
    for raw, adj in zip((r.p_total for r in result.results), result.p_total_holm, strict=True):
        assert adj >= raw - 1e-12


def test_sweep_requires_at_least_one_pair() -> None:
    ratings, cond = _three_stratum()
    with pytest.raises(ValueError, match="at least one"):
        sweep(ratings, pairs=[], conditioner=cond)


def _patch_brant_p(monkeypatch: pytest.MonkeyPatch, p: float) -> None:
    """Force the Brant PO test to return a fixed omnibus p so po_alpha can be straddled."""
    from metajudge.dif import _BrantResult  # pyright: ignore[reportPrivateUsage]

    def fake(*_args: object, **_kwargs: object) -> _BrantResult:
        return _BrantResult(
            omnibus_chi2=1.0, omnibus_df=1, omnibus_p=p, per_predictor={}, converged=True
        )

    monkeypatch.setattr("metajudge.dif._brant_test", fake)


def test_audit_forwards_po_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    from metajudge.report import audit

    ratings, cond = _three_stratum()
    _patch_brant_p(monkeypatch, 0.5)
    # po_violation is (brant.converged and brant.omnibus_p < po_alpha); 0.5 straddles 0.4/0.6.
    clear = audit(ratings, focal="B", reference="A", conditioner=cond, po_alpha=0.4)
    tripped = audit(ratings, focal="B", reference="A", conditioner=cond, po_alpha=0.6)
    assert clear.dif.po_violation is False
    assert tripped.dif.po_violation is True


def test_cluster_bootstrap_forwards_po_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    ratings, cond = _three_stratum()
    _patch_brant_p(monkeypatch, 0.5)
    tripped = cluster_bootstrap_dif(
        ratings, focal="B", reference="A", conditioner=cond, n_boot=5, po_alpha=0.6
    )
    clear = cluster_bootstrap_dif(
        ratings, focal="B", reference="A", conditioner=cond, n_boot=5, po_alpha=0.4
    )
    assert tripped.base.po_violation is True
    assert clear.base.po_violation is False


# --- conditioner_common_support: item-level overlap diagnostic ---
#
# Definition: with F = focal per-item conditioner values and R = reference per-item
# conditioner values, [lo, hi] = [max(min(F), min(R)), min(max(F), max(R))]; support is
# the fraction of values (across both F and R) landing inside [lo, hi]. Disjoint ranges
# (lo > hi) give 0.0; identical ranges give 1.0.


def test_common_support_helper_partial_overlap_hand_computed() -> None:
    # F = [3,4,5,6,7], R = [1,2,3,4,5]. Overlap [lo,hi] = [max(3,1), min(7,5)] = [3,5].
    # F in range: 3,4,5 (3 values). R in range: 3,4,5 (3 values). (3+3)/(5+5) = 0.6.
    focal_vals: NDArray[np.float64] = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    reference_vals: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _common_support(focal_vals, reference_vals) == pytest.approx(0.6, abs=1e-12)


def test_common_support_helper_disjoint_is_zero() -> None:
    # F = [10..14], R = [1..5]: lo = max(10,1) = 10, hi = min(14,5) = 5. lo > hi -> 0.0.
    focal_vals: NDArray[np.float64] = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    reference_vals: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _common_support(focal_vals, reference_vals) == 0.0


def test_common_support_helper_identical_range_is_one() -> None:
    # F == R -> every value falls inside [lo, hi] = [min, max] -> 1.0.
    focal_vals: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reference_vals: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _common_support(focal_vals, reference_vals) == pytest.approx(1.0, abs=1e-12)


def test_common_support_helper_empty_side_is_nan() -> None:
    empty: NDArray[np.float64] = np.array([], dtype=np.float64)
    non_empty: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
    assert math.isnan(_common_support(empty, non_empty))
    assert math.isnan(_common_support(non_empty, empty))
    assert math.isnan(_common_support(empty, empty))


def _partial_overlap_ratings() -> tuple[Ratings, dict[Hashable, float]]:
    """5 reference + 5 focal items, external per-item conditioner values that partly
    overlap by construction: R = [1,2,3,4,5], F = [3,4,5,6,7]."""
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4]]
    foc = [[3, 2, 3], [4, 3, 4], [2, 1, 2], [5, 4, 5], [1, 2, 1]]
    ratings = _make(ref, foc)
    conditioner: dict[Hashable, float] = {
        "ref0": 1.0,
        "ref1": 2.0,
        "ref2": 3.0,
        "ref3": 4.0,
        "ref4": 5.0,
        "foc0": 3.0,
        "foc1": 4.0,
        "foc2": 5.0,
        "foc3": 6.0,
        "foc4": 7.0,
    }
    return ratings, conditioner


def test_conditioner_common_support_partial_overlap_hand_computed() -> None:
    ratings, conditioner = _partial_overlap_ratings()
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    # Same hand computation as the helper test above: (3+3)/(5+5) = 0.6.
    assert res.conditioner_common_support == pytest.approx(0.6, abs=1e-12)


def test_conditioner_common_support_disjoint_ranges_is_zero() -> None:
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4]]
    foc = [[3, 2, 3], [4, 3, 4], [2, 1, 2], [5, 4, 5], [1, 2, 1]]
    ratings = _make(ref, foc)
    conditioner: dict[Hashable, float] = {
        "ref0": 1.0,
        "ref1": 2.0,
        "ref2": 3.0,
        "ref3": 4.0,
        "ref4": 5.0,
        "foc0": 10.0,
        "foc1": 11.0,
        "foc2": 12.0,
        "foc3": 13.0,
        "foc4": 14.0,
    }
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_common_support == 0.0


def test_conditioner_common_support_identical_range_is_one() -> None:
    ref = [[1, 2, 2], [3, 3, 4], [5, 4, 5], [2, 1, 2], [4, 5, 4]]
    foc = [[3, 2, 3], [4, 3, 4], [2, 1, 2], [5, 4, 5], [1, 2, 1]]
    ratings = _make(ref, foc)
    conditioner: dict[Hashable, float] = {
        "ref0": 1.0,
        "ref1": 2.0,
        "ref2": 3.0,
        "ref3": 4.0,
        "ref4": 5.0,
        "foc0": 1.0,
        "foc1": 2.0,
        "foc2": 3.0,
        "foc3": 4.0,
        "foc4": 5.0,
    }
    res = logistic_dif(ratings, focal="foc", reference="ref", conditioner=conditioner)
    assert res.conditioner_common_support == pytest.approx(1.0, abs=1e-12)


def test_conditioner_common_support_rest_score_representative_is_item_mean() -> None:
    # Rest-score path (no external conditioner): the per-item representative is the item mean.
    # Reference item means [1, 2, 3], focal item means [3, 4, 5]; the overlapping range is
    # [3, 3], which holds exactly one value per side, so common_support = (1 + 1) / 6. This
    # pins the rest-score branch of the representative-value selection, not just the external
    # branch the other fixtures cover.
    ratings = _make([[1, 1], [2, 2], [3, 3]], [[3, 3], [4, 4], [5, 5]])
    res = logistic_dif(ratings, focal="foc", reference="ref")
    assert res.conditioner_source == "rest_score"
    assert res.conditioner_common_support == pytest.approx(2 / 6, abs=1e-12)
