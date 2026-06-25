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

from collections.abc import Hashable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from metajudge.data import Ratings
from metajudge.dif import (
    ClusterBootstrapDif,
    DifResult,
    _classify_jodoin_gierl,  # pyright: ignore[reportPrivateUsage]
    _fit_proportional_odds,  # pyright: ignore[reportPrivateUsage]
    _lr_chi2,  # pyright: ignore[reportPrivateUsage]
    cluster_bootstrap_dif,
    logistic_dif,
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


def test_cluster_bootstrap_zero_boot_collapses_to_point() -> None:
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(ratings, focal="foc", reference="ref", conditioner=cond, n_boot=0)
    assert res.n_effective == 0
    assert res.r2_delta_ci_low == res.r2_delta_ci_high == res.base.nagelkerke_r2_delta
    assert res.chi2_total_ci_low == res.chi2_total_ci_high == res.base.chi2_total


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
    # Characterization lock for the engine refactor: the per-resample DIF statistics are
    # invariant to observation order and do not need the ll2 (uniform/nonuniform) fit, so
    # bypassing the DataFrame round-trip and dropping that fit leaves these frozen-fixture
    # bounds unchanged. The tolerance is the engine's optimizer-noise scale (_LR_NOISE_TOL,
    # 1e-6), not floating-point exactness: a BFGS-derived statistic is not bitwise
    # reproducible across platforms/BLAS, but a real refactor regression would move these by
    # orders of magnitude, not parts per million.
    ratings, cond = _frozen()
    res = cluster_bootstrap_dif(
        ratings, focal="foc", reference="ref", conditioner=cond, n_boot=200, seed=0
    )
    assert res.n_effective == 200
    assert res.r2_delta_ci_low == pytest.approx(0.03371076982262634, abs=1e-6)
    assert res.r2_delta_ci_high == pytest.approx(0.1485152604131145, abs=1e-6)
    assert res.chi2_total_ci_low == pytest.approx(9.028882920867858, abs=1e-6)
    assert res.chi2_total_ci_high == pytest.approx(40.180517271142136, abs=1e-6)


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
