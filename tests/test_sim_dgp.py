import numpy as np
import pytest
from sim.dgp import FOCAL, REFERENCE, DgpParams, simulate

from metajudge import brant_test


def _analytic_category_probs(thresholds: tuple[float, ...]) -> np.ndarray:  # type: ignore[type-arg]
    # At eta = 0 the cumulative-logit category probs come straight from the thresholds.
    p_gt = 1.0 / (1.0 + np.exp(-np.asarray(thresholds, dtype=float)))  # P(X > level_k)
    upper = np.concatenate([[1.0], p_gt])  # P(X > level_{k-1})
    lower = np.concatenate([p_gt, [0.0]])  # P(X > level_k)
    return upper - lower  # type: ignore[return-value]


def test_thresholds_recover_marginal_category_probs() -> None:
    # trait_slope = 0, rater_sd = 0, no impact, no DIF: every observation has eta = 0,
    # so empirical category frequencies must converge to the analytic threshold probs.
    params = DgpParams(
        n_items_per_group=4000,
        n_raters=3,
        trait_slope=0.0,
        rater_sd=0.0,
        mu_focal=0.0,
    )
    sample = simulate(params, seed=20260624)
    scores = sample.ratings._long["score"].to_numpy()  # (test-only inspection of private attr)
    counts = np.bincount(scores, minlength=params.n_categories + 1)[1:]
    empirical = counts / counts.sum()
    analytic = _analytic_category_probs(params.thresholds)
    assert np.allclose(empirical, analytic, atol=0.01)


def _mean_score_by_item(sample) -> dict[str, float]:
    long = sample.ratings._long
    return long.groupby("item")["score"].mean().to_dict()


def test_higher_trait_gives_higher_scores() -> None:
    params = DgpParams(n_items_per_group=1500, n_raters=3, trait_slope=1.0, rater_sd=0.0)
    sample = simulate(params, seed=11)
    means = _mean_score_by_item(sample)
    theta = sample.theta
    items = list(theta)
    order = np.argsort([theta[i] for i in items])
    ranked = [items[k] for k in order]
    third = len(ranked) // 3
    low = np.mean([means[i] for i in ranked[:third]])
    high = np.mean([means[i] for i in ranked[-third:]])
    assert high > low + 0.5  # higher latent trait -> higher mean rating


def test_uniform_dif_shifts_focal_scores_at_matched_trait() -> None:
    # No impact (mu_focal = 0): focal and reference share the trait distribution, so any
    # mean-score gap is the planted uniform DIF, not impact.
    params = DgpParams(
        n_items_per_group=2000, n_raters=3, trait_slope=1.0, rater_sd=0.0, dif_uniform=0.8
    )
    sample = simulate(params, seed=7)
    long = sample.ratings._long
    ref_mean = long[long["stratum"] == REFERENCE]["score"].mean()
    foc_mean = long[long["stratum"] == FOCAL]["score"].mean()
    assert foc_mean > ref_mean + 0.2  # positive b2 lifts focal ratings


def test_seed_is_reproducible_and_varies() -> None:
    params = DgpParams(n_items_per_group=50, n_raters=3)
    a = simulate(params, seed=99)
    b = simulate(params, seed=99)
    c = simulate(params, seed=100)
    assert a.ratings.wide().equals(b.ratings.wide())
    assert a.theta == b.theta
    assert not a.ratings.wide().equals(c.ratings.wide())


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="thresholds must have"):
        simulate(DgpParams(n_items_per_group=10, n_raters=2, thresholds=(1.0, 0.0)), seed=1)
    with pytest.raises(ValueError, match="strictly decreasing"):
        simulate(
            DgpParams(n_items_per_group=10, n_raters=2, thresholds=(0.0, 0.0, 0.0, 0.0)), seed=1
        )
    with pytest.raises(ValueError, match="reliability"):
        simulate(
            DgpParams(n_items_per_group=10, n_raters=2, conditioner_reliability=0.0), seed=1
        )
    with pytest.raises(ValueError, match="n_raters"):
        simulate(DgpParams(n_items_per_group=10, n_raters=0), seed=1)


def _endog_exog(sample):
    long = sample.ratings._long
    endog = long["score"].to_numpy().astype(int)
    theta = sample.theta
    trait = long["item"].map(theta).to_numpy().astype(float)
    return endog, trait.reshape(-1, 1)


def test_po_holds_cell_does_not_trip_brant() -> None:
    params = DgpParams(n_items_per_group=800, n_raters=3, trait_slope=1.0, rater_sd=0.0)
    sample = simulate(params, seed=3)
    endog, exog = _endog_exog(sample)
    result = brant_test(endog, exog, names=["trait"])
    assert result.omnibus_p > 0.01  # proportional odds holds by construction


def test_po_violation_cell_trips_brant() -> None:
    # A strong across-cutpoint slope spread breaks proportional odds on the trait predictor.
    params = DgpParams(
        n_items_per_group=800, n_raters=3, trait_slope=1.0, rater_sd=0.0, po_violation=0.8
    )
    sample = simulate(params, seed=3)
    endog, exog = _endog_exog(sample)
    result = brant_test(endog, exog, names=["trait"])
    assert result.omnibus_p < 1e-3  # strict large-N threshold (synthesis section 3)


def test_nonuniform_dif_steepens_focal_trait_slope() -> None:
    # b3 > 0 makes the focal group's score-vs-trait slope steeper than the reference group's.
    params = DgpParams(
        n_items_per_group=2000,
        n_raters=3,
        trait_slope=1.0,
        rater_sd=0.0,
        dif_nonuniform=0.8,
    )
    sample = simulate(params, seed=5)
    long = sample.ratings._long
    theta = sample.theta
    long = long.assign(trait=long["item"].map(theta))

    def _slope(frame) -> float:
        return float(np.polyfit(frame["trait"].to_numpy(), frame["score"].to_numpy(), 1)[0])

    ref_slope = _slope(long[long["stratum"] == REFERENCE])
    foc_slope = _slope(long[long["stratum"] == FOCAL])
    assert foc_slope > ref_slope + 0.1
