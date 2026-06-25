import numpy as np
from sim.dgp import DgpParams, simulate


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
