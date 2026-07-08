# E07 DIF: empirical derivation of the ordinal-PO Jodoin-Gierl A/B/C bands

Date: 2026-07-05 Status: accepted (constants updated 2026-07-05; see "Resolution" below)

## Context

`docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md` ships the Jodoin and Gierl (2001) A/B/C thresholds (0.035 negligible/moderate, 0.070 moderate/large) directly on the ordinal proportional-odds (PO) Nagelkerke R-squared change, and says so plainly: Jodoin and Gierl validated these cutoffs on the two-category logistic R-squared, so applying them to the ordinal PO change is "an applied convention... not a direct result" of the original validation, and "no published simulation has validated the 0.035 and 0.070 bands for the ordinal proportional-odds R-squared change specifically." That ADR named deriving the ordinal bands by Monte Carlo as a separate research task. This is that task, executed (`scripts/derive_ordinal_bands.py`, seed `BAND_SEED = 20260704`, raw draws in `sim/results/band_derivation_points.csv` and `band_derivation_summary.csv`).

## Method

1. Hold an interpretable DIF magnitude fixed: the uniform-DIF coefficient `b2`, a constant shift in the cumulative logit between focal and reference groups. It is a log odds-ratio (`OR = exp(b2)`) with the same meaning for a 2-category logistic outcome and a 5-category PO outcome, so it transports cleanly between them.
2. Reproduce the dichotomous regime Jodoin and Gierl calibrated on (`n_categories=2`, where the shipped PO fit reduces to ordinary logistic regression) and find the `b2` where the dichotomous mean R-squared change crosses 0.035 and 0.070. Those `b2` values are the true DIF magnitudes the shipped bands encode.
3. Read the ordinal (5-category) mean R-squared change at those same anchor `b2` values. That is the derived ordinal-PO band.

Both curves use the shipped engine directly (`metajudge.dif._dif_stats`, `_fit_proportional_odds`, `_nagelkerke`) on the shared cumulative-logit DGP (`sim.dgp`), at the SummEval demo's scale (n_obs = 4,800: 800 items per stratum, 3 raters). 200 replications per (metric, `b2`) point across a 13-point grid; per-replication PO fits that fail to converge are dropped (see `n_converged` in the raw draws). Nothing about the numerics is reimplemented; only the `b2` sweep and the crossing-inversion are new code.

## Result

| Anchor | `b2` | Odds ratio | Shipped (dichotomous-calibrated) | Derived (ordinal-PO) |
| --- | --- | --- | --- | --- |
| A/B | 0.807 | 2.24 | 0.035 | **0.0376** (MC SE 0.00039) |
| B/C | 1.175 | 3.24 | 0.070 | **0.0757** (MC SE 0.00054) |

The MC SE above is the within-run scatter at 200 reps; it is not the full uncertainty, because the band also drifts with the replication count (see Limits).

The derived ordinal bands are 7-8% higher than the shipped dichotomous-calibrated values, not lower. That direction matters: at a fixed shipped threshold, an ordinal DIF effect needs slightly _less_ true magnitude to cross into B or C than the dichotomous calibration implies, so the shipped 0.035/0.070 read very slightly liberal (a hair too easy to trip) relative to the derived ordinal bands, not conservative.

For reference, converting the field's other candidate cutoffs to the same ordinal-magnitude axis: lordif's own default (0.02, Choi, Gibbons and Crane 2011) corresponds to `b2 = 0.587` (OR 1.80) on this DGP, well below the ordinal-equivalent of the shipped 0.035 (`b2 = 0.777`, OR 2.17). The full curve (13 `b2` points, both metrics) is in `sim/results/band_derivation_points.csv`.

**The SummEval demo's headline class-A result is stable under every candidate threshold.** Its observed ordinal R-squared change is 0.002. Margin below each candidate A/B boundary: 10x (lordif 0.02), 17.5x (shipped 0.035), 18.8x (derived 0.0376). All three classify it class A. The miscalibration this study measures does not change any number this repo has already reported.

## Resolution (2026-07-05): constants moved to the derived values

`_JG_NEGLIGIBLE` and `_JG_LARGE` in `src/metajudge/dif.py` are updated from the shipped dichotomous-calibrated 0.035/0.070 to the derived ordinal-PO **0.0376/0.0757**. Rationale for deciding now rather than deferring:

1. The derived values are a direct empirical fit for the statistic this repo actually computes (ordinal PO Nagelkerke R-squared change), not an inherited convention from a different regime (dichotomous logistic R-squared). Once the miscalibration is measured, keeping the less-accurate constant is not a neutral default; it is the choice to keep a documented liberal bias.
2. `2026-07-02-e07-overlap-threshold-calibration.md` already set the precedent of moving a shipped constant from stated-convention to measured value on this repo (a larger, 3.5x move). This is a smaller instance of the same policy.
3. Timing: this repo is pre-publish (`dev` unmerged, nothing tagged on `main` yet), so the stability cost the original decision worried about (a future "class B" not matching a past tagged "class B" at the margin) is close to zero right now. Moving the constant after publishing and tagging results would manufacture that exact comparability problem; moving it now is the cheapest point in the project's history to do it.
4. No reported result changes classification: the SummEval demo's headline class-A result (R-squared change 0.002) stays class A under both the old and new thresholds (17.5x vs. 18.8x margin below the respective A/B boundary).

One pinned test fixture's expected class did change: `test_matches_pinned_oracle_with_external_conditioner` (`tests/test_dif.py`) has R-squared change 0.074462, which sits between the old C cutoff (0.070) and the new one (0.0757); its expected `dif_class` moved from `"C"` to `"B"`. This is exactly the margin case anticipated in the original decision text, not a numerical bug. `_classify_jodoin_gierl`'s boundary test and two `sim` validation tests were updated to the new threshold values; no simulation or oracle logic changed.

## Limits

- One DGP family (cumulative-logit PO, matching `sim/dgp.py`), one scale (n_obs = 4,800, 800/stratum, 3 raters), one seed. Not varied: rater count, item count, category count other than 2-vs-5, nonuniform DIF, or a degraded external conditioner.
- The anchors are single crossing points (A/B, B/C) on a 13-point grid with linear interpolation; the grid was chosen dense near both crossings (`OR` roughly 2 and 2.7) but is not adaptively refined.
- The shipped constants are the `--reps 200` run (about 90 minutes), now the script default, fully reproducible from `BAND_SEED = 20260704`.
- The derived bands have not converged in the replication count. The B/C band is 0.0757 at 200 reps and 0.0815 at 400, and the run is deterministic under `BAND_SEED`, so this is bias, not sampling scatter. The band is a nonlinear crossing-inversion of the simulated R-squared curve, and the average of that transform over a noisy curve is biased away from the noiseless value; the bias shrinks only as reps grow, which the per-estimate MC SE (0.0004) does not measure. The reps-to-infinity band is above 0.0815 and is not pinned here. This changes no reported classification: the SummEval demo clears the A/B band by more than 10x, and the one near-boundary fixture (R-squared change 0.074462) is class B under both 0.0757 and 0.0815. Tightening this band is tracked as a follow-up (a convergence check across 200/400/800 reps, with the boundary reported as an interval rather than a point).

## References

- `scripts/derive_ordinal_bands.py` (the study), `sim/results/band_derivation_points.csv` and `band_derivation_summary.csv` (raw draws and the derived summary).
- `docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md` (the ADR that named this as a deferred research task and states the dichotomous-to-ordinal transfer caveat).
- `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md` (the precedent for moving a shipped constant from a stated convention to a measured value).
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349.
- Choi, S. W., Gibbons, L. E., and Crane, P. K. (2011). lordif: An R Package for Detecting Differential Item Functioning Using Iterative Hybrid Ordinal Logistic Regression/Item Response Theory Calibration. Journal of Statistical Software, 39(8).
- Zumbo, B. D. (1999). A Handbook on the Theory and Methods of Differential Item Functioning (DIF): Logistic Regression Modeling as a Unitary Framework for Binary and Likert-Type (Ordinal) Item Scores.
