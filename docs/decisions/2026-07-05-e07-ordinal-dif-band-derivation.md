# E07 DIF: empirical derivation of the ordinal-PO Jodoin-Gierl A/B/C bands

Date: 2026-07-05 Status: proposed (empirical finding recorded; the constants-update decision below is open)

## Context

`docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md` ships the Jodoin and Gierl (2001) A/B/C thresholds (0.035 negligible/moderate, 0.070 moderate/large) directly on the ordinal proportional-odds (PO) Nagelkerke R-squared change, and says so plainly: Jodoin and Gierl validated these cutoffs on the two-category logistic R-squared, so applying them to the ordinal PO change is "an applied convention... not a direct result" of the original validation, and "no published simulation has validated the 0.035 and 0.070 bands for the ordinal proportional-odds R-squared change specifically." That ADR named deriving the ordinal bands by Monte Carlo as a separate research task. This is that task, executed (`scripts/derive_ordinal_bands.py`, seed `BAND_SEED = 20260704`, raw draws in `sim/results/band_derivation_points.csv` and `band_derivation_summary.csv`).

## Method

1. Hold an interpretable DIF magnitude fixed: the uniform-DIF coefficient `b2`, a constant shift in the cumulative logit between focal and reference groups. It is a log odds-ratio (`OR = exp(b2)`) with the same meaning for a 2-category logistic outcome and a 5-category PO outcome, so it transports cleanly between them.
2. Reproduce the dichotomous regime Jodoin and Gierl calibrated on (`n_categories=2`, where the shipped PO fit reduces to ordinary logistic regression) and find the `b2` where the dichotomous mean R-squared change crosses 0.035 and 0.070. Those `b2` values are the true DIF magnitudes the shipped bands encode.
3. Read the ordinal (5-category) mean R-squared change at those same anchor `b2` values. That is the derived ordinal-PO band.

Both curves use the shipped engine directly (`metajudge.dif._dif_stats`, `_fit_proportional_odds`, `_nagelkerke`) on the shared cumulative-logit DGP (`sim.dgp`), at the SummEval demo's scale (n_obs = 4,800: 800 items per stratum, 3 raters). 400 replications per (metric, `b2`) point across a 13-point grid, all converged. Nothing about the numerics is reimplemented; only the `b2` sweep and the crossing-inversion are new code.

## Result

| Anchor | `b2` | Odds ratio | Shipped (dichotomous-calibrated) | Derived (ordinal-PO) |
| --- | --- | --- | --- | --- |
| A/B | 0.807 | 2.24 | 0.035 | **0.0376** (MC SE 0.00039) |
| B/C | 1.175 | 3.24 | 0.070 | **0.0757** (MC SE 0.00054) |

The derived ordinal bands are 7-8% higher than the shipped dichotomous-calibrated values, not lower. That direction matters: at a fixed shipped threshold, an ordinal DIF effect needs slightly _less_ true magnitude to cross into B or C than the dichotomous calibration implies, so the shipped 0.035/0.070 read very slightly liberal (a hair too easy to trip) relative to the derived ordinal bands, not conservative.

For reference, converting the field's other candidate cutoffs to the same ordinal-magnitude axis: lordif's own default (0.02, Choi, Gibbons and Crane 2011) corresponds to `b2 = 0.587` (OR 1.80) on this DGP, well below the ordinal-equivalent of the shipped 0.035 (`b2 = 0.777`, OR 2.17). The full curve (13 `b2` points, both metrics) is in `sim/results/band_derivation_points.csv`.

**The SummEval demo's headline class-A result is stable under every candidate threshold.** Its observed ordinal R-squared change is 0.002. Margin below each candidate A/B boundary: 10x (lordif 0.02), 17.5x (shipped 0.035), 18.8x (derived 0.0376). All three classify it class A. The miscalibration this study measures does not change any number this repo has already reported.

## Decision: shipped constants unchanged, pending a call on updating them

`_JG_NEGLIGIBLE = 0.035` and `_JG_LARGE = 0.070` in `src/metajudge/dif.py` are **not** changed by this record. Two things are true at once and the second has not been decided:

1. The miscalibration is now measured, not assumed: ~7-8% relative, in the liberal direction, within one DGP family (cumulative-logit, 5-category, this repo's stratum sizes and rater counts). This is a narrower correction than the overlap-threshold recalibration (`2026-07-02-e07-overlap-threshold-calibration.md`, which moved a constant by 3.5x), and it does not flip any classification this repo has reported so far.
2. Whether to move `_JG_NEGLIGIBLE`/`_JG_LARGE` to the derived 0.0376/0.0757 is a judgment call this record does not make: doing so is a small breaking change to every future DIF classification near the boundary, and (per `AGENTS.md`) any previously reported number is tagged to the exact commit that produced it, so a constant change does not retroactively invalidate a citation but does mean "class B" on a fresh run and "class B" on an old tagged run could mean measurably different things at the margin. That trade-off (scientific accuracy of a ~7-8% shift vs. threshold stability across releases) is for a maintainer to decide, not to default.

## Limits

- One DGP family (cumulative-logit PO, matching `sim/dgp.py`), one scale (n_obs = 4,800, 800/stratum, 3 raters), one seed. Not varied: rater count, item count, category count other than 2-vs-5, nonuniform DIF, or a degraded external conditioner.
- The anchors are single crossing points (A/B, B/C) on a 13-point grid with linear interpolation; the grid was chosen dense near both crossings (`OR` roughly 2 and 2.7) but is not adaptively refined.
- Runtime is a few minutes at the defaults (`--reps 400`), fully reproducible from `BAND_SEED = 20260704`.

## References

- `scripts/derive_ordinal_bands.py` (the study), `sim/results/band_derivation_points.csv` and `band_derivation_summary.csv` (raw draws and the derived summary).
- `docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md` (the ADR that named this as a deferred research task and states the dichotomous-to-ordinal transfer caveat).
- `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md` (the precedent for moving a shipped constant from a stated convention to a measured value).
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349.
- Choi, S. W., Gibbons, L. E., and Crane, P. K. (2011). lordif: An R Package for Detecting Differential Item Functioning Using Iterative Hybrid Ordinal Logistic Regression/Item Response Theory Calibration. Journal of Statistical Software, 39(8).
- Zumbo, B. D. (1999). A Handbook on the Theory and Methods of Differential Item Functioning (DIF): Logistic Regression Modeling as a Unitary Framework for Binary and Likert-Type (Ordinal) Item Scores.
