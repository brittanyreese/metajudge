# E07 DIF: cluster bootstrap for clustered-error inference

Date: 2026-06-23
Status: accepted
Supersedes: none (extends 2026-06-22-e07-dif-ordinal-logistic-regression.md)

## Context

`logistic_dif` pools each (item, rater) cell into one row and fits the three nested proportional-odds models on `n = items x raters` observations treated as independent. They are not independent: the same rater scores many items, and the same item is scored by many raters. The likelihood-ratio chi-square and its p-values assume independent observations, so under positive intraclass correlation the analytic test is anti-conservative: standard errors are deflated, the chi-square is inflated, and the false-positive DIF rate rises above the nominal level.

A methodology review flagged this as the sharpest inference gap. A literature check (2026-06-23) confirmed it:

- French and Finch (2010, Journal of Educational Measurement 47(3)) showed by simulation that single-level logistic-regression DIF fails to hold the nominal .05 Type-I rate under multilevel data, while a hierarchical (random-effects) model restores it.
- Musca et al. (2011, Frontiers in Psychology) showed that ignoring intraclass correlation inflates Type-I error severely in least-squares analysis of nested data.
- Cameron and Miller (2015) and Liang and Zeger (1986) establish cluster-robust inference as the standard remedy when full random-effects modeling is not used.
- den Noortgate and De Boeck (2005, JEBS) give the principled fix: a logistic mixed / GEE DIF model.

The literature also establishes that the magnitude (and even the sign) of the inflation is design-dependent: French and Finch found severe inflation only when the grouping variable is between-cluster, and the correction can cost power otherwise. The magnitude is therefore an empirical quantity for this crossed rater-by-item design, not a constant to borrow.

## Decision

Keep the analytic engine unchanged (its statistics remain pinned to R `MASS::polr`) and add a non-parametric cluster bootstrap as a robustness layer for the inference: `cluster_bootstrap_dif(...) -> ClusterBootstrapDif`.

1. Cluster unit: the item. Resample items with replacement, stratified within the focal and reference groups so group sizes are preserved, and carry each resampled item's full block of rater scores together. Keeping the rater block intact preserves the within-item rater correlation that the analytic i.i.d. model ignores; resampling at the item level makes the effective sample the number of independent items per group, not the number of (item, rater) cells.
2. For each resample, refit through the validated `logistic_dif` engine and collect the total-DIF chi-square and the Nagelkerke R-squared change. Degenerate resamples (a draw with no ordinal variation) are dropped and the realized count is reported as `n_effective`, mirroring the alpha bootstrap.
3. Report percentile (2.5/97.5) confidence intervals for the Nagelkerke R-squared change and for the total chi-square. The effect-size interval is the primary output: if its lower bound sits in the negligible (class A) band, the analytic point estimate is not robust to the clustering.

The bootstrap is seeded and reproducible. Like the alpha percentile CI, it has no external reference oracle; the tests pin its reproducibility and verify that it widens relative to a naive observation-level resample on within-item-clustered data (the behavior that demonstrates the correction).

## Scope and limits

- The item-cluster bootstrap addresses within-item rater dependence and item-level sampling. It does not remove cross-item within-rater dependence (one judge scoring many items). That residual two-way dependence is handled correctly only by a random-effects / GEE model, which is full-instrument scope outside E07; this is the scipy-only interim, not the final word.
- The bootstrap is a screening robustness check, not a confirmatory test. The framing in the report card stays "screen, not verdict."
- Runtime dependencies are unchanged (numpy, scipy, pandas, krippendorff); the bootstrap reuses the existing engine and `numpy.random`.

## Alternatives considered

- Drop the p-values and report only the effect size and class. Simpler, but discards information a cluster bootstrap can instead quantify. Rejected in favor of measuring the uncertainty.
- A logistic mixed model / GEE now. The principled fix, but it is full-instrument random-effects machinery and exceeds E07's scipy-only, no-new-dependency constraint.
- A rater-cluster bootstrap. Captures the repeated-judge dependence but treats items (which carry the DIF grouping) as fixed; the item is the more natural replication unit for a group effect defined at the item level. Recorded as a possible future option.

## References

- French, B. F., and Finch, W. H. (2010). Hierarchical logistic regression: Accounting for multilevel data in DIF detection. Journal of Educational Measurement, 47(3), 299-317.
- Musca, S. C., et al. (2011). Data with hierarchical structure: impact of intraclass correlation and sample size on Type-I error. Frontiers in Psychology, 2:74.
- Cameron, A. C., and Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317-372.
- Liang, K.-Y., and Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. Biometrika, 73(1), 13-22.
- den Noortgate, W. V., and De Boeck, P. (2005). Assessing and explaining differential item functioning using logistic mixed models. Journal of Educational and Behavioral Statistics, 30(4), 443-464.
