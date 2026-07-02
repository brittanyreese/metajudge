# E07 report card: clustering-robust DIF flag, not raw analytic p-values

Date: 2026-07-01
Status: accepted
Extends: 2026-06-23-e07-dif-cluster-bootstrap.md

## Context

`audit()` builds a `ReportCard` and `to_markdown` renders it. It used to print the analytic uniform and nonuniform DIF p-values as headline numbers. Two problems compound there.

1. Those p-values are anti-conservative. The analytic likelihood-ratio test pools each (item, rater) cell as one independent observation; under the crossed rater x item design the cells are not independent, so the test inflates significance (French & Finch, 2010; established in 2026-06-23-e07-dif-cluster-bootstrap.md).

2. On this card they are also orphaned. `_classify_jodoin_gierl` derives the A/B/C class from the Nagelkerke R-squared change alone; the p-values feed no decision the card renders. Full Jodoin-Gierl / lordif is a conjunction rule (flag iff the chi-square is significant *and* the R-squared change clears the negligible band), but this package already screens on magnitude. So the p-values were a significance signal that (a) nothing acted on and (b) was not even valid.

A methodology review asked the sharper question: do raw analytic p-values belong on a screening card at all? SPEC frames E07 as a screening card (R8) whose differentiation is "honest caveats" (R7), and commits no specific rendered fields. A screen should lead on its decision variable and must not present an invalid significance number as if it were sound.

## Decision

Lead the DIF block with the effect size and A/B/C class (the screen's decision variable), and replace the raw analytic significance headline with a clustering-robust flag derived from the item-cluster bootstrap.

- `ReportCard` gains an optional `dif_bootstrap: ClusterBootstrapDif | None`. `Flags` gains `dif_robustly_nonnegligible: bool | None`: `True`/`False` from whether the bootstrap Nagelkerke R-squared-change CI lower bound clears the negligible band (`_JG_NEGLIGIBLE`), `None` when no bootstrap was run or its CI is unreliable.
- `audit(..., robust=False)` (default) stays the fast analytic screen: effect size + class, significance reported as **not assessed**, with a pointer to the robust path. `audit(robust=True, n_boot=...)` runs `cluster_bootstrap_dif` and renders the R-squared-change CI plus the robust flag.
- The analytic uniform/nonuniform lines are kept but tagged `[analytic, unclustered]`. They carry the DIF *shape* (uniform vs nonuniform), which the total-only bootstrap does not, but the tag stops them being read as a clustering-robust significance test.

The rest-score / external / non-convergence / proportional-odds warnings are unchanged and still sit above the numbers.

## Consequences

- The card never presents significance it cannot defend. Fast mode says significance was not clustering-assessed; robust mode reports the bootstrap flag. Either way the anti-conservative analytic p-values are demoted and tagged, not headlined.
- The caveat now rides the numbers (inline tag + flag line) instead of a standing prose note, so it does not decay into wallpaper the way an always-present banner would.
- `audit()` default runtime is unchanged; the ~1000-refit cost is opt-in via `robust=True`, consistent with the prior ADR's "screening robustness check, not a confirmatory test."
- The robust flag rests on the item-cluster bootstrap, so it inherits that layer's limit: it addresses within-item rater dependence, not cross-item within-rater dependence. The fully correct fix (mixed-effects / GEE DIF) remains E04 / full-instrument scope.

## Alternatives considered

- Keep raw analytic p-values with a standing caveat note (the first patch in this review). Rejected: an always-on banner decays into wallpaper, and it still headlines an invalid number.
- Drop the analytic component tests entirely. Rejected: the tagged uniform/nonuniform split is the only source of DIF-shape information; the bootstrap reports the total only.
- Run the bootstrap in `audit()` by default. Rejected: turns the fast screen into ~1000 refits per call and conflates the screen with the opt-in confirmatory step.
- Cluster-robust sandwich SE (CR2/CR3) for a fast robust Wald p-value. Deferred: a valid robust p-value without a bootstrap, but a larger engine change than E07 warrants now; recorded as a future option alongside the mixed model.
