# E07 DIF: nested-strata conditioner/group confound is a stated scope limit, not a bug

Status: Accepted for E07. Date: 2026-07-01.
Extends the DIF method record (`2026-06-22-e07-dif-ordinal-logistic-regression.md`), including its "conditioner and its data precondition" section, which is the matching record. It does not change the estimator; it names, first-class, a structural limit of the "DIF across output strata" operationalization and records how the report card and the engine handle it.

## Context

A senior review of the shipped engine flagged that "DIF across output strata," as operationalized here, differs from textbook examinee DIF in a way that can confound the effect it measures. This record states the difference plainly, explains why it is inherent to the design rather than a defect to fix, and pins the engine and report-card behavior that keep the caveat in front of the reader.

## The structural fact

In classical DIF the same items are administered to both the focal and reference groups, and the conditioner (a total or rest score) estimates examinee ability on those shared items. Group membership varies across examinees; the items are common.

metajudge inverts that. Each item maps to exactly one stratum (`Ratings.from_long` enforces one stratum per item), so the "group" is a property of the item, not of an examinee answering a shared item. The strata partition the items into disjoint sets. The conditioner (the leave-one-rater-out rest score, or an external per-item quality score) is therefore matched between two nested item sets, never within a common item answered by both groups.

This is the intended operationalization (SPEC R2, R3: DIF across output strata), and it is a legitimate screen. But it carries a confound that shared-item DIF does not.

## The confound

The DIF test asks: at a matched conditioner (quality) level, do focal-stratum items receive a systematically different score distribution than reference-stratum items? The match is only as clean as the overlap between the two strata's conditioner distributions.

When the strata genuinely differ in quality (long summaries really are more complete than short ones; code answers really are more verifiable than prose answers), the conditioner correlates with the group. Three regimes follow:

1. Near-perfect confounding (`|corr(conditioner, group)| > 0.999`). Quality and group are inseparable and DIF is not identifiable. The engine raises a named `ValueError` (`_dif_stats`) rather than returning a number it cannot defend. This is the existing identifiability guard and it is correct.
2. Strong-but-partial confounding (high correlation, below the refusal threshold). DIF is identifiable, but the conditioner enters as a single linear term, so a nonlinear quality-to-response relationship over a region where the two strata barely overlap leaves residual confounding. This is matching-criterion impurity in the sense of Clauser, Mazor, and Hambleton (1993), arising here from the study design (nested strata) rather than from a contaminated conditioner. The reported effect size can absorb a genuine quality difference between strata as apparent DIF.
3. Well-overlapping strata. The match is clean and the screen behaves like ordinary conditioned DIF.

The engine cannot tell regime 2 from regime 3 from the fit alone: both converge and both return a finite effect size. The distinction is a property of the design and the data, and it is the reader's to weigh.

## Decision

1. The estimator is unchanged. A more elaborate conditioner (a nonlinear or multi-term match, or an IRT-theta conditioner as in lordif) is out of the two-pillar E07 scope, and would not remove the confound where the strata barely overlap. It would trade a stated, visible limit for a hidden one.
2. The confound is stated first-class on the report card, above the statistics, independent of the conditioner source, so a reader who excerpts the headline numbers cannot drop it. The wording frames the effect size as screening evidence, not a confound-free fairness verdict.
3. The identifiability refusal (regime 1) stays. Refusing an unidentifiable DIF is preferable to reporting an indefensible one, consistent with the single-rater and constant-conditioner guards.
4. The supported reading is recorded here and referenced from the report card: a non-negligible effect size under this screen flags a stratum pair worth investigating with a stronger design (shared items, or an external conditioner known to overlap across strata), not a settled instrument-level bias finding.

## Per-run diagnostic

The decision above states the confound as a design fact, true of every run. It does not tell a reader whether a given run sits in regime 2 or regime 3. `DifResult` now carries three fields that answer that, per run:

1. `conditioner_group_corr: float`. The Pearson correlation between the standardized per-row conditioner and the group indicator. This is the same quantity the engine already computes for its identifiability guard (the one that refuses when `|corr| > 0.999`), now exposed on the result instead of only checked internally.
2. `conditioner_common_support: float`. An item-level overlap measure: the share of items, focal and reference pooled, whose per-item conditioner value falls in the range both strata cover. It is `1.0` when the two strata cover the same range and moves toward `0.0` as they separate. The per-item representative is the external conditioner value where one was supplied, or the item's mean rating under the rest-score default. This is the more interpretable, reader-facing signal: it says how much the two strata actually overlap, not just how linearly correlated the match is.
3. `conditioner_overlap_weak: bool`. An advisory flag, set when `_OVERLAP_WEAK_CORR <= |conditioner_group_corr| <= 0.999`. As originally shipped the constant was `0.7`; it was recalibrated to `0.2` on 2026-07-02 from measured false-classification rates (see the superseding record below). It marks regime 2 from the confound analysis above: DIF is identifiable, but the conditioner correlates with the group enough that the reported effect size may be absorbing a real between-strata quality gap as apparent DIF.

The flag drives the report card. When `conditioner_overlap_weak` is set, `ReportCard.to_markdown` replaces the blanket nested-strata caveat from item 2 of the Decision above with a run-specific warning that names the correlation and common-support numbers for that run, placed above the DIF statistics.

HONEST NOTE (superseded 2026-07-02): as shipped, `0.7` was a project convention marking where the single linear match should be read with suspicion, not an empirically validated cutoff, and this record named its validation (a simulation over strata-separation levels) as a separate research task. That task has since run: the operating-characteristics study measured false Jodoin-Gierl B/C rates under a true null and recalibrated the constant to `0.2`. The decision and the measured curve are in `2026-07-02-e07-overlap-threshold-calibration.md`.

`conditioner_common_support` is reported on the result and, when the flag fires, in the report-card warning, but it is not itself thresholded. There is deliberately no second gate on it, to avoid calibrating a second cutoff on top of the first.

The flag is advisory only, like `po_violation` elsewhere in this module: the `|corr| > 0.999` refusal in `_dif_stats` remains the only hard stop. The estimator, the refusal, and the pinned DIF oracle values are unchanged by this diagnostic.

## What this changes in the build

- `report.py` `to_markdown` gains a conditioner-source-independent caveat, placed above the DIF statistics, naming the nested-strata match and the residual-confounding risk.
- No change to `dif.py` numerics. The existing `|r| > 0.999` refusal and the single-linear-term docstring caveat in `logistic_dif` remain the code-level expression of regimes 1 and 2.
- This record is the citable rationale the report-card note points to.

## Relationship to the existing DIF limitations

The method ADR already lists related limits: within-unit non-independence (optimistic standard errors, hence a screening not a confirmatory framing), the untested proportional-odds assumption, the thin rest score with few raters, and rest-score contamination by panel-shared bias. This record adds the one that review found under-stated: the group is an item-level property and the conditioner is matched between nested item sets, so a real quality gap between strata can masquerade as DIF. It is the design-level companion to the conditioner-level contamination point already on record.

## References

Clauser, B. E., Mazor, K. M., and Hambleton, R. K. (1993). The effects of purification of the matching criterion on the identification of DIF using the Mantel-Haenszel procedure. Applied Measurement in Education, 6(4), 269-279. doi:10.1207/s15324818ame0604_2

Dorans, N. J., and Holland, P. W. (1992). DIF detection and description: Mantel-Haenszel and standardization. ETS Research Report Series, 1992(1). doi:10.1002/j.2333-8504.1992.tb01440.x

Shealy, R., and Stout, W. (1993). A model-based standardization approach that separates true bias/DIF from group ability differences and detects test bias/DTF as well as item bias/DIF. Psychometrika, 58(2), 159-194. doi:10.1007/BF02294572

Zumbo, B. D. (1999). A handbook on the theory and methods of differential item functioning (DIF): Logistic regression modeling as a unitary framework for binary and Likert-type (ordinal) item scores. Ottawa: Directorate of Human Resources Research and Evaluation, Department of National Defence.
