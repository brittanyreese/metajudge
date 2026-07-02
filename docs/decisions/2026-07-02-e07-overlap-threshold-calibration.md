# E07 DIF: conditioner-overlap flag recalibrated from convention to measurement

Date: 2026-07-02
Status: accepted
Supersedes: the threshold value and its "honest note" in
`2026-07-01-e07-dif-nested-strata-confound.md` (the confound analysis and the
three-field diagnostic surface are unchanged)

## Context

The nested-strata confound record shipped `conditioner_overlap_weak` with
`_OVERLAP_WEAK_CORR = 0.7` and stated plainly that 0.7 was "a project convention
marking where the single linear match should be read with suspicion, not an
empirically validated cutoff," and that validating it via simulation over
strata-separation levels was a separate research task. The 2026-07-02 multi-persona
review (methodology reviewer, finding 3) named this the central methodological hole:
the design's headline confound was governed by an uncalibrated advisory constant.

The operating-characteristics study (S3 in
[docs/sim-operating-characteristics.md](../sim-operating-characteristics.md), 16 null
cells, 6,317 converged replications across the rest-score-under-impact and
degraded-external-conditioner regimes, seeds pinned in `sim/oc_study.py`, code at
commit `ab751d0`) is that research task, executed.

## What the study measured

1. The convention watched the wrong part of the scale. Observed conditioner-group
   correlation rarely reaches 0.7 even at extreme strata separation (mean 0.52 at
   impact -2.0 with 3 raters). Cells whose false-DIF rate was already 100% fired the
   0.7 flag in 0% to 5% of replications. As shipped, the flag was decorative.
2. Binned by the correlation each run observes (the only quantity the flag can see),
   the false Jodoin-Gierl B/C classification rate under a true null is 0.5% at
   |corr| <= 0.2, 17% in (0.2, 0.4], and above 50% in (0.4, 0.6]. Past 0.4 the
   impurity inflates the mean effect size itself to 0.040, inside the B band, under
   no true DIF.
3. The false-DIF rate by p-value is far worse (79% already in the (0.2, 0.4] bin),
   which independently confirms the report card's existing choice to lead with the
   effect-size class and demote analytic p-values.
4. `conditioner_common_support` (a range-overlap measure) stayed above 0.9 in almost
   every confounded cell. It has no discriminating power for this risk and keeps no
   role in the flag.

## Decision

1. `_OVERLAP_WEAK_CORR` moves from 0.7 to 0.2. The flag now fires where the measured
   false B/C rate departs from the safe band (0.5% below 0.2, 17% above), instead of
   at a value the diagnostic almost never reaches.
2. The report-card warning wording drops "strongly (but not perfectly) correlated"
   (wrong at the calibrated threshold) and instead names the calibrated safe band and
   what was measured beyond it.
3. The flag stays advisory. It does not gate or suppress output; the identifiability
   refusal at |corr| > 0.999 is unchanged.
4. `conditioner_common_support` remains a reported descriptive field, explicitly
   carrying no calibration weight.

## Consequences

- Runs in the 0.2 to 0.7 observed-correlation band, which includes every strongly
  confounded simulation cell and the interop-doc fixture (0.872 already fired), now
  warn where they previously read as clean. That is the correction working: the S3
  false B/C rate in that band ranges from 17% to over 50%.
- The SummEval demo (|corr| = 0.147) stays below the flag, now with a measured basis
  (0.5% false B/C at that overlap) rather than an unvalidated one.
- The threshold is calibrated within one DGP family and two confound mechanisms; the
  limits section of the study document lists what was not varied. The value is now a
  measurement with stated scope, not a convention.

## References

- docs/sim-operating-characteristics.md (S3), raw draws in `sim/results/`.
- docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md (the confound analysis
  this calibrates).
- docs/reviews/2026-07-02-multi-persona-review.md (methodology finding 3; Devil's
  Advocate CRITICAL on unreported operating characteristics).
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates
  using an effect size measure with the logistic regression procedure for DIF
  detection. Applied Measurement in Education, 14(4), 329-349.
