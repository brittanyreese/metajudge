# E07 DIF: ordinal logistic regression replaces dichotomized Mantel-Haenszel

Status: Accepted for E07. Date: 2026-06-22.
Supersedes the dichotomized Mantel-Haenszel DIF pillar and the earlier matching-method iterations. This is the single current DIF decision record.

## Decision

The DIF pillar uses ordinal (proportional-odds) logistic regression DIF, the Zumbo (1999) framework as implemented for ordinal scores by Crane et al. (2006) and the lordif package (Choi, Gibbons, and Crane, 2011). It replaces the dichotomized Mantel-Haenszel estimator entirely. The public entry point becomes `logistic_dif`; `mantel_haenszel_dif` and its 2x2 pooling helper are removed from the shipped surface.

## Why the old estimator was wrong

The dichotomized Mantel-Haenszel pillar matched each unit on its own per-rater mean and dichotomized that same mean at the median. The matching variable and the studied response were the same quantity, so the analysis was circular. The symptom was a verdict that swung with the bin count: significant at two match bins, null at eight. Conditioning on a score that contains the studied response is a documented defect in the DIF literature, known as matching-criterion contamination (Holland and Thayer, 1986; Dorans and Holland, 1992; Clauser, Mazor, and Hambleton, 1993). The fix is not a finer estimator on the same circular conditioner; it is an independent conditioner plus an estimator native to the 1-to-5 ordinal scale.

## The model

Three nested proportional-odds (cumulative logit) models per comparison, following lordif (Choi et al., 2011) and Crane et al. (2006):

- Model 1: `logit P(Y >= k) = alpha_k + b1 * conditioner`
- Model 2: `... + b2 * group` (adds the group main effect)
- Model 3: `... + b3 * (conditioner x group)` (adds the interaction)

Three likelihood-ratio chi-square tests fall out of the nesting:

- Total DIF, Model 1 vs Model 3, 2 degrees of freedom.
- Uniform DIF, Model 1 vs Model 2, 1 degree of freedom.
- Nonuniform DIF, Model 2 vs Model 3, 1 degree of freedom.

`Y` is the ordinal 1-to-5 score, `group` is the focal-vs-reference stratum, and `conditioner` is the independent quality measure described next.

## The conditioner and its data precondition

DIF needs a quality conditioner that does not contain the studied response. The field standard is for the method to compute that conditioner internally as a purified or leave-one-out rest score, not to demand an external one (Swaminathan and Rogers, 1990; French and Maller, 2007; Magis et al., 2010). The library follows that standard, with an explicit override for callers who hold a stronger conditioner:

- Default: a leave-one-rater-out rest score. For each (unit, rater) row, the conditioner is the mean score the other raters gave that unit. This treats raters as exchangeable measures of unit quality, the same assumption the reliability pillar (alpha, ICC) already makes, so the two pillars rest on one coherent premise. It needs at least two raters per unit.
- Override: an explicit per-unit conditioner, for an external gold quality score or a leave-one-criterion-out mean across rubric dimensions. This is the stronger path and is preferred for short scales or a single-rater judge.
- Precondition guard: with a single rater and no external conditioner there is no way to form an independent conditioner. The function raises rather than returning a number it cannot defend. This encodes the standing rule that we do not claim DIF where the data cannot support an independent conditioner.

## Effect size and classification

Effect size is the Nagelkerke pseudo-R-squared change from Model 1 to Model 3, computed from first principles, not read from any package attribute. The metric is pinned to Nagelkerke explicitly to avoid the McFadden-versus-Nagelkerke mix-up that makes threshold tables non-comparable (Gomez-Benito, Hidalgo, and Padilla, 2009).

Magnitude class uses the Jodoin and Gierl (2001) thresholds on the Nagelkerke R-squared change: below 0.035 is negligible (A), 0.035 to 0.070 is moderate (B), above 0.070 is large (C). These are the simulation-validated, more conservative successors to the original Zumbo and Thomas (1997) cutoffs of 0.13 and 0.26, and the field has largely moved to them. One caveat stated honestly: Jodoin and Gierl validated these cutoffs on the two-category logistic R-squared, so applying the same bands to the ordinal proportional-odds Nagelkerke change is an applied convention, an extension of the original validation rather than a direct result of it. It traces to the Zumbo (1999) ordinal extension of the effect-size measure, not to a lordif default: lordif's own default R-squared-change cutoff is 0.02, and Choi et al. (2011) recommend deriving detection thresholds empirically by Monte Carlo. No published simulation has validated the 0.035 and 0.070 bands for the ordinal proportional-odds R-squared change specifically (literature check, 2026-06-23). The strict-versus-closed endpoint handling (A below 0.035, B on the closed-open interval, C at or above 0.070) is a project choice; the source states the bands, not the endpoints. The A/B/C labels mirror the familiar ETS scale but are a distinct R-squared magnitude rule, not the ETS Mantel-Haenszel delta classification, and the report card says so.

The three chi-square tests are nested likelihood-ratio tests (Model 1 vs 3, 1 vs 2, 2 vs 3). Pre-clamp they telescope exactly: the total equals the sum of the uniform and nonuniform statistics, because the Model 2 log-likelihood cancels. The report card still presents them as three separate tests, and the only departure from exact additivity is the per-test clamp at zero, a numerical-noise guard that doubles as a divergent-fit flag.

## Oracle and testing

Runtime dependencies stay numpy, pandas, scipy, and krippendorff. The proportional-odds fit is implemented in scipy alone (ordered thresholds parameterized by positive increments, fit by maximum likelihood). statsmodels is never imported at runtime.

Per the numerical TDD convention, every reported statistic is pinned to an external oracle. The oracle is R `MASS::polr`, the canonical proportional-odds fit: the three nested models were fit with polr on the frozen fixture and their likelihood-ratio chi-squares and Nagelkerke R-squared change pinned as literal constants. The scipy engine reproduces polr's three log-likelihoods to six decimals. The engine is also cross-checked in-process against statsmodels `Logit` in the two-category limit, where proportional odds reduces to ordinary logistic regression; the fitter matches the Logit MLE to floating point.

statsmodels `OrderedModel(distribution="logit")` was evaluated as the oracle and rejected. It does not reproduce the canonical proportional-odds likelihood: its fit disagrees with both polr and the binary-limit Logit MLE (it returns systematically shrunken slopes and a log-likelihood that, evaluated under the textbook cumulative-logit formula at its own parameters, does not match what it reports). The discrepancy is stable across its bfgs, lbfgs, newton, and powell fitters, so it is not a non-convergence artifact; the exact parameterization cause was not pinned down and is left open, because two independent gold standards (R polr and statsmodels Logit) agree with the engine to high precision, which settles correctness without needing it. OrderedModel is not used. When a reference value and a literal disagree, the reference wins and the literal is corrected; tolerances are never loosened to pass.

## What this changes in the build

1. `dif.py` drops `mantel_haenszel_dif`, `_mh_from_tables`, and the dichotomization, and gains `logistic_dif` plus the scipy proportional-odds fitter.
2. `DifResult` loses the odds-ratio and MH-delta fields and gains the three chi-square statistics with their p-values, the Nagelkerke R-squared change, the A/B/C class, the conditioner source, the observation count, and a `converged` flag.
3. `__init__.py`, `report.py`, `demo.py`, and the public-API test track the rename and the new result shape. The report card prints the uniform and nonuniform verdicts and the effect-size class.

## Limitations stated honestly

- With stratum assigned per unit, the group and the conditioner vary at the unit level while the response varies by rater, so rows within a unit are not independent. The standard errors are mildly optimistic. The honest framing is a screening audit, not a confirmatory significance claim.
- The proportional-odds assumption is not tested here. Where raters use the scale very differently it can be violated; lordif mitigates this with an IRT-theta conditioner, which this scope does not implement.
- The leave-one-rater-out rest score is a thin conditioner when raters are few. For short rubric scales an external or leave-one-criterion-out conditioner is preferable and is supported through the override.
- The rest score is contaminated when the bias is shared across the rater panel. It removes the studied rater's own response but not a bias every rater carries, so for instrument-level bias (the same judge applied to a group) it understates the DIF. On the SummEval demo this shows up directly: the frozen fixture, built with a uniform bias shared across raters, lands at class C under an independent conditioner but class A under the rest score. Both numbers are pinned in the tests as an honest illustration, and the rule is stated in the function docstring: use a valid independent external conditioner for stronger instrument-level analysis; the rest score detects panel-relative (differential-rater) functioning.
- Degenerate inputs are refused rather than returned as indefensible numbers, matching the single-rater guard: a single response category (no ordinal variation), a constant conditioner (nothing to match on), and a conditioner near-perfectly collinear with the group (DIF not identifiable, as under perfect within-item rater agreement) each raise a named `ValueError`. The likelihood-ratio statistics are clamped at zero so optimizer noise cannot surface a negative chi-square. `DifResult.converged` exposes whether all three model fits converged.

## References

Choi, S. W., Gibbons, L. E., and Crane, P. K. (2011). lordif: An R package for detecting differential item functioning using iterative hybrid ordinal logistic regression / item response theory and Monte Carlo simulations. Journal of Statistical Software, 39(8). doi:10.18637/jss.v039.i08

Clauser, B. E., Mazor, K. M., and Hambleton, R. K. (1993). The effects of purification of the matching criterion on the identification of DIF using the Mantel-Haenszel procedure. Applied Measurement in Education, 6(4), 269-279. doi:10.1207/s15324818ame0604_2

Crane, P. K., Gibbons, L. E., Jolley, L., and van Belle, G. (2006). Differential item functioning analysis with ordinal logistic regression techniques. Medical Care, 44(11 Suppl 3), S115-S123. doi:10.1097/01.mlr.0000245183.28384.ed

Dorans, N. J., and Holland, P. W. (1992). DIF detection and description: Mantel-Haenszel and standardization. ETS Research Report Series, 1992(1). doi:10.1002/j.2333-8504.1992.tb01440.x

French, B. F., and Maller, S. J. (2007). Iterative purification and effect size use with logistic regression for differential item functioning detection. Educational and Psychological Measurement, 67(3), 373-393. doi:10.1177/0013164406294781

Gomez-Benito, J., Hidalgo, M. D., and Padilla, J. L. (2009). Efficacy of effect size measures in logistic regression: An application for detecting DIF. Methodology, 5(1), 18-25. doi:10.1027/1614-2241.5.1.18

Holland, P. W., and Thayer, D. T. (1986). Differential item functioning and the Mantel-Haenszel procedure. ETS Research Report Series, 1986(2). doi:10.1002/j.2330-8516.1986.tb00186.x

Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349. doi:10.1207/s15324818ame1404_2

Magis, D., Beland, S., Tuerlinckx, F., and De Boeck, P. (2010). A general framework and an R package for the detection of dichotomous differential item functioning. Behavior Research Methods, 42(3), 847-862. doi:10.3758/brm.42.3.847

Swaminathan, H., and Rogers, H. J. (1990). Detecting differential item functioning using logistic regression procedures. Journal of Educational Measurement, 27(4), 361-370. doi:10.1111/j.1745-3984.1990.tb00754.x

Zumbo, B. D. (1999). A handbook on the theory and methods of differential item functioning (DIF): Logistic regression modeling as a unitary framework for binary and Likert-type (ordinal) item scores. Ottawa: Directorate of Human Resources Research and Evaluation, Department of National Defence.
