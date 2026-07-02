# E07 interop: map eval-instrument output to Ratings with rater = judge

Status: Accepted for E07. Date: 2026-06-23.
Records the measurement frame for the adapter from an LLM judge-runner's output (Epic's `evaluation-instruments`, `post.frame_from_evals`) into metajudge's `Ratings`, and why that frame is the citable one.

## Decision

`Ratings.from_eval_instruments(frames, *, criterion, stratum=None)` ingests one `frame_from_evals` output per judge, keyed by judge id, and produces a long-format `Ratings` under this frame: rater = the judge (or run) instance, item = the evaluated sample, score = one selected rubric criterion. Rubric criteria are a separate facet, audited one criterion at a time, and are never mapped to the rater axis. The adapter consumes only the DataFrame output, never imports the eval tool, and adds no runtime dependency.

## Why this frame, and not criteria-as-raters

A judge-runner's output is naturally a samples by criteria table for a single judge. The tempting shortcut, feeding rubric criteria into the rater axis of ICC or alpha, is a category error. With criteria in the rater slot the coefficient computed is an internal-consistency index (whether the rubric's dimensions co-vary as one scale), which is what Cronbach's alpha estimates (Cronbach, 1951; Cortina, 1993; McNeish, 2018). That is a different construct from inter-rater reliability, which asks whether independent scorers agree (Hallgren, 2012). metajudge audits the judge instrument, so the rater must be the scoring agent.

The intraclass-correlation framework is defined on a subjects by raters layout where raters are the scoring agents and the scored things are the targets or items (Shrout and Fleiss, 1979; McGraw and Wong, 1996; Koo and Li, 2016). The rater-mediated assessment literature makes the criterion-as-separate-facet explicit: many-facet Rasch and generalizability-theory models estimate rater severity and rubric criterion difficulty as distinct facets in one analysis (Li, 2022; Wang and Luo, 2019; Lim, 2011). Treating multiple judge instances, or multiple humans, as exchangeable raters for a two-way random-effects ICC is the standard practice the LLM-as-judge literature already uses (Zheng et al., 2023), with the caveat that chance-corrected agreement or an ICC is required rather than raw percent agreement, which overstates reliability (Norman, Rivera, and Hughes, 2026).

## One criterion at a time, not pooled

When a rubric has several dimensions, the adapter scores one dimension at a time rather than stacking sample-by-criterion rows into a single ICC. Generalizability theory and many-facet Rasch treat the criterion as its own facet that carries variance and interacts with raters, so pooling dimensions into one coefficient confounds between-criterion difficulty with rater disagreement and assumes criteria are exchangeable replicates, which they are not (Brennan, 2001; Li, 2022; Wang and Luo, 2019). A caller who wants a cross-dimension summary runs the audit per criterion and reports them side by side. Repeated runs of one judge (temperature or seed variation) are a legitimate alternative rater facet under the same shape, read as intra-rater or test-retest reliability (Hallgren, 2012; Norman et al., 2026).

## The exemplar and its limits

Epic's `evaluation-instruments` ships example clinical inputs, not saved judge outputs; producing real outputs needs a live model call and the full corpus is PHI-bearing and DUA-gated. The interop exemplar therefore runs on a small, de-identified fixture built in Epic's real `frame_from_evals` schema (the `(criterion, {class, score, notes})` MultiIndex), with no PHI and no API call. It demonstrates the seam and the report card on schema-faithful data and is labeled illustrative, not a finding about any clinical instrument. The shipped numerical demo remains SummEval (corpus-lock ADR, 2026-06-22); the Epic seam is the positioning and integration path onto a client's own judge outputs.

## Role split

Epic `evaluation-instruments` is the judge runner: clinical input plus a rubric in, LLM scores out. metajudge is the auditor: scores in, a reliability and DIF report card out. `Ratings.from_eval_instruments` is the integration layer between them, and it lives on `Ratings` beside `from_long` so the public surface stays small.

## What this changes in the build

`data.py` gains the `from_eval_instruments` classmethod and a `_select_criterion` helper that reads both `frame_from_evals` shapes (flat criteria columns, and the detailed `(criterion, field)` MultiIndex). No new runtime dependency, no change to the existing pillars. The interop note in `docs/interop-epic.md` carries a runnable example, verified to produce the report card it shows.

## References

Brennan, R. L. (2001). Generalizability theory. Springer. doi:10.1007/978-1-4757-3456-0_6

Cortina, J. M. (1993). What is coefficient alpha? An examination of theory and applications. Journal of Applied Psychology, 78(1), 98-104. doi:10.1037/0021-9010.78.1.98

Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16(3), 297-334. doi:10.1007/BF02310555

Hallgren, K. A. (2012). Computing inter-rater reliability for observational data: An overview and tutorial. Tutorials in Quantitative Methods for Psychology, 8(1), 23-34. doi:10.20982/tqmp.08.1.p023

Koo, T. K., and Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine, 15(2), 155-163. doi:10.1016/j.jcm.2016.02.012

Li, W. (2022). Scoring rubric reliability and internal validity in rater-mediated EFL writing assessment: Insights from many-facet Rasch measurement. Reading and Writing, 35(10), 2409-2431. doi:10.1007/s11145-022-10279-1

Lim, G. S. (2011). The development and maintenance of rating quality in performance writing assessment: A longitudinal study of new and experienced raters. Language Testing, 28(4), 543-560. doi:10.1177/0265532211406422

McGraw, K. O., and Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. Psychological Methods, 1(1), 30-46. doi:10.1037/1082-989X.1.1.30

McNeish, D. (2018). Thanks coefficient alpha, we'll take it from here. Psychological Methods, 23(3), 412-433. doi:10.1037/met0000144

Norman, J. D., Rivera, M. U., and Hughes, D. A. (2026). Reliability without validity: A systematic, large-scale evaluation of LLM-as-a-judge models across agreement, consistency, and bias. arXiv:2606.19544.

Shrout, P. E., and Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological Bulletin, 86(2), 420-428. doi:10.1037/0033-2909.86.2.420

Wang, J., and Luo, K. (2019). Evaluating rater judgments on ETIC Advanced writing tasks: An application of generalizability theory and many-facet Rasch. Studies in Language Assessment. doi:10.58379/vmak1620

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.
