# E07 ICC: refuse on incomplete data, defer the estimator to E04

Status: Accepted for E07. Date: 2026-06-23.
Records the standing decision on how `reliability.py::icc` handles a missing or partially-crossed targets x raters matrix, and the method-currency pass behind it.

## Decision

`icc` stays the balanced Shrout-Fleiss (1979) two-way random-effects, absolute-agreement estimator. On any missing cell it raises a `ValueError` that names the correct method for incomplete data, names the biased fallback it is declining to use, and points to the E04 variance pillar where the correct estimator belongs. E07 does not ship an incomplete-data ICC. Listwise deletion is not adopted as a default, and a hand-rolled no-omission estimator is not built now.

## Why not just handle the missing cells

The question looks like a small robustness upgrade and is not. Three findings from a method-currency pass over the current literature settle it.

First, the estimand changes. ICC(2,1)/(2,k) is defined on a complete crossed design; the Shrout-Fleiss mean-square decomposition (`ss_rows + ss_cols + ss_error = ss_total`) holds only when every target is rated by every rater. The moment the matrix is unbalanced, the correct object is a variance-components estimate, not a balanced-ANOVA coefficient (ten Hove, Jorgensen, and van der Ark, 2024).

Second, the correct estimator is the deferred E04 engine. The current methods consensus for incomplete interrater designs is maximum-likelihood estimation of a random-effects (variance-components / generalizability-theory) model, with ten Hove et al. (2025) finding ML of a random-effects model with Monte-Carlo confidence intervals preferred over Bayesian and common-factor alternatives on bias, RMSE, and coverage. That is the same variance-decomposition machinery (REML over a crossed random-effects model) that the project SPEC scopes to E04 and the lit review places as the second open gap. A defensible no-omission ICC cannot be reached without doing a slice of that deferred work.

Third, the lightweight shortcut is not a strong enough oracle. `irrNA::iccNA` (Brueckl and Heuer, 2022) generalizes the McGraw and Wong (1996) ICCs to randomly incomplete data via Ebel's (1951) method, with no imputation and no omission, and reduces to the standard ICCs on balanced data. It is genuinely lightweight (R `stats` only) and is the one real lightweight standard for unbalanced ICC. But it is a single software-validated package with no independent simulation backing of the strength the gold-standard ML estimator carries. Pinning a number to iccNA alone, in a regime where iccNA and the gold-standard ML estimator can genuinely disagree, would certify a contested reliability coefficient on the commodity pillar. That is the reputational-damage shape the non-compressible-correctness rule (SPEC R19) exists to prevent, and there is no differentiating upside on the reliability pillar to justify the risk.

An adversarial review of the build-it option rejected it on all four of: scope (the method is E04-shaped), oracle strength (iccNA is too thin), demo relevance (the shipped corpus never exercises the path), and opportunity cost against the finish track. The refusal is the honest answer: it states the correct method and declines to fake it.

## What the standard tools do

The refusal is squarely inside current practice. On a missing cell, Python `pingouin.intraclass_corr` raises by default (`nan_policy='raise'`); R `irr::icc` and `psych::ICC` perform listwise deletion. Listwise is documented as biased and information-discarding for incomplete designs, not recommended as correct. The library matches pingouin's conservative default and adds a message that explains why.

## Demo relevance

The locked demo corpus (SummEval, corpus-lock ADR 2026-06-22) maps the worked demo to the fully crossed three-expert coherence slice, so the missing-cell branch does not fire on the shipped demo. One honest caveat: SummEval's crowd-rater assignments are not fully crossed in the raw release, so a user who loads the crowd raters can hit the refusal. That is an argument for a clear, cited message, which this decision provides, not for a half-strength estimator under the commodity pillar.

## What this changes in the build

The single `ValueError` in `reliability.py::icc` gains a cited, method-naming message. The missing-cell rejection test in `tests/test_icc.py` asserts the message names the correct estimator (variance-components), the biased fallback (listwise), and the E04 deferral, so the guidance is a tested contract rather than prose that can rot. No runtime dependency is added; the balanced estimator and its Shrout-Fleiss and pingouin oracles are unchanged.

## When this is revisited

Incomplete-data reliability becomes in scope at E04, where the variance-components / G-theory estimator is built and validated against the ML standard (ten Hove et al., 2025), and where R `lme4::lmer` or an equivalent ML fit is the oracle rather than iccNA alone. If E07 is ever re-scoped to make incomplete-data reliability a headline feature, the same rule applies: it is oracle-validated against the ML estimator, not iccNA, and that is an E04-sized task by the project's own SPEC.

## References

Brueckl, M., and Heuer, F. (2022). irrNA: Coefficients of interrater reliability, generalized for randomly incomplete datasets (R package version 0.2.3). CRAN. doi:10.32614/CRAN.package.irrNA

Ebel, R. L. (1951). Estimation of the reliability of ratings. Psychometrika, 16(4), 407-424. doi:10.1007/BF02288803

Koo, T. K., and Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine, 15(2), 155-163. doi:10.1016/j.jcm.2016.02.012

McGraw, K. O., and Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. Psychological Methods, 1(1), 30-46. doi:10.1037/1082-989X.1.1.30

Nakagawa, S., and Schielzeth, H. (2010). Repeatability for Gaussian and non-Gaussian data: a practical guide for biologists. Biological Reviews, 85(4), 935-956. doi:10.1111/j.1469-185X.2010.00141.x

Shrout, P. E., and Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological Bulletin, 86(2), 420-428. doi:10.1037/0033-2909.86.2.420

ten Hove, D., Jorgensen, T. D., and van der Ark, L. A. (2024). Updated guidelines on selecting an intraclass correlation coefficient for interrater reliability, with applications to incomplete observational designs. Psychological Methods, 29(5), 967-979. doi:10.1037/met0000516

ten Hove, D., Jorgensen, T. D., and van der Ark, L. A. (2025). How to estimate intraclass correlation coefficients for interrater reliability from planned incomplete data. Multivariate Behavioral Research. doi:10.1080/00273171.2025.2507745
