# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Tightened DIF report language: external conditioners are now labeled as external-conditioner DIF and carry an explicit validity caveat instead of being treated as automatic instrument-level evidence.
- `Ratings.from_long` now refuses duplicate item-rater cells and missing stratum values rather than silently averaging duplicates or carrying null strata into DIF.
- `cluster_bootstrap_dif` now drops non-converged bootstrap refits from confidence intervals and reflects the surviving count in `n_effective`. CI bounds are `nan` when no resamples survive (previously collapsed to the point estimate).
- Alpha CI warnings now surface thin bootstrap intervals even when no degenerate resamples were dropped.

### Added

- `DifResult.conditioner_is_external` property: returns `True` when the conditioner came from an external source rather than the panel rest score. Previously this test required comparing `dif.conditioner_source == "external"` by hand.
- `_brant_test` now short-circuits on near-singular Fisher information (condition number > 1e10), returning `converged=False` instead of passing a large pseudoinverse through the Wald statistic and producing a spurious `po_violation` signal.

### Changed

- `Flags` dataclass simplified: `converged` and `po_violation` fields removed (use `card.dif.converged` and `card.dif.po_violation` directly). The `conditioner_is_external` and `alpha_ci_degraded` fields remain.

### Removed

- `brant_test` function and `BrantResult` type are no longer part of the public API. The Brant proportional-odds diagnostic is now a private implementation detail inside `dif.py`. Code importing `from metajudge import brant_test` or `from metajudge.diagnostics import brant_test` must be updated; `diagnostics.py` no longer exists.
- `Ratings.n_items` and `Ratings.n_raters` properties removed. Use `len(ratings.items)` and `len(ratings.raters)` respectively.
- `Flags.converged` removed (breaking change for any caller reading `card.flags.converged`; migrate to `card.dif.converged`).
- `Flags.po_violation` removed (breaking change; migrate to `card.dif.po_violation`).

## [0.1.0] - 2026-06-26

First public release: the E07 two-pillar report card. A pip-installable layer that audits a multi-rater ordinal scoring instrument (an LLM-as-judge or a human rater panel) for reliability and differential item functioning, and prints a one-screen report card. Runtime dependencies are limited to `numpy`, `pandas`, `scipy`, and `krippendorff`.

### Added

- `Ratings`: a long-format (rater, item, stratum, score) data model and the coder-by-unit matrix the pillars consume, plus `Ratings.from_eval_instruments`, a zero-dependency adapter from Epic `evaluation-instruments` score frames.
- Reliability pillar: Krippendorff's alpha with a percentile bootstrap CI, and ICC(2,1)/(2,k) from the Shrout-Fleiss two-way ANOVA. `AlphaResult` reports both the requested `n_bootstrap` and the realized `n_effective`, with a `ci_reliable` flag when too few resamples survive.
- DIF pillar: ordinal proportional-odds logistic-regression DIF in the Zumbo tradition (uniform, nonuniform, and total tests), with the Nagelkerke R-squared change classified A/B/C on the Jodoin-Gierl thresholds. `cluster_bootstrap_dif` adds a cluster (item-block) bootstrap for clustered-error inference, with a configurable CI level and a `ci_reliable` flag matching the reliability pillar.
- `audit()` facade and a `ReportCard` that renders the pillars to a Markdown report, with honest caveats surfaced above the numbers: panel-relative vs instrument-level DIF labeling, a non-convergence warning, and an ICC refusal on incomplete data that names the estimator that handles it.
- A vendored SummEval expert-coherence demo corpus and `load_demo()`, plus a runnable worked example in `examples/audit_summeval.py`.

### Numerical correctness

- Every statistic is pinned to an external reference: Krippendorff's alpha against the `krippendorff` package, ICC against the Shrout-Fleiss worked example and `pingouin`, and DIF against R `MASS::polr` with a `statsmodels` cross-check in the two-category limit. `pingouin` and `statsmodels` are test oracles only, never runtime imports.

[Unreleased]: https://github.com/brittanyreese/metajudge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/brittanyreese/metajudge/releases/tag/v0.1.0
