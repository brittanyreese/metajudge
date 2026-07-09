# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-08

### Added

- Real-data DIF demonstrations on two public corpora. An ELLIPSE human-rater panel audit (`examples/audit_ellipse.py`) and a self-contained, dependency-free ELLIPSE LLM-judge path (`examples/audit_ellipse_llm.py`, `examples/_ellipse_judge.py`): the judge speaks the OpenAI `/v1/chat/completions` schema against any endpoint (local Ollama, `mlx_lm.server`, or a hosted API), with model, prompt, seed, and decoding pinned, and a committed qwen2.5:7b pilot reproduces the audit with no GPU or network. Decision record: `docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md`.
- SummEval consistency-DIF worked example across several rubric dimensions (`examples/audit_summeval_consistency.py`), with `scripts/prep_demo.py` gaining `--field` and `--multi-dim`. Decision record: `docs/decisions/2026-07-04-e07-summeval-consistency-dif.md`.
- Monte-Carlo derivation of the ordinal proportional-odds Jodoin-Gierl A/B/C bands (`scripts/derive_ordinal_bands.py`, raw draws in `sim/results/`), fitting cutoffs to the statistic this repo computes rather than inheriting the two-category logistic values. Decision record: `docs/decisions/2026-07-05-e07-ordinal-dif-band-derivation.md`, with a provenance test binding the shipped constants to the derivation summary.
- Two simulated teaching examples where the audit fires rather than clears: `examples/audit_catches_bias.py` (two panels with matched reliability, where only DIF separates the biased one) and `examples/audit_conditioner_choice.py` (the external conditioner catches a panel-shared bias the rest score misses).
- Collapse-mitigation knobs on the LLM judge (`reasoning` and `trait_scoped_anchors`, exposed as `--reasoning` / `--trait-scoped-anchors`) and a knob-attribution ablation (`examples/ablation_knob_attribution.py`). A live GPT-4o ablation attributes the score-collapse fix to the reasoning step (60% to 5% on its own, Fisher p=1.3e-7); trait-scoped anchors alone do not measurably help at this n (60% to 40%, p=0.12).

### Changed

- Jodoin-Gierl A/B/C thresholds moved from the two-category-calibrated 0.035/0.070 to the derived ordinal-PO 0.0376/0.0757. A score at the old boundary can shift class; no result this repo reports changes classification.
- The ELLIPSE LLM judge scores one rubric trait per call, replacing the single all-seven-traits response that induced a within-response halo. OpenRouter runs pin the provider route and log `system_fingerprint` per call.

### Fixed

- `Ratings` rejects a non-numeric score at the boundary instead of failing opaquely inside a pillar.
- `dif_class` degrades to `"?"` when the proportional-odds fit does not converge, rather than reporting a class read off a bad fit.
- `krippendorff_alpha` and `icc` guard two silent-NaN edge cases.
- The report card surfaces the small-N Jodoin-Gierl calibration caveat when n is below the band's calibration floor.

## [0.2.1] - 2026-07-03

### Changed

- Documentation and release metadata only, no code changes since 0.2.0: the AI-assistance note now leads with external-oracle validation, the citation metadata carries an ORCID, and the release is archived on Zenodo.

## [0.2.0] - 2026-07-02

### Added

- Operating-characteristics study for the DIF pillar (`docs/sim-operating-characteristics.md`, raw draws in `sim/results/`, runner `scripts/run_oc_study.py`): baseline Type-I/power, cluster-stress cells showing the analytic test does not inflate in this crossed design, PO-violation robustness, a power curve, an unbalanced-groups check, and the conditioner-overlap calibration curve. Simulation harness gains per-replication overlap diagnostics and unbalanced-group support (`DgpParams.n_items_focal`).
- Weekly `Rigor` CI workflow: runs the full-precision slow tests and the live R `MASS::polr` oracle refit that the fast PR suite deliberately skips. Releases (`publish.yml`) now gate on the same combination.
- The README shows the committed live Gemini judge-panel report card inline next to the SummEval card.

- `IccResult` now carries McGraw & Wong (1996) exact F-based 95% confidence intervals for ICC(2,1) and ICC(2,k) (`icc1_ci_low/high`, `icck_ci_low/high`), matching pingouin's `ICC(A,1)`/`ICC(A,k)` bounds. The report card renders them, so the reliability pillar no longer ships a bare point estimate.
- `cluster_bootstrap_dif` now reports bias-corrected and accelerated (BCa, Efron 1987) confidence intervals when they are computable and affordable, exposed via `ClusterBootstrapDif.ci_method` (`"bca"` or `"percentile"`). The acceleration comes from a leave-one-cluster-out jackknife; BCa is gated to samples where it both helps most and is cheap (jackknife no larger than the bootstrap), and falls back to the percentile interval otherwise. Validated against `scipy.stats.bootstrap(method="BCa")`.
- `holm_adjust`: Holm-Bonferroni familywise-error correction for screening DIF across multiple stratum pairs, exported at the top level. Reproduces `statsmodels multipletests(method="holm")`.
- `DifResult.conditioner_is_external` property: returns `True` when the conditioner came from an external source rather than the panel rest score. Previously this test required comparing `dif.conditioner_source == "external"` by hand.
- `_brant_test` now short-circuits on near-singular Fisher information (condition number > 1e10), returning `converged=False` instead of passing a large pseudoinverse through the Wald statistic and producing a spurious `po_violation` signal.
- `DifResult` gains `conditioner_group_corr`, `conditioner_common_support`, and `conditioner_overlap_weak`, a per-run diagnostic for the nested-strata conditioner/group confound. When `conditioner_overlap_weak` fires, `ReportCard.to_markdown` swaps the blanket nested-strata caveat for a run-specific warning naming the correlation and common-support numbers. See `docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md`.
- `LevelOfMeasurement` (a `Literal["nominal", "ordinal", "interval", "ratio"]`) is now exported at the top level and `krippendorff_alpha`/`audit`'s `level` parameter is validated against it with a clear `ValueError`, instead of a bare `str` passthrough that surfaced an opaque `TypeError` from inside the `krippendorff` package on a typo.
- The reliability block of the report card now carries a validity caveat: high agreement (alpha, ICC) is not evidence the rubric measures the intended construct.

### Changed

- `conditioner_overlap_weak` recalibrated: the threshold moved from the 0.7 convention to a measured 0.2 (false Jodoin-Gierl B/C rate under a true null: 0.5% at |corr| <= 0.2, 17% in (0.2, 0.4], above 50% past 0.4; the 0.7 flag almost never fired even in cells with a 100% false-DIF rate). Runs observing |corr| in [0.2, 0.7) now warn where they previously read as clean, and the report-card warning cites the calibrated band. See `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md`.
- The clustering-robust DIF flag caveat now names the CI method: for BCa it notes the bias correction; for the percentile fallback it keeps the boundary-fragility warning (the 0-bounded R²-change makes the percentile method least accurate near the Jodoin-Gierl boundary). `ClusterBootstrapDif` documents both.
- `logistic_dif` documents two scope limitations: the conditioner enters linearly (residual confounding under a nonlinear quality-response relationship) and no familywise correction is applied across multiple stratum pairs.
- `logistic_dif`'s unknown-stratum error now lists the available (stringified) levels and notes labels are matched as strings, so integer stratum labels no longer fail opaquely.
- `Ratings.wide()` uses `DataFrame.pivot` (raises on duplicate item-rater cells) instead of `pivot_table` (silent mean), keeping the one-cell-per-pair invariant on every construction path.
- `krippendorff_alpha` documents that its bootstrap resamples units, not raters, so the CI understates panel-sampling uncertainty for a small panel.
- Tightened DIF report language: external conditioners are now labeled as external-conditioner DIF and carry an explicit validity caveat instead of being treated as automatic instrument-level evidence.
- `Ratings.from_long` now refuses duplicate item-rater cells and missing stratum values rather than silently averaging duplicates or carrying null strata into DIF.
- `cluster_bootstrap_dif` now drops non-converged bootstrap refits from confidence intervals and reflects the surviving count in `n_effective`. CI bounds are `nan` when no resamples survive (previously collapsed to the point estimate).
- Alpha CI warnings now surface thin bootstrap intervals even when no degenerate resamples were dropped.
- `Flags` dataclass simplified: `converged` and `po_violation` fields removed (use `card.dif.converged` and `card.dif.po_violation` directly). The `conditioner_is_external` and `alpha_ci_degraded` fields remain.
- `dif.py`'s module docstring now states the design inversion up front (group is an item-level property, strata are disjoint item sets, the conditioner matches between nested sets) instead of leaving it to the ADR alone.
- `_classify_jodoin_gierl`'s docstring now discloses that the Jodoin & Gierl (2001) thresholds were calibrated on the two-category logistic R-squared, and applying them to the ordinal proportional-odds change follows lordif convention rather than a direct validation.
- `Ratings.__init__` now documents that it does no validation and points callers at `from_long`/`from_eval_instruments`.

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

[Unreleased]: https://github.com/brittanyreese/metajudge/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/brittanyreese/metajudge/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/brittanyreese/metajudge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brittanyreese/metajudge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brittanyreese/metajudge/releases/tag/v0.1.0
