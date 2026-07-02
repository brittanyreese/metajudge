# Roadmap

The phase plan behind metajudge. Scope, non-goals, and the ship gate live in [SPEC.md](../SPEC.md); design decisions are recorded as ADRs in [decisions/](decisions/). This document is the build sequence.

## E07: the two-pillar report card (first public artifact)

Ship a pip-installable layer that audits an LLM judge instrument with a two-pillar report card (reliability plus DIF on one stratum) on real multi-rater judge outputs.

Status: shipped. `v0.1.0` is tagged (see [CHANGELOG.md](../CHANGELOG.md)); the sections below record what shipped for that tag, and post-tag work is tracked in the CHANGELOG's Unreleased section.

### Architecture

A long-format `Ratings` model feeds two pillar modules:

- Reliability is the commodity layer. Krippendorff alpha wraps the vetted `krippendorff` package, with a bootstrap CI added on top, and ICC(2,1)/(2,k) is implemented from the Shrout-Fleiss two-way ANOVA.
- DIF is the lead pillar. A from-scratch ordinal proportional-odds logistic-regression implementation (Zumbo tradition) with Nagelkerke R-squared change classified on the Jodoin-Gierl thresholds, plus a cluster (item-block) bootstrap for clustered-error inference. Validated against R `MASS::polr` as the oracle.

A report module renders the pillars into a Markdown report card. The engine is built and locked against hand-derivable golden cases and reference-library differential tests first; the demo corpus is wired in last so data sourcing never blocks correctness.

### Constraints

- Python 3.11+. CI matrix: 3.11, 3.12, 3.13.
- Runtime dependencies limited to `numpy`, `pandas`, `scipy`, `krippendorff`. Any addition is justified in the PR. `pingouin` and `statsmodels` are dev/test oracles only, never runtime imports.
- DIF and alpha correctness are non-compressible (SPEC R19). The rigor tests are never weakened to fit scope.
- Type hints on every public function, explicit return types. No bare `except`, no mutable default args, no `import *`.

### Build sequence

1. Project and CI scaffold: `pyproject.toml`, src layout, smoke test, green CI on the full version matrix. (done)
2. `Ratings` data model: long-format (rater, item, stratum, score) plus the coder-by-unit matrix every pillar consumes. (done)
3. Reliability, Krippendorff alpha with a bootstrap CI. (done)
4. Reliability, ICC(2,1) and ICC(2,k) from the Shrout-Fleiss ANOVA. (done)
5. DIF, ordinal logistic regression with Jodoin-Gierl A/B/C classification and a cluster bootstrap, the lead pillar. (done)
6. Report-card renderer: pillars into a Markdown/text report. (done)
7. Public API surface: a single `audit(...)` entry point over the pillars. (done)
8. Demo on a real judge corpus plus README walkthrough. Gated (SPEC R18) on the corpus carrying multi-rater or multi-run judge scores with enough per-cell N to make the strata meaningful. (done, PR #2: SummEval expert coherence)

### Numerical correctness

Every statistic is pinned to an external reference, not just internal consistency: hand-derivable golden cases, differential tests against the reference libraries (`krippendorff`, R `irr`/`TAM`, statsmodels), and property tests with hypothesis. When a reference value and a literal disagree, the reference wins and the literal is corrected; assertion tolerances are not loosened to make a test pass.

## Ship tasks (E07 package finish)

Separate from the engine build, done as E07 is packaged for release:

- Coverage gate plus badges; finalize the README. (done, v0.1.0)
- `CHANGELOG.md` in Keep a Changelog format. (done, v0.1.0)
- A worked example (`examples/`) that runs the audit on the demo corpus. (done, v0.1.0)
- Tag `v0.1.0` on a clean `main`. (done)
- PyPI trusted publishing via OIDC (release workflow building with `uv build` on a `v*` tag): deferred until there is a reason to publish.

## Archived E04: the full instrument hypothesis

E04 would have added a validity pillar, variance decomposition, and a many-facet Rasch (MFRM) engine. That plan is archived, not deferred. The MFRM / IRT-over-judges paradigm it would implement is now well covered by active measurement-for-LLM-eval literature and tooling (py-irt, tinyBenchmarks, HELM-IRT), so building a second engine for it is low-leverage.

metajudge's durable role is the narrower one it already fills: a reliability plus DIF library for *subjective* multi-rater scoring, where no gold label exists and the question is whether the instrument is reliable and shows differential functioning across output strata. That niche is not covered by the ground-truth IRT tools.

Reviving E04 requires a new prior-art and leverage review; it is not an active next step.
