# metajudge

Audit a scoring instrument, an LLM-as-judge or a human rater panel, before you trust its numbers.

[![CI](https://github.com/brittanyreese/metajudge/actions/workflows/ci.yml/badge.svg)](https://github.com/brittanyreese/metajudge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/metajudge)](https://pypi.org/project/metajudge/)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An LLM judge or scoring rubric is a measurement instrument. Before you report its numbers, you want to know whether the raters agree, whether scores drift between output types, and how large any drift is. `metajudge` answers those questions and prints a one-screen report card. The statistics are rater-agnostic: it audits any multi-rater ordinal panel, whether the raters are LLM judges or human annotators. It audits the scoring instrument, not the model under test.

It complements the ground-truth side of LLM evaluation (accuracy benchmarks, IRT-over-judges): those need a gold label per item, while `metajudge` works on *subjective* ordinal scores (coherence, helpfulness, quality) where no gold label exists and the question is whether the instrument is reliable and unbiased across output strata.

## Try it on the demo

The library ships a real corpus (SummEval expert coherence), so the example below runs end to end after install:

```bash
pip install metajudge
python -c "from metajudge import load_demo, audit; print(audit(load_demo(), focal='abstractive', reference='extractive').to_markdown())"
```

A runnable script is at [`examples/audit_summeval.py`](https://github.com/brittanyreese/metajudge/blob/main/examples/audit_summeval.py), with expected output in `examples/sample_output.txt`:

```python
from metajudge import load_demo, audit

ratings = load_demo()   # SummEval expert coherence: 1600 items, 3 expert raters,
                        # stratum = system family (extractive vs abstractive)
report = audit(ratings, focal="abstractive", reference="extractive")
print(report.to_markdown())
```

It prints the actual report card below (these are the live demo numbers, not a mock-up):

```
# metajudge report card

## Reliability
- Krippendorff's alpha (ordinal): 0.554 [95% CI 0.529, 0.578]
- ICC(2,1): 0.573; ICC(2,k): 0.801 (1600 targets x 3 raters)

## DIF (panel-relative, rest-score conditioner)
> Note: the rest-score conditioner cannot see bias shared across the entire rater panel, so this is panel-relative DIF, not an instrument-level fairness clearance. Pass an external quality conditioner to test for instrument-level bias.

- abstractive vs extractive (conditioner: rest_score, n=4800)
- Uniform DIF: chi2(1)=12.15, p=0.0005
- Nonuniform DIF: chi2(1)=0.17, p=0.6773
- Effect size (Nagelkerke R2 delta): 0.002 (Jodoin-Gierl class A)
```

## Audit your own judge

To audit a real instrument, point metajudge at the output of an existing judge runner. `Ratings.from_eval_instruments` maps the per-judge score frames produced by Epic's [`evaluation-instruments`](https://github.com/epic-open-source/evaluation-instruments) (`frame_from_evals`) into the `Ratings` the audit consumes, with rater = judge, item = sample, score = one rubric criterion. It is a local DataFrame transform that adds no dependency. A runnable, no-PHI walkthrough is in [docs/interop-epic.md](https://github.com/brittanyreese/metajudge/blob/main/docs/interop-epic.md).

## Cluster-robust DIF confidence intervals

The analytic likelihood-ratio test pools every (item, rater) cell as independent. In a crossed rater-by-item design that is anti-conservative: scores for the same item are correlated across raters. `cluster_bootstrap_dif` keeps the analytic point estimate and adds percentile confidence intervals (default 95%) for the effect size and the total-DIF chi-square by resampling whole item blocks. These are robustness intervals, not corrected p-values.

```python
from metajudge import load_demo, cluster_bootstrap_dif

ratings = load_demo()
# Each resample refits the engine, so runtime grows with n_boot and corpus size.
# The 1600-item demo takes about a minute at n_boot=200; the default is n_boot=1000.
cb = cluster_bootstrap_dif(
    ratings, focal="abstractive", reference="extractive", n_boot=200, seed=0
)
print(f"R² delta: {cb.base.nagelkerke_r2_delta:.3f}")
print(f"95% cluster CI: [{cb.r2_delta_ci_low:.3f}, {cb.r2_delta_ci_high:.3f}]")
print(f"CI reliable: {cb.ci_reliable}  (n_effective={cb.n_effective})")
```

Degenerate resamples (draws with no ordinal variation) are dropped; check `cb.ci_reliable` before reading the bounds. When fewer than 100 resamples survive, the bounds are indicative only and `cb.base` (the analytic estimate) is the honest number to report.

## How to read the report card

Reliability, Krippendorff's alpha and ICC. Both measure how much the raters agree. Krippendorff's convention treats alpha at or above 0.667 as tentatively reliable and 0.80 as reliable, so the demo's 0.554 sits below that floor: these coherence scores are only marginally reliable, which is the kind of result this tool exists to surface. ICC(2,k) is higher than ICC(2,1) because averaging three raters cancels some of the per-rater noise. The reliability estimators assume a complete crossed design; on a matrix with missing cells `icc` refuses and names the estimator that does handle incomplete data, rather than returning a number it cannot defend. The reasoning is recorded as a dated ADR.

DIF, differential item functioning. This asks whether abstractive outputs are scored differently from extractive outputs once you match on overall quality. The engine is ordinal logistic regression in the Zumbo tradition (single-pass, not lordif's iterative purification), run as three nested proportional-odds models, so it reports a uniform-DIF test, a nonuniform-DIF test, and an effect size (the Nagelkerke pseudo-R-squared change) classified A, B, or C by the Jodoin-Gierl thresholds. The demo shows why the card prints both a p-value and an effect size: at n = 4800 the uniform-DIF test is significant (p = 0.0005), but the effect size is 0.002, class A, which is negligible. The signal is detectable; the magnitude is not.

The matching variable is a leave-one-rater-out rest score across the three expert raters, which uses the same exchangeable-rater assumption as the reliability pillar. That rest score detects bias relative to the rater panel and understates bias the whole panel shares, so the card labels this path panel-relative DIF. For instrument-level bias, pass an explicit external quality conditioner: `audit(ratings, focal=..., reference=..., conditioner=...)` accepts a sample-id to quality-score mapping. Read the default DIF output as a screening audit, not a confirmatory significance claim. When a p-value lands near a decision threshold, `cluster_bootstrap_dif` runs the same engine with item-block resampling and returns a 95% cluster-robust interval alongside the unchanged point estimate.

## Scope, stated honestly

- It covers two pillars today: reliability (Krippendorff's alpha with a bootstrap CI, and ICC(2,1)/(2,k)) and DIF across one stratum. Validity and variance decomposition are a later phase and are not in this release.
- DIF is one stratum at a time. The analytic likelihood-ratio test pools observations as independent; in a crossed rater-by-item design that is anti-conservative. Use `cluster_bootstrap_dif` to get item-block-resampled confidence intervals alongside the analytic point estimate. The tool is a screen that flags instruments worth a closer look, not a final verdict.
- The demo numbers illustrate the report-card format on a real corpus. They are not a published claim about SummEval.

## Install

```bash
pip install metajudge
```

Requires Python 3.11 or later.

## Develop

```bash
uv sync               # install runtime + dev deps
uv run pytest         # tests (coverage runs automatically)
uv run ruff check .   # lint
uv run ruff format .  # format
uv run pyright        # strict type-check
```

## Numerical correctness

Every statistic is pinned to an external reference, not to internal consistency alone. A wrong reliability or DIF number is a wrong answer, so each test asserts against a value a trusted tool produced:

- Krippendorff's alpha is checked against the `krippendorff` package.
- ICC(2,1)/(2,k) is checked against the Shrout-Fleiss (1979) worked example and the `pingouin` ICC values, reproduced to six decimals.
- DIF is checked against R `MASS::polr`, the canonical proportional-odds fit, with an in-process cross-check against `statsmodels` logistic regression in the two-category limit. `statsmodels` and `pingouin` are test oracles only; they are never imported at runtime.

When a reference value and a literal disagree, the reference wins and the literal is corrected. Tolerances are not loosened to make a test pass.

## Citing

If you use metajudge in published work, cite it via the [`CITATION.cff`](https://github.com/brittanyreese/metajudge/blob/main/CITATION.cff) file (GitHub's "Cite this repository" generates APA and BibTeX from it). The methods the tool implements are credited to their original authors in [docs/REFERENCES.md](https://github.com/brittanyreese/metajudge/blob/main/docs/REFERENCES.md).

## License

MIT. See [LICENSE](https://github.com/brittanyreese/metajudge/blob/main/LICENSE). The bundled SummEval demo corpus is redistributed under its own MIT license; see [its source notice](https://github.com/brittanyreese/metajudge/blob/main/src/metajudge/data/SOURCE.md).

## Decisions and provenance

Every choice that changes the build is a dated, cited ADR. The curated index of what was decided and why is [docs/DECISIONS.md](https://github.com/brittanyreese/metajudge/blob/main/docs/DECISIONS.md): the ordinal-DIF engine, the ICC refusal on incomplete data, and the SummEval corpus lock. The why-this-build context is in [docs/PROVENANCE.md](https://github.com/brittanyreese/metajudge/blob/main/docs/PROVENANCE.md), and the full records live in [docs/decisions/](https://github.com/brittanyreese/metajudge/tree/main/docs/decisions/).
