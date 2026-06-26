# Decisions

Every choice that changes the build is recorded as a dated, cited ADR in [decisions/](decisions/). This is the curated index: what was decided and why, newest first. The why-this-build context is in [PROVENANCE.md](PROVENANCE.md); the requirements those decisions feed are in [SPEC.md](../SPEC.md).

## E07 report card

Eval-instruments interop: map judge output to Ratings with rater = judge (2026-06-23). `Ratings.from_eval_instruments` ingests one `frame_from_evals` output per judge (Epic's evaluation-instruments) under the frame rater = judge, item = sample, score = one rubric criterion. Criteria are a separate facet, audited one at a time, never raters; a criteria-as-raters frame would measure internal consistency (Cronbach), not inter-rater reliability. A method-currency pass and adversarial review validated the frame against the ICC, MFRM, and G-theory literature. No dependency is added. [ADR](decisions/2026-06-23-e07-eval-instruments-interop.md)

ICC on incomplete data: refuse and name the right estimator (2026-06-23). Shrout-Fleiss ICC(2,1)/(2,k) is defined on a complete crossed design. For a matrix with missing cells, the correct estimand is a variance-components estimator (ten Hove et al. 2024), which is the deferred E04 variance pillar, so `icc` refuses rather than returns a biased listwise number, and the error names the correct method. A method-currency pass and adversarial review rejected building a lightweight irrNA/Ebel estimator now: its only oracle is too thin for a non-compressible reliability number, and the machinery belongs in E04. [ADR](decisions/2026-06-23-e07-icc-incomplete-data-refuse.md)

DIF engine: ordinal logistic regression, not Mantel-Haenszel (2026-06-22). The dichotomized Mantel-Haenszel pillar matched each unit on its own per-rater mean, so the matching variable contained the studied response and the analysis was circular (the verdict swung with the bin count). It is replaced by proportional-odds logistic-regression DIF (the Zumbo and lordif framework) conditioned on an independent leave-one-rater-out rest score, with the Nagelkerke R-squared change classified by the Jodoin-Gierl thresholds. [ADR](decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md)

Demo corpus lock: SummEval expert coherence (2026-06-22). The worked demo uses the SummEval expert-coherence subset, a real crossed subject-by-rater matrix that computes alpha, ICC, and DIF without generating any data. It was chosen over HealthBench because it ships a crossed structure with per-rater scores, so data sourcing never blocks the correctness work. [ADR](decisions/2026-06-22-e07-corpus-lock-summeval.md)

## Research behind the build

The four-pillar framing and the method choices rest on a literature review of the reliability, DIF, and rater-mediated-measurement methods the library implements; the sources are consolidated in [REFERENCES.md](REFERENCES.md). A research finding graduates into an ADR here only when it changes the build.
