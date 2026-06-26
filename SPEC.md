---
date: 2026-06-21
topic: e04-judge-instrument-audit-layer
---

# metajudge — Requirements

## Summary

A pip-installable layer that audits an LLM judge/rubric instrument before its scores
are trusted, running on top of existing eval tools (DeepEval, AutoRubric, inspect_ai)
rather than competing with them. The full design has four pillars: reliability,
validity evidence, DIF across output strata, and variance decomposition, led by DIF
and integrated validity evidence. The library ships free under MIT. E07, the shipped
release, is the reliability + DIF report card; the remaining pillars and a many-facet
Rasch (MFRM) engine are the deferred E04 scope. A 2026-06-21 systematic literature
review found that MFRM unifies all four pillars in one model, which makes it the
natural engine for the full instrument.

## Problem Frame

LLM-as-judge is now a common way to score subjective model output, but the judge
itself is rarely audited as a measurement instrument. Shipped practitioner libraries
cover the commodity layer: AutoRubric and llm-judge-calibration ship reliability
statistics (kappa, ICC, Krippendorff's alpha) and stop there. DIF across output strata
and integrated statistical validity evidence have no shipped-library coverage; variance
decomposition exists only as paper methodology (arxiv 2604.11581). DeepEval
and inspect_ai audit the model-as-subject (type-X), not the judge instrument (type-Y),
so they do not address this surface.

That gap maps onto a psychometric skill set: rubric design is close to instrument
validation, variance decomposition to measurement-error modeling, inter-annotator
agreement to inter-rater reliability. metajudge packages the uncovered area plus the
four-pillar integration as the measurement-rigor layer the category is missing.

Adjacent methods work is active: Fair-IRT and fl-IRT-ing publish DIF /
measurement-invariance methods without shipping a practitioner library, and AutoRubric
is maintained. metajudge packages the DIF + validity surface those methods do not ship.

## Key Decisions

- **Fairness is the framing; the library ships MIT-free.** The fairness/bias tooling
  category already has free tools (holisticai, Giskard, langfair), so the library is
  not positioned as a revenue product. Clinical/health is a separate, reactive track
  on the same core, not a v1 surface.

- **The asset is a paper plus preprint alongside the library.** The pillar runs on real
  judge outputs are also the empirics for a methods paper and arXiv preprint, so the
  library build and the writeup share data rather than running a separate empirical
  campaign.

- **Serial sequencing.** E07 (the reliability + DIF report card) ships first as the
  smallest public artifact, then folds into E04 as a module. The full E04 instrument
  follows.

- **DIF is the lead pillar because it is native to fairness.** DIF originates in
  educational-test fairness; "DIF across output strata" and "bias across protected
  groups" are the same statistic. Leading with DIF keeps the novel pillar and the
  go-to vocabulary the same concept, with no translation tax.

- **MFRM is the unifying engine for the full instrument; E07 stays simple.** The
  literature review found many-facet Rasch measurement integrates all four pillars in
  one model (rater severity = reliability, infit/outfit = validity, interaction terms =
  DIF, variance components = variance), with a peer-reviewed AES precedent (Yamashita
  2025) but no Type-Y application. MFRM is the engine under the deferred E04, with the
  four-pillar report card as the accessible output layer. E07 stays reliability + DIF
  (simple, fast). G-theory (Gao 2025) is the companion variance method.

## Requirements

**Product / artifact**

R1. The artifact is a pip-installable layer on top of existing eval tools (DeepEval, AutoRubric, inspect_ai), not a competing framework.
R2. It implements four pillars: reliability, validity evidence, DIF across output strata, variance decomposition.
R3. It leads with DIF across strata plus integrated validity evidence, not the commoditized reliability core.
R4. It audits the judge/rubric instrument (type-Y), distinct from model-as-subject evaluation (type-X).

**Positioning / verticals**

R5. Fairness/bias is the launch framing; the library ships free under an MIT license.
R6. Clinical/health is a separate, reactive track on the same core, not a v1 product surface; no proactive clinical product is built.
R7. Differentiation is the four-pillar integration plus the DIF/validity surface and demonstrated fit on real instruments.

**Sequencing / shipping**

R8. E07 (the reliability + DIF report card) ships first as the smallest public artifact, then folds into E04 as a module.
R9. E04 full instrument follows E07. Its engine is MFRM (many-facet Rasch), with the four-pillar report card as the accessible output layer on top. MFRM tooling in Python is thin (FACETS is proprietary; the mature implementations are R: TAM, immer, sirt), so E04 wraps R via rpy2 or implements the estimation. This makes E04 a genuine methods-plus-engineering build, materially larger than E07.
R10. The library and the paper are one motion in the empirics sense: the pillar runs on real judge outputs are the paper's data, and no separate empirical campaign is run. The journal version is reframed in prose for the chosen venue with no new runs. One motion means shared data, not shared prose.

**Academic / paper leg**

R11. The work is documented as a methods paper plus an arXiv preprint alongside the library.
R12. The preprint ships at or near E07 and stays DIF + validity. The journal paper follows slowly in parallel, headlined by the MFRM port (the first Type-Y application; precedent Yamashita 2025, companion G-theory Gao 2025) and drawing on E04's MFRM results. The MFRM contribution is the journal's, not the preprint's.
R13. The paper stays off the critical path; it never delays E07. The preprint is fast and good-enough; the journal version is the slow tail.
R14. Venue is chosen at preprint time by the framing then in force. The candidate tracks and their journals are recorded in Outstanding Questions, not fixed here.

**Scope boundary**

R15. No paid product ships in v1; the library is MIT-free.
R16. Any future paid form is out of v1 scope.

**E07 first-artifact scope**

R17. E07 is reliability (Krippendorff's alpha / ICC) plus DIF on one stratum, packaged for pip with tests, a README, and one worked demo.
R18. Demo data reuses a public eval/judge corpus (e.g. HealthBench rubrics or an existing eval set) across strata, multi-run, never self-generated from scratch. Data-sourcing is the rabbit hole; reuse is the guard against it.
R19. The rigor pass, DIF specification correctness and alpha correctness, is non-compressible. A wrong DIF is reputational damage, so rigor is never the piece cut to hit the effort box.
R20. E07 ships with a preprint v1 draft as part of the artifact. A minimal v1 draft satisfies this; depth is the journal's job. The preprint is part of the artifact and is never traded against the rigor pass.

## Ship Gate and Stop Conditions

SC1. Effort-boxed ship gate.
- **Covers R8, R17–R20.** The gate is an effort box, not a calendar deadline: core code is about a weekend; the public artifact (code + reused data + rigor pass + preprint v1 draft) has a roughly 2-week target and a 3-week cap.
- **When** the box is at risk of slipping, **the cut order is** data-sourcing first (narrow the corpus), then thin the preprint draft toward its minimal v1 form. Never drop the preprint (R20) and never cut the rigor pass. Over-scoping the artifact is the failure mode the box guards against.
- **At ship time**, re-check actual prior art.

SC2. Prior-art trip with pre-ship re-verify.
- **Covers R3, R11.** This is the contribution's own falsification criterion.
- **When** a shipped, pip-installable practitioner library covers DIF across strata *and* integrated statistical validity evidence for an LLM-judge instrument before publication, **the response is** to drop the novel-method positioning and reframe to integration + clinical-fit. The work continues; the messaging changes.
- **Before publishing**, re-run the DIF+validity prior-art search in the days ahead of launch; if tripped, reframe the messaging before publishing, not after.

## Success Criteria

- E07 is public within the effort box (roughly 2-week target, 3-week cap) with a defensible two-pillar report card (reliability + DIF on one stratum, the R17 scope) on real judge outputs. The remaining pillars follow with E04.
- An arXiv preprint at or near E07 ship timestamps the integrated four-pillar tooling contribution and the DIF-across-output-strata operationalization. It does not claim the DIF-for-fairness concept, which is published prior art (fl-IRT-ing 2024, Fair-IRT); the timestamp defends the shipped-tooling slice, not the method.
- The build and the paper share one motion: shared empirics, no separate data campaign for the paper.

## Scope Boundaries

**Deferred for later**
- A hosted or certified clinical tier (regulated hosting, certification liability): needs proven repeat demand and is out of scope for this library.
- The journal paper's full polish: the slow tail after the preprint and E04.
- Any paid product surface: out of scope for this library.

**Outside this product's identity**
- E05 (evidence + PRO-validation kit) and E06 (power simulator): separate artifacts, separate scope.
- A hosted SaaS with auth, hosting, support, churn: out of scope for this library.
- A reliability-only library: walks into the commoditized room (AutoRubric, llm-judge-calibration).

## Dependencies / Assumptions

- **Adjacent methods work is active.** Fair-IRT, fl-IRT-ing, two 2026 variance papers, and an active AutoRubric move in nearby space. The sequencing keeps metajudge's method choices current with that literature rather than locking them from defaults.
- **The library build and the paper/preprint share one workspace** (R10), so the engine and its write-up move together rather than in separate repos. The research and document-output toolchain is provisioned in-repo.
- **LLM-eval is the chosen entry point for the four-pillar method.** The supporting analysis treats it as a hypothesis to test against real use, not a settled fact.
- **MFRM tooling is thin (feasibility, load-bearing for E04).** Python lacks a mature many-facet Rasch implementation; FACETS is proprietary, and the mature path is R (TAM / immer / sirt). E04's MFRM engine wraps R via rpy2 or implements the estimation, which raises E04's cost above the E07 estimate. E07 is unaffected (reliability + DIF only).

## Outstanding Questions

**Deferred to planning**
- Which public eval/judge corpus to reuse for the E07 demo (HealthBench rubrics, an existing eval set, or another), sourced, not generated. Gate: confirm the chosen corpus carries the strata and per-cell N for a defensible DIF before committing it. If it does not, that is a scope-cut trigger (drop to a stratum the data supports), never a rigor-cut.
- Package name, public API surface, repo structure.
- Preprint venue, chosen at ship time by the framing in force. Candidate tracks: methods (Psychometrika, Behavior Research Methods, Minds & Machines); NLP-eval (ACL/EMNLP findings or an eval workshop); clinical (npj Digital Medicine, JMIR); software-resource (JOSS alongside the library).
- How the variance module folds from standalone E07 into E04 without rework.

## Sources / Research

- The four-pillar framing rests on a systematic literature review (37 sources, 7 areas),
  kept as private working notes for the methods writeup.
- `docs/decisions/` — dated, cited ADRs where a research finding changes the build.
