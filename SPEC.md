---
date: 2026-06-21
topic: e04-judge-instrument-audit-layer
---

# metajudge: Requirements

## Summary

A pip-installable screening layer that audits subjective multi-rater ordinal scoring before its numbers are trusted, running on top of existing eval tools (DeepEval, AutoRubric, inspect_ai) rather than competing with them. The shipped artifact is the E07 reliability + DIF report card: Krippendorff's alpha, ICC for complete crossed designs, and ordinal DIF across one output stratum at a time. The original E04 four-pillar / many-facet Rasch (MFRM) full-instrument hypothesis is archived as historical product context, not an active roadmap commitment.

## Problem Frame

LLM-as-judge is now a common way to score subjective model output, but the judge itself is rarely audited as a measurement instrument. Shipped practitioner libraries cover the commodity layer: AutoRubric and llm-judge-calibration ship reliability statistics (kappa, ICC, Krippendorff's alpha) and stop there. DIF across output strata has little shipped-library coverage in the LLM-judge workflow. DeepEval and inspect_ai audit the model-as-subject (type-X), not the judge instrument (type-Y), so they do not address this surface.

That gap maps onto a psychometric skill set: rubric design is close to instrument validation, differential item functioning to fairness screening, and inter-annotator agreement to inter-rater reliability. metajudge packages the narrow uncovered surface it can defend today: reliability plus DIF for subjective ordinal panels where no gold label exists.

Adjacent methods work is active: Fair-IRT and fl-IRT-ing publish DIF / measurement-invariance methods without shipping this practitioner surface, and AutoRubric is maintained. metajudge packages a small, tested screening card rather than a comprehensive validation framework.

## Key Decisions

- Fairness is the framing; the library ships free under an MIT license. The fairness/bias tooling category already has free tools (holisticai, Giskard, langfair), so the library is not positioned as a revenue product. Clinical/health is a separate, reactive track on the same core, not a v1 surface.

- The paper and preprint are built alongside the library. Running the pillars on real judge outputs generates the empirics for both a methods paper and an arXiv preprint, so the library build and the writeup share data rather than running a separate empirical campaign.

- E07 is the public artifact. It does not imply a scheduled E04 follow-on.

- DIF is the lead pillar because it is native to fairness. DIF originates in educational-test fairness; "DIF across output strata" and "bias across protected groups" are the same statistic. Leading with DIF keeps the novel pillar and the go-to vocabulary the same concept, with no translation tax.

- Archived E04 hypothesis: many-facet Rasch measurement could unify a larger instrument (rater severity = reliability, infit/outfit = validity, interaction terms = DIF, variance components = variance). That remains useful research context, but the active product does not promise an MFRM engine.

## Requirements

### Product / artifact

R1. The artifact is a pip-installable layer on top of existing eval tools (DeepEval, AutoRubric, inspect_ai), not a competing framework.
R2. It implements two pillars today: reliability and DIF across output strata.
R3. It leads with DIF across strata, not the commoditized reliability core.
R4. It audits the judge/rubric instrument (type-Y), distinct from model-as-subject evaluation (type-X).

### Positioning / verticals

R5. Fairness/bias is the launch framing; the library ships free under an MIT license.
R6. Clinical/health is a separate, reactive track on the same core, not a v1 product surface; no proactive clinical product is built.
R7. Differentiation is the DIF screening surface, honest caveats, and demonstrated fit on real subjective scoring data.

### Sequencing / shipping

R8. E07 (the reliability + DIF report card) is the smallest public artifact and the active package identity.
R9. The E04 full-instrument/MFRM plan is archived. Reviving it requires a fresh prior-art and leverage review before implementation.
R10. The library and the paper are one motion in the empirics sense: the pillar runs on real judge outputs are the paper's data, and no separate empirical campaign is run. The journal version is reframed in prose for the chosen venue with no new runs. One motion means shared data, not shared prose.

### Academic / paper leg

R11. The work is documented as a methods paper plus an arXiv preprint alongside the library.
R12. Any preprint at or near E07 stays reliability + DIF and does not claim full instrument validation. A future MFRM paper is out of active scope unless E04 is explicitly revived.
R13. The paper stays off the critical path; it never delays E07. The preprint is fast and good-enough; the journal version is the slow tail.
R14. Venue is chosen at preprint time by the framing then in force. The candidate tracks and their journals are recorded in Outstanding Questions, not fixed here.

### Scope boundary

R15. No paid product ships in v1; the library is MIT-free.
R16. Any future paid form is out of v1 scope.

### E07 first-artifact scope

R17. E07 is reliability (Krippendorff's alpha / ICC) plus DIF on one stratum, packaged for pip with tests, a README, and one worked demo.
R18. Demo data reuses a public eval/judge corpus (e.g. HealthBench rubrics or an existing eval set) across strata, multi-run, never self-generated from scratch. Data-sourcing is the rabbit hole; reuse is the guard against it.
R19. The rigor pass, DIF specification correctness and alpha correctness, is non-compressible. A wrong DIF is reputational damage, so rigor is never the piece cut to hit the effort box.
R20. E07 ships with a preprint v1 draft as part of the artifact. A minimal draft satisfies this, depth is the journal's job, and it is never traded against the rigor pass.

## Ship Gate and Stop Conditions

SC1. Effort-boxed ship gate.
- Covers R8, R17-R20. The gate is an effort box, not a calendar deadline: core code is about a weekend; the public artifact (code + reused data + rigor pass + preprint v1 draft) has a roughly 2-week target and a 3-week cap.
- If the box is at risk of slipping, cut data-sourcing first (narrow the corpus), then thin the preprint draft toward its minimal v1 form. Never drop the preprint (R20) and never cut the rigor pass. Over-scoping the artifact is the failure mode the box guards against.
- At ship time, re-check actual prior art.

SC2. Prior-art trip with pre-ship re-verify.
- Covers R3, R11. This is the contribution's own falsification criterion.
- If a shipped, pip-installable practitioner library covers DIF across output strata for subjective LLM-judge panels before publication, drop the novel-method positioning and reframe to implementation quality plus interoperability. The work continues; the messaging changes.
- Before publishing, re-run the DIF prior-art search in the days ahead of launch; if tripped, reframe the messaging before publishing, not after.
- Re-check log (2026-07-02, multi-persona review, live search): **does not trip.** The closest work found, Choi et al. (2026, arXiv:2602.00521), diagnoses LLM-judge reliability via IRT/GRM and explicitly defers DIF-style stratum comparisons to future work. Fair-IRT (Xu et al. 2025) and fl-IRT-ing (Bachmann et al. 2024) apply IRT to fairness/bias measurement but not to DIF across LLM-judge output strata. No shipped, pip-installable practitioner library was found covering this surface; see `docs/REFERENCES.md` for full citations.

## Success Criteria

- E07 is public within the effort box (roughly 2-week target, 3-week cap) with a defensible two-pillar screening card (reliability + DIF on one stratum, the R17 scope) on real judge outputs.
- An arXiv preprint at or near E07 ship timestamps the DIF-across-output-strata operationalization for subjective LLM-judge panels. It does not claim full instrument validation or the DIF-for-fairness concept, which is published prior art (fl-IRT-ing 2024, Fair-IRT); the timestamp defends the shipped-tooling slice, not the method.
- The build and the paper share one motion: shared empirics, no separate data campaign for the paper.

## Scope Boundaries

### Deferred for later

- A hosted or certified clinical tier (regulated hosting, certification liability): needs proven repeat demand and is out of scope for this library.
- A full-instrument journal paper or MFRM engine: archived E04 scope, not active roadmap work.
- Any paid product surface: out of scope for this library.

### Outside this product's identity

- E05 (evidence + PRO-validation kit) and E06 (power simulator): separate artifacts, separate scope.
- A hosted SaaS with auth, hosting, support, churn: out of scope for this library.
- A reliability-only library: walks into the commoditized room (AutoRubric, llm-judge-calibration).

## Dependencies / Assumptions

- Adjacent methods work is active: Fair-IRT, fl-IRT-ing, variance papers, and an active AutoRubric move in nearby space. Any scope expansion re-checks that literature before committing to a new pillar.
- The library build and the paper/preprint share one workspace (R10), so the engine and its write-up move together rather than in separate repos. The research and document-output toolchain is provisioned in-repo.
- LLM-eval is the chosen entry point for the two-pillar screening card. The supporting analysis treats it as a hypothesis to test against real use, not a settled fact.
- MFRM tooling in Python remains thin (FACETS proprietary; mature paths in R such as TAM / immer / sirt), which is one reason the original E04 engine is archived rather than scheduled.

## Outstanding Questions

### Deferred to planning

- Which public eval/judge corpus to reuse for the E07 demo (HealthBench rubrics, an existing eval set, or another), sourced, not generated. Gate: confirm the chosen corpus carries the strata and per-cell N for a defensible DIF before committing it. If it does not, that is a scope-cut trigger (drop to a stratum the data supports), never a rigor-cut.
- Package name, public API surface, repo structure.
- Preprint venue, chosen at ship time by the framing in force. Candidate tracks: methods (Psychometrika, Behavior Research Methods, Minds & Machines); NLP-eval (ACL/EMNLP findings or an eval workshop); clinical (npj Digital Medicine, JMIR); software-resource (JOSS alongside the library).
- What evidence would justify reviving the archived E04 full-instrument scope.

## Sources / Research

- The two-pillar scope rests on a literature review across reliability, DIF, and rater-mediated-measurement methods.
- `docs/REFERENCES.md`: the consolidated bibliography for every method the library implements.
- `docs/decisions/`: dated, cited ADRs where a research finding changes the build.
