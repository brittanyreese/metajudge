# Provenance

This file records why metajudge is built the way it is: the problem framing, the load-bearing conclusions from the prior-art survey, and the requirements they feed. The conclusions are summarized below; the requirements are in [SPEC.md](../SPEC.md); every decision that changes the build is a dated, cited ADR in [decisions/](decisions/); and the methods those decisions rest on are consolidated in [REFERENCES.md](REFERENCES.md).

## Load-bearing facts carried over

- The contribution is the narrow measurement-rigor surface metajudge can defend today: reliability plus DIF across output strata for subjective multi-rater ordinal scoring.
- Adjacent methods work is active (Fair-IRT, fl-IRT-ing, variance papers, and judge-calibration tooling), so E07 keeps its method choices current with that literature.
- The original full-instrument / MFRM (many-facet Rasch) plan is archived product context, not an active roadmap commitment. Python MFRM tooling is thin (FACETS proprietary; R via TAM/immer/sirt), which is part of why that engine is not scheduled here.
- The library ships MIT-free. Its value is the published method and the demonstrated fit on real instruments.

## Research trail

- [DECISIONS.md](DECISIONS.md): the curated index of every build-changing ADR (corpus lock, ordinal-DIF engine, ICC refusal on incomplete data), each linked to its full record in [decisions/](decisions/).
- [REFERENCES.md](REFERENCES.md): the consolidated bibliography for every method the library implements.

The literature review behind the two-pillar scope is distilled into those decision records and bibliography: each method choice traces to the sources it rests on, and the honest limits of applying a method outside its original validation are stated in the relevant ADR.
