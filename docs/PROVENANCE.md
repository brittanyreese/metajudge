# Provenance

This file records why metajudge is built the way it is: the problem framing, the load-bearing conclusions from the prior-art survey, and the requirements they feed. The conclusions are summarized below; the requirements are in [SPEC.md](../SPEC.md); every decision that changes the build is a dated, cited ADR in [decisions/](decisions/); and the methods those decisions rest on are consolidated in [REFERENCES.md](REFERENCES.md).

## Load-bearing facts carried over

- The contribution is the integrated measurement-rigor surface (DIF across output strata + integrated validity), uncovered by shipped tools (AutoRubric and llm-judge-calibration ship reliability only).
- Adjacent methods work is active (Fair-IRT, fl-IRT-ing, two 2026 variance papers), so E07 ships the reliability + DIF slice first and keeps its method choices current with that literature.
- The full-instrument engine is MFRM (many-facet Rasch), the eventual E04 engine and the methods-paper headline. Python MFRM tooling is thin (FACETS proprietary; R via TAM/immer/sirt), so the MFRM engine is a real build, larger than E07.
- The library ships MIT-free. Its value is the published method and the demonstrated fit on real instruments.

## Research trail

- [DECISIONS.md](DECISIONS.md): the curated index of every build-changing ADR (corpus lock, ordinal-DIF engine, ICC refusal on incomplete data), each linked to its full record in [decisions/](decisions/).
- [REFERENCES.md](REFERENCES.md): the consolidated bibliography for every method the library implements.

The literature review behind the four-pillar framing is distilled into those decision records and bibliography: each method choice traces to the sources it rests on, and the honest limits of applying a method outside its original validation are stated in the relevant ADR.
