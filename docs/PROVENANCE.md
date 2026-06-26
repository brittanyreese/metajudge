# Provenance

This build came out of a prior direction-decision analysis. That analysis (problem framing, claims, prior-art survey, and a structured review) lives in a separate working project and is referenced, not copied, so this repo stays a clean product repo. The load-bearing conclusions it produced are recorded below; the requirements they fed into are in [SPEC.md](../SPEC.md), and decisions that change the build are captured as dated ADRs in [decisions/](decisions/).

## Load-bearing facts carried over

- The contribution is the integrated measurement-rigor surface (DIF across output strata + integrated validity), uncovered by shipped tools (AutoRubric and llm-judge-calibration ship reliability only).
- Adjacent methods work is active (Fair-IRT, fl-IRT-ing, two 2026 variance papers), so E07 ships the reliability + DIF slice first and keeps its method choices current with that literature.
- The full-instrument engine is MFRM (many-facet Rasch), the eventual E04 engine and the methods-paper headline. Python MFRM tooling is thin (FACETS proprietary; R via TAM/immer/sirt), so the MFRM engine is a real build, larger than E07.
- The library ships MIT-free. Its value is the published method and the demonstrated fit on real instruments.

## Research trail

- [DECISIONS.md](DECISIONS.md): the curated index of every build-changing ADR (corpus lock, ordinal-DIF engine, ICC refusal on incomplete data), each linked to its full record in [decisions/](decisions/).

The systematic literature review behind the four-pillar framing, and the earlier method-exploration notes (corpus scans, DIF method iterations, the citation audit), are kept as private working notes for the methods writeup, outside the published tree.
