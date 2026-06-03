# metajudge: project context

Pip-installable layer that audits an LLM judge/rubric instrument before it ships: a
reliability + DIF report card (E07), growing into a four-pillar instrument (E04).
Intent, scope, and the ship gate live in `SPEC.md`; the phase plan in
`docs/roadmap.md`. This file is the canonical, shareable working context.

## Commands

The repo uses uv. CI runs these exact steps (`.github/workflows/ci.yml`); match them
locally before pushing:

```bash
uv sync --extra dev          # install runtime + dev deps
uv run ruff check .          # lint
uv run ruff format --check . # format (drop --check to apply)
uv run pyright               # type-check (strict)
uv run pytest                # tests (auto-runs coverage: --cov=metajudge)
```

Pre-commit mirrors lint/format: `uv run pre-commit install` once, then it runs on
every commit.

## Git workflow

Full conventions are in [CONTRIBUTING.md](CONTRIBUTING.md). In short:

- Conventional Commits. One logical change per commit; keep formatting-only
  changes in their own `style:` commit (never bundle a repo-wide reformat into a
  feature commit).
- Trunk-based: `main` stays green and releasable. Small changes go straight to
  `main`; non-trivial work uses a short-lived branch merged back promptly so
  `main` never goes stale.
- Pre-commit and CI must pass before merging to `main`.
- Never commit scaffolding or secrets; the pre-commit guard
  (`scripts/check_no_scaffolding.sh`) blocks them. `AGENTS.md` is the shareable
  context; the gitignored `CLAUDE.md` is private.
- Tag releases `vX.Y.Z`. For any reported number, tag the exact commit that
  produced it so it is citeable.

## Architecture

- `src/metajudge/` (src layout, package `metajudge`, build backend hatchling).
  - `data.py`: `Ratings` long-format data model (rater, item, stratum, score).
  - `reliability.py`: Krippendorff alpha (+ bootstrap CI), ICC(2,1)/(2,k) via
    Shrout-Fleiss ANOVA.
  - `dif.py`: ordinal proportional-odds logistic-regression DIF (Zumbo) with
    Jodoin-Gierl A/B/C classification and a cluster bootstrap.
- `tests/`: one test module per statistic (alpha, ICC, DIF, data), plus
    `test_smoke.py`. Note `reliability.py` spans `test_alpha.py` + `test_icc.py`.
- `docs/`: `roadmap.md` (phase plan), `PROVENANCE.md`, and `decisions/` (dated,
  cited ADRs where a research finding changes the build).

Python 3.11+ (CI matrix 3.11/3.12/3.13). Strict pyright over `src` and `tests`.
Ruff line length 100, rule set `E,F,I,UP,B,SIM,RUF`.

## Binding rules (do not violate)

- Numerical correctness is non-compressible (SPEC R19). A wrong DIF or alpha is
  reputational damage. Test every statistic against a pinned external reference (the
  `krippendorff` package, R `irr`/`TAM` fixtures, statsmodels), not just internal
  consistency. Never cut the rigor pass to save time.
- Runtime dependencies are limited to `numpy`, `pandas`, `scipy`, `krippendorff`.
  Any addition is justified in the PR. `pingouin` and `statsmodels` are dev/test
  oracles only, never runtime imports.
- Type hints on every public function, explicit return types (pyright runs strict).
  No bare `except`, no mutable default args, no `import *`.
- Test-first for every feature and fix. For a stats function the failing-test target
  is a pinned external reference value, not internal consistency: write the
  reference-value assertions against a known oracle up front, watch them fail, then
  implement until they pass.
- Git: Conventional Commits.

## Gotchas

- The numerical-reference convention is the shape of the test loop here. When a
  reference value and a literal disagree, the reference wins and the literal is
  corrected; assertion tolerances are not loosened to make a test pass.
- `--cov` is wired into pytest `addopts`, so a bare `uv run pytest` already produces
  coverage.

## Methods are provisional

Method and library choices for later pillars (validity, variance decomposition, the
MFRM engine in E04) are validated against current literature before a phase plan
locks them, not assumed from defaults. The standing decisions live in
`docs/decisions/`; the supporting reviews are kept as private working notes.
