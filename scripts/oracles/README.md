# DIF oracle provenance

The DIF pillar's pinned reference constants live in `tests/test_dif.py`
(`_OR_*` for the external-conditioner path, `_RS_*` for the leave-one-rater-out
rest-score path). SPEC R19 makes DIF numerical correctness non-compressible and
`CONTRIBUTING.md` requires every reported number to be regenerable from a clean
clone. These two files are that recipe.

| File | Role |
| --- | --- |
| `gen_olr_oracle.py` | Reproduces the pinned constants and falsifies them against R. |
| `fit_polr_dif.R` | Canonical proportional-odds fit (R `MASS::polr`) of the three nested DIF models. |

## What is reproduced vs frozen

- **`_QUALITY` (conditioner)** is regenerated from its seed,
  `numpy.random.default_rng(20260622).normal(size=24)`, and checked against the
  literal. It is fully reproducible.
- **`_SCORES` (responses)** is the frozen "scored once" fixture. Its canonical
  home is `tests/test_dif.py`; the harness reads it from there (single source of
  truth, no duplicated literals). The original stochastic scoring is intentionally
  frozen so the oracle constants stay stable across runs and platforms.
- **`_OR_*` / `_RS_*`** are reproduced by handing the engine's own designs
  (z-scored conditioner, focal indicator, interaction) to `MASS::polr` and
  comparing its likelihood-ratio chi-squares, p-values, and Nagelkerke R² change
  to the pinned literals, at the same tolerances `tests/test_dif.py` asserts.

If a regenerated constant ever disagrees with a pinned literal, the reference
(`polr`) wins and the literal must be corrected — never loosen the tolerance
(numerical-reference convention, `AGENTS.md`).

## Running

```bash
uv run python scripts/oracles/gen_olr_oracle.py
```

Exit codes: `0` reproduced, `1` a constant disagrees, `2` R/`MASS` unavailable
(the Python-side seed and design checks still run).

## Requirements

R with the `MASS` package:

```r
install.packages("MASS")
```

`tests/test_oracle_provenance.py` runs this harness in CI when `Rscript` and
`MASS` are present, and skips cleanly when they are not (R is not a project
runtime or CI dependency — the engine itself is scipy-only).
