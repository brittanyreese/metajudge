# CI: parallelize the Rigor slow suite, accept the two-core ceiling

Date: 2026-07-03
Status: accepted

## Context

The Rigor workflow runs the slow leg the fast PR suite skips: the full-precision
operating-characteristics cells (`@pytest.mark.slow`, several at `n_reps=400`, two
bootstrap cells at `n_reps=100` by `n_boot=200`) and the live R `MASS::polr` oracle
refit. `publish.yml` runs the same combination as a release gate, so a release pays
this cost twice, once in the pre-tag Rigor run and once at tag time before the PyPI
upload.

The first Rigor run to ever execute this suite (the v0.2.0 release branch) took 77
minutes (4624 seconds) in the `pytest --run-slow` step. It also surfaced a stale
test literal (a nonuniform-power bound of 0.50 that contradicted the study's own
recorded 0.468), fixed separately. With the literal corrected the suite is green, but
77 minutes twice per release is a slow gate.

The replications are embarrassingly parallel: each draws its sample from an
independent seed (`base_seed + rep`), so distributing cells across cores is
bit-identical to the serial run, not just statistically close.

## What was measured

Three release-branch Rigor runs, same commit lineage, `pytest --run-slow` step only:

| Configuration | Duration |
|---|---|
| Serial | 77:04 |
| `pytest-xdist -n auto` | 68:48 |
| `-n auto` plus BLAS threads pinned to 1 | 58:01 |

`-n auto` alone bought almost nothing. The cause is thread oversubscription: numpy
and scipy already spread each fit across both cores through OpenBLAS, so the xdist
worker processes competed with the BLAS threads for the same two cores. Pinning
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1 gives each
worker a real core. Single-threaded BLAS is also more reproducible than multi-threaded
BLAS, and the DGP matrices are small enough that per-fit BLAS parallelism was worth
little anyway.

The gain stops at roughly 1.3x, not the 2x two cores would suggest, because pinning
also slows each individual fit, and the two effects partly cancel. A local run on a
many-core machine finished the same suite in 21:16. That figure is the practical
floor: xdist cannot split a single test item, so the suite cannot drop below the
runtime of its longest cell, one of the two bootstrap tests, no matter how many cores
are added.

## Decision

1. Run the two slow legs (`rigor.yml`, `publish.yml`) under `pytest-xdist -n auto`
   with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` set to 1.
2. Stay on the standard two-core `ubuntu-latest` runner. A four-core larger runner
   would reach roughly 30 minutes and an eight-core one roughly 22, but both bill per
   minute, and a non-default runner label breaks CI for anyone who forks the
   repository. Portable, free CI is worth more here than a faster gate no reader of
   the repository ever waits on.
3. Do not split the bootstrap cells to beat the 21-minute floor. That means moving
   replication parallelism inside `run_cell_bootstrap`, which touches the numerical
   harness for a gain the two-core runner cannot use anyway.
4. `n_reps` and `n_boot` are load-bearing precision parameters and are not reduced to
   save time.

## Consequences

- The release gate drops from 77 to 58 minutes, still green, still bit-identical to
  the serial run. Reproducibility improves: single-threaded BLAS removes the last
  source of run-to-run float variation.
- CI stays runnable by anyone who clones the repository. The speed left on the table
  is a runner-size choice, recorded here rather than taken, with the numbers to
  revisit it if the suite ever gates something a human waits on.
- The shipped wheel is untouched. `sim/` and `tests/` are sdist-only, and
  `pytest-xdist` is a dev dependency, so nothing here reaches a PyPI install.
  `logistic_dif` at about 27 ms per call and `krippendorff_alpha` at about 88 ms need
  no optimization for a tool run once against an instrument.

## References

- .github/workflows/rigor.yml, .github/workflows/publish.yml (the two slow legs).
- sim/harness.py (`run_cell`, `run_cell_bootstrap`, the independently seeded loops).
- docs/sim-operating-characteristics.md (what the slow cells compute).
