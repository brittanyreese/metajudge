# tests/test_examples.py
"""Guards against examples/README-committed output drifting from what the code prints.

Regression test for the A2 review finding: README's demo block, examples/sample_output.txt,
and docs/interop-epic.md all fell behind main after a caveat-changing commit landed. This
runs the example and diffs its stdout against the committed sample, so that class of drift
fails CI instead of shipping silently.

Also pins the E07 SummEval consistency/fluency/relevance DIF finding
(docs/decisions/2026-07-04-e07-summeval-consistency-dif.md): a pre-registered family of
three dimensions, LODO external conditioner, Holm-corrected. The fast tests use the
analytic-only path (`logistic_dif`); the cluster-bootstrap CI and the full-script
output-drift check both call `audit(robust=True, n_boot=400)` on the full 4,800-observation
corpus, which takes roughly two minutes per dimension, so those two are `@pytest.mark.slow`
and run only under `--run-slow` (docs/decisions/2026-07-03-ci-slow-suite-performance.md).
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest
from examples.audit_summeval_consistency import (
    AUDITED_DIMS,
    FOCAL,
    N_BOOT,
    REFERENCE,
    SEED,
    build_ratings,
    load_data,
    lodo_conditioner,
)

from metajudge.dif import cluster_bootstrap_dif, holm_adjust, logistic_dif

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# The cluster bootstrap's per-resample convergence is sensitive to platform-level
# floating-point/BLAS differences (e.g. macOS vs Linux CI runners), so n_effective can
# drift by a resample or two even with a fixed seed; normalize it rather than pin an
# exact count. This mirrors the already-documented n_effective drift nit (200 -> 199
# between v0.1.0 and HEAD). Matches both the `n_effective=X of Y` phrasing
# (examples' own print statements) and `to_markdown()`'s `X/Y resamples` phrasing.
_N_EFFECTIVE_RE = re.compile(r"n_effective=\d+ of \d+|\d+/\d+ resamples")

# Fit- and bootstrap-derived floats (alpha, ICC, effect sizes, p-values, CI bounds) vary
# in low-order digits across BLAS backends (macOS Accelerate vs Linux OpenBLAS), so these
# output-drift guards check structure and prose, not exact stochastic values; numeric
# correctness is pinned separately in the oracle-backed statistic tests.
_FLOAT_RE = re.compile(r"\d+\.\d+")


def _normalize(text: str) -> str:
    masked = _N_EFFECTIVE_RE.sub("n_effective=<n> of <n_boot>", text.strip())
    return _FLOAT_RE.sub("<n>", masked)


def test_audit_summeval_output_matches_committed_sample(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(_EXAMPLES_DIR / "audit_summeval.py"), run_name="__main__")
    captured = capsys.readouterr()
    expected = (_EXAMPLES_DIR / "sample_output.txt").read_text()
    assert _normalize(captured.out) == _normalize(expected)


def test_consistency_primary_pinned_result() -> None:
    """Pins the primary pre-registered dimension: a decisive nonuniform DIF that is
    nonetheless Jodoin-Gierl class A (negligible effect size). Fast, analytic-only path
    (no cluster bootstrap); see test_consistency_cluster_bootstrap_pinned (slow) for the
    cluster-robust confirmation.
    """
    df = load_data()
    ratings = build_ratings(df, "consistency")
    conditioner = lodo_conditioner(df, "consistency")
    result = logistic_dif(ratings, focal=FOCAL, reference=REFERENCE, conditioner=conditioner)

    assert result.converged
    assert result.dif_class == "A"
    assert result.p_nonuniform < 1e-3
    assert result.nagelkerke_r2_delta == pytest.approx(0.0146, abs=5e-4)
    assert result.p_uniform == pytest.approx(0.19, abs=0.01)
    assert result.conditioner_group_corr == pytest.approx(0.153, abs=0.01)
    assert not result.conditioner_overlap_weak


def test_fluency_secondary_pinned_result() -> None:
    """Pins the secondary pre-registered dimension."""
    df = load_data()
    ratings = build_ratings(df, "fluency")
    conditioner = lodo_conditioner(df, "fluency")
    result = logistic_dif(ratings, focal=FOCAL, reference=REFERENCE, conditioner=conditioner)

    assert result.converged
    assert result.dif_class == "A"
    assert result.p_nonuniform == pytest.approx(0.0026, abs=5e-4)
    assert result.nagelkerke_r2_delta == pytest.approx(0.0028, abs=3e-4)


def test_relevance_exploratory_pinned_result() -> None:
    """Pins the third pre-registered (exploratory-role) dimension."""
    df = load_data()
    ratings = build_ratings(df, "relevance")
    conditioner = lodo_conditioner(df, "relevance")
    result = logistic_dif(ratings, focal=FOCAL, reference=REFERENCE, conditioner=conditioner)

    assert result.converged
    assert result.dif_class == "A"
    assert result.p_uniform == pytest.approx(0.0005, abs=3e-4)
    assert result.nagelkerke_r2_delta == pytest.approx(0.0025, abs=3e-4)


def test_summeval_family_holm_correction_keeps_significance() -> None:
    """Holm-adjusting p_total across the 3-dimension family does not erase significance."""
    df = load_data()
    p_totals: list[float] = []
    for dim in AUDITED_DIMS:
        ratings = build_ratings(df, dim)
        conditioner = lodo_conditioner(df, dim)
        result = logistic_dif(ratings, focal=FOCAL, reference=REFERENCE, conditioner=conditioner)
        p_totals.append(result.p_total)
    p_holm = holm_adjust(p_totals)
    assert all(p < 0.01 for p in p_holm)


@pytest.mark.slow
def test_consistency_cluster_bootstrap_pinned() -> None:
    """Cluster-robust confirmation of the primary result (slow: ~2 min, --run-slow only)."""
    df = load_data()
    ratings = build_ratings(df, "consistency")
    conditioner = lodo_conditioner(df, "consistency")
    boot = cluster_bootstrap_dif(
        ratings,
        focal=FOCAL,
        reference=REFERENCE,
        conditioner=conditioner,
        n_boot=N_BOOT,
        seed=SEED,
    )
    assert boot.ci_reliable
    assert boot.r2_delta_ci_low == pytest.approx(0.0063, abs=0.003)
    assert boot.r2_delta_ci_high == pytest.approx(0.0295, abs=0.005)


@pytest.mark.slow
def test_audit_summeval_consistency_output_matches_committed_sample(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output-drift guard for examples/audit_summeval_consistency.py (slow: runs all 3
    dimensions' cluster bootstraps, ~5-6 min; --run-slow only)."""
    runpy.run_path(str(_EXAMPLES_DIR / "audit_summeval_consistency.py"), run_name="__main__")
    captured = capsys.readouterr()
    expected = (_EXAMPLES_DIR / "sample_output_summeval_consistency.txt").read_text()
    assert _normalize(captured.out) == _normalize(expected)
