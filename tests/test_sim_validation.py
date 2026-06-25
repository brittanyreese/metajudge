"""Validation studies: Type-I/power, band calibration, conditioner comparison, bootstrap CI."""

from __future__ import annotations

import pytest
from sim.dgp import DgpParams
from sim.harness import run_cell, summarize_cell
from sim.validation import band_calibration_cells, conditioner_comparison_cells, type1_power_cells


def test_validation_module_imports() -> None:
    assert callable(type1_power_cells)
    assert callable(band_calibration_cells)
    assert callable(conditioner_comparison_cells)


def test_type1_control_quick() -> None:
    """Under H0 with no impact, Type-I rate must stay below 0.15 (n_reps=30, wide bound)."""
    params = DgpParams(n_items_per_group=100, n_raters=3, mu_focal=0.0)
    df = run_cell(params, n_reps=30, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.reject_total_rate < 0.15, (
        f"Type-I={summary.reject_total_rate:.3f} exceeds 0.15 under H0"
    )


def test_power_strong_dif_quick() -> None:
    """At dif_uniform=1.5, rejection rate must exceed 0.30 (n_reps=30)."""
    params = DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=1.5)
    df = run_cell(params, n_reps=30, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.reject_total_rate > 0.30, (
        f"Power={summary.reject_total_rate:.3f} below 0.30 at dif_uniform=1.5"
    )


@pytest.mark.slow
def test_type1_control_full() -> None:
    """All null cells must hold Type-I in [0.025, 0.075] at alpha=0.05 (n_reps=400).

    Bounds are nominal +/- 2 binomial SE: sqrt(0.05 * 0.95 / 400) = 0.0109; the interval
    [0.028, 0.072] rounds conservatively to [0.025, 0.075].
    """
    null_cells = [
        c for c in type1_power_cells() if c.dif_uniform == 0.0 and c.dif_nonuniform == 0.0
    ]
    for params in null_cells:
        df = run_cell(params, n_reps=400, base_seed=20260625)
        summary = summarize_cell(df)
        assert 0.025 <= summary.reject_total_rate <= 0.075, (
            f"Type-I out of [0.025, 0.075] for mu_focal={params.mu_focal}: "
            f"{summary.reject_total_rate:.3f}"
        )


@pytest.mark.slow
def test_power_full() -> None:
    """Power at dif_uniform=1.5 must exceed 0.80 (n_reps=400, n=100/group, 3 raters)."""
    params = DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=1.5)
    df = run_cell(params, n_reps=400, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.reject_total_rate > 0.80, (
        f"Power={summary.reject_total_rate:.3f} below 0.80 at dif_uniform=1.5"
    )


@pytest.mark.slow
def test_power_nonuniform_full() -> None:
    """Nonuniform DIF at dif_nonuniform=0.3 must be detected with power > 0.50 (n_reps=400)."""
    params = DgpParams(n_items_per_group=100, n_raters=3, dif_nonuniform=0.3)
    df = run_cell(params, n_reps=400, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.reject_nonuniform_rate > 0.50, (
        f"Nonuniform power={summary.reject_nonuniform_rate:.3f} below 0.50"
    )


def test_band_null_r2_in_A_band() -> None:
    """Under H0 (no DIF), mean R2_delta must stay below the A-band threshold 0.035."""
    params = DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=0.0)
    df = run_cell(params, n_reps=30, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.mean_r2_delta < 0.035, (
        f"Null mean_r2_delta={summary.mean_r2_delta:.4f} is not in A-band (< 0.035)"
    )


def test_band_r2_increases_with_dif() -> None:
    """Mean R2_delta at dif_uniform=1.5 must exceed mean R2_delta at dif_uniform=0."""
    r2_null = summarize_cell(
        run_cell(
            DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=0.0),
            n_reps=30,
            base_seed=20260625,
        )
    ).mean_r2_delta
    r2_strong = summarize_cell(
        run_cell(
            DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=1.5),
            n_reps=30,
            base_seed=20260625,
        )
    ).mean_r2_delta
    assert r2_strong > r2_null, f"R2 did not increase: null={r2_null:.4f} strong={r2_strong:.4f}"


@pytest.mark.slow
def test_band_r2_strict_monotone_full() -> None:
    """Mean R2_delta must be strictly increasing across all six band calibration cells.

    Uses n_reps=200 per cell.
    """
    cells = band_calibration_cells()
    r2_values: list[float] = []
    for idx, params in enumerate(cells):
        df = run_cell(params, n_reps=200, base_seed=20260625 + idx * 100_000)
        r2_values.append(summarize_cell(df).mean_r2_delta)
    for i in range(len(r2_values) - 1):
        assert r2_values[i] < r2_values[i + 1], (
            f"R2 not strictly monotone at step {i}: "
            f"dif={cells[i].dif_uniform} R2={r2_values[i]:.4f} >= "
            f"dif={cells[i + 1].dif_uniform} R2={r2_values[i + 1]:.4f}"
        )


@pytest.mark.slow
def test_band_strong_dif_reaches_C_band() -> None:
    """At dif_uniform=1.5, mean R2_delta must exceed the C-band threshold 0.070 (n_reps=200)."""
    params = DgpParams(n_items_per_group=100, n_raters=3, dif_uniform=1.50)
    df = run_cell(params, n_reps=200, base_seed=20260625)
    summary = summarize_cell(df)
    assert summary.mean_r2_delta > 0.070, (
        f"Strong DIF mean_r2_delta={summary.mean_r2_delta:.4f} did not reach C-band (> 0.070)"
    )
