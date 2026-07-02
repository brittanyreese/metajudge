"""Tests for sim.oc_study: block specs and the block runner (not the full study)."""

from __future__ import annotations

import pandas as pd
import pytest
from sim.dgp import DgpParams
from sim.oc_study import Block, run_block, study_blocks


def test_study_blocks_all_covers_every_substudy() -> None:
    blocks = study_blocks("all")
    names = [b.name for b in blocks]
    assert names == [
        "s1_baseline",
        "s2_cluster_stress_analytic",
        "s2_cluster_stress_bootstrap",
        "s3_overlap_rest",
        "s3_overlap_external_degraded",
        "s4_po_robustness",
        "s5_sample_size",
        "s5_unbalanced",
    ]
    # Disjoint seed blocks so no two blocks share a replication stream.
    seeds = [b.base_seed for b in blocks]
    assert len(set(seeds)) == len(seeds)


def test_study_blocks_single_study_selection() -> None:
    blocks = study_blocks("s3")
    assert [b.name for b in blocks] == ["s3_overlap_rest", "s3_overlap_external_degraded"]
    assert all(b.kind == "analytic" for b in blocks)
    assert blocks[0].conditioner == "rest_score"
    assert blocks[1].conditioner == "external"


def test_study_blocks_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown study"):
        study_blocks("s9")


def test_run_block_analytic_smoke() -> None:
    block = Block(
        name="smoke",
        cells=[DgpParams(n_items_per_group=40, n_raters=3)],
        conditioner="external",
        kind="analytic",
        base_seed=1,
    )
    df = run_block(block, n_reps=2, boot_reps=2, n_boot=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # Tidy: cell params and block provenance ride along with every row.
    for col in ("cell", "n_items_per_group", "conditioner", "block"):
        assert col in df.columns
    assert set(df["block"]) == {"smoke"}
    assert set(df["conditioner"]) == {"external"}


def test_run_block_bootstrap_smoke() -> None:
    block = Block(
        name="smoke_boot",
        cells=[DgpParams(n_items_per_group=40, n_raters=3)],
        conditioner="external",
        kind="bootstrap",
        base_seed=1,
    )
    df = run_block(block, n_reps=400, boot_reps=2, n_boot=50)
    assert len(df) == 2  # bootstrap blocks use boot_reps, not n_reps
    for col in ("r2_ci_low", "ci_reliable", "cell", "rater_sd", "block"):
        assert col in df.columns
