# sim/oc_study.py
"""Operating-characteristics study blocks (S1-S5) for the E07 DIF pillar.

Declarative block specs plus a runner; the CLI shell lives in
``scripts/run_oc_study.py``. Sub-study design and rationale:
docs/reviews/2026-07-02-multi-persona-review.md (Tier 2, item 7).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from sim.dgp import DgpParams
from sim.harness import run_cell_bootstrap, run_grid
from sim.validation import (
    cluster_stress_cells,
    overlap_external_degraded_cells,
    overlap_rest_cells,
    po_robustness_cells,
    sample_size_cells,
    type1_power_cells,
    unbalanced_cells,
)

BASE_SEED = 20260702
_BLOCK_SEED_STRIDE = 10_000_000  # blocks never share a replication stream


@dataclass(frozen=True)
class Block:
    """One runnable block of the study: a cell grid plus run configuration."""

    name: str
    cells: list[DgpParams]
    conditioner: str  # "external" | "rest_score"
    kind: str  # "analytic" | "bootstrap"
    base_seed: int


def _all_blocks() -> list[Block]:
    specs: list[tuple[str, list[DgpParams], str, str]] = [
        ("s1_baseline", type1_power_cells(), "external", "analytic"),
        ("s2_cluster_stress_analytic", cluster_stress_cells(), "external", "analytic"),
        ("s2_cluster_stress_bootstrap", cluster_stress_cells(), "external", "bootstrap"),
        ("s3_overlap_rest", overlap_rest_cells(), "rest_score", "analytic"),
        ("s3_overlap_external_degraded", overlap_external_degraded_cells(), "external", "analytic"),
        ("s4_po_robustness", po_robustness_cells(), "external", "analytic"),
        ("s5_sample_size", sample_size_cells(), "external", "analytic"),
        ("s5_unbalanced", unbalanced_cells(), "external", "analytic"),
    ]
    return [
        Block(
            name=name,
            cells=cells,
            conditioner=cond,
            kind=kind,
            base_seed=BASE_SEED + i * _BLOCK_SEED_STRIDE,
        )
        for i, (name, cells, cond, kind) in enumerate(specs)
    ]


def study_blocks(study: str) -> list[Block]:
    """Blocks for one sub-study (``"s1"``..``"s5"``) or ``"all"``."""
    blocks = _all_blocks()
    if study == "all":
        return blocks
    selected = [b for b in blocks if b.name.startswith(f"{study}_")]
    if not selected:
        known = sorted({b.name.split("_", 1)[0] for b in blocks})
        raise ValueError(f"unknown study {study!r}; expected one of {known} or 'all'")
    return selected


def run_block(block: Block, *, n_reps: int, boot_reps: int, n_boot: int) -> pd.DataFrame:
    """Run one block and return a tidy per-replication frame with provenance columns.

    Analytic blocks run ``n_reps`` replications per cell through ``run_grid``;
    bootstrap blocks run ``boot_reps`` replications per cell through
    ``run_cell_bootstrap`` (each replication itself draws ``n_boot`` resamples).
    """
    if block.kind == "analytic":
        df = run_grid(
            block.cells,
            n_reps=n_reps,
            base_seed=block.base_seed,
            conditioner=block.conditioner,
        )
    elif block.kind == "bootstrap":
        parts: list[pd.DataFrame] = []
        for idx, params in enumerate(block.cells):
            cell_df = run_cell_bootstrap(
                params,
                n_reps=boot_reps,
                n_boot=n_boot,
                base_seed=block.base_seed + idx * 100_000,
                conditioner=block.conditioner,
            )
            cell_df.insert(0, "cell", idx)
            for field, value in asdict(params).items():
                cell_df[field] = [value] * len(cell_df)
            parts.append(cell_df)
        df = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(f"unknown block kind {block.kind!r}")
    df["conditioner"] = block.conditioner
    df["block"] = block.name
    return df
