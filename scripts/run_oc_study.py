#!/usr/bin/env python
"""Run the E07 DIF operating-characteristics study and write per-block CSVs.

Usage:
    uv run python scripts/run_oc_study.py --study all --out sim/results
    uv run python scripts/run_oc_study.py --study s3

Analytic blocks default to 400 replications per cell (binomial SE at the
nominal 0.05 rate: sqrt(0.05*0.95/400) = 0.011). Bootstrap blocks default to
50 replications per cell (SE 0.031) with 200 resamples each, because a
bootstrap replication costs ~200x an analytic one. Seeds are fixed in
sim/oc_study.py; re-running a block reproduces its CSV byte-for-byte.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.oc_study import run_block, study_blocks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        default="all",
        choices=["s1", "s2", "s3", "s4", "s5", "all"],
        help="Sub-study to run (default: all).",
    )
    parser.add_argument("--out", default="sim/results", help="Output directory for per-block CSVs.")
    parser.add_argument("--reps", type=int, default=400, help="Replications per analytic cell.")
    parser.add_argument(
        "--boot-reps", type=int, default=50, help="Replications per bootstrap cell."
    )
    parser.add_argument(
        "--n-boot", type=int, default=200, help="Bootstrap resamples per replication."
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in study_blocks(args.study):
        t0 = time.perf_counter()
        df = run_block(block, n_reps=args.reps, boot_reps=args.boot_reps, n_boot=args.n_boot)
        path = out_dir / f"{block.name}.csv"
        df.to_csv(path, index=False)
        elapsed = time.perf_counter() - t0
        n_conv = (
            int(df["converged"].sum())
            if "converged" in df.columns
            else int(df["base_converged"].sum())
        )
        print(f"{block.name}: {len(df)} rows ({n_conv} converged) -> {path} [{elapsed:.0f}s]")


if __name__ == "__main__":
    main()
