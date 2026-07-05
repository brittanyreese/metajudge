#!/usr/bin/env python
"""Empirically derive ordinal proportional-odds A/B/C effect-size bands by Monte Carlo.

The shipped DIF engine classifies a Nagelkerke R-squared change on the Jodoin & Gierl
(2001) thresholds (0.035 negligible/moderate, 0.070 moderate/large). Those bands were
calibrated on the *dichotomous* logistic Nagelkerke; transferring them to the *ordinal*
proportional-odds (PO) Nagelkerke this library computes is an unvalidated convention
(the ADR admits it; lordif's own default cutoff is 0.02). This study derives the bands
the ordinal case actually warrants, mirroring Jodoin & Gierl's logic:

1. Hold an interpretable DIF magnitude fixed. The anchor is the uniform-DIF coefficient
   ``b2`` (a constant shift in the cumulative logit between focal and reference), which
   is a log odds-ratio: ``OR = exp(b2)``. It has the same meaning for a 2-category
   logistic outcome and a 5-category PO outcome, so it transports cleanly between them.

2. Reproduce the dichotomous regime Jodoin & Gierl calibrated on (K=2, the PO fit
   reduces to ordinary logistic) and find the anchor ``b2`` where the dichotomous mean
   R-squared change crosses 0.035 and 0.070. Those ``b2`` values are the true DIF
   magnitudes their bands encode.

3. Read the *ordinal* (K=5) mean R-squared change at those same anchor ``b2`` values.
   That is the derived ordinal-PO band. If it differs from 0.035/0.070, the transfer is
   miscalibrated and the bands should move to the derived values.

Both metrics use the shipped engine (``metajudge.dif._dif_stats`` /
``_fit_proportional_odds`` / ``_nagelkerke``) on the shared cumulative-logit DGP
(``sim.dgp``), at the SummEval demo's scale (n_obs = 4800). Nothing about the numerics
is reimplemented here.

Regenerate (seed pinned below; deterministic):

    uv run python scripts/derive_ordinal_bands.py --out sim/results

Runtime is a few minutes at the defaults. All randomness flows from BAND_SEED.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.dgp import FOCAL, DgpParams, simulate

# This study reuses the shipped engine's fit rather than reimplementing it, so it imports
# the private numerics (_dif_stats -> _fit_proportional_odds -> _nagelkerke) and the shipped
# JG thresholds directly. The pyright suppression is scoped to this one line.
from metajudge.dif import (
    _JG_LARGE,  # pyright: ignore[reportPrivateUsage]
    _JG_NEGLIGIBLE,  # pyright: ignore[reportPrivateUsage]
    _dif_stats,  # pyright: ignore[reportPrivateUsage]
    _emit_item_rows,  # pyright: ignore[reportPrivateUsage]
)

# Single seed for the whole study; every replication seed derives from it.
BAND_SEED = 20260704
_METRIC_SEED_STRIDE = 1_000_000  # dichotomous vs ordinal never share a replication stream
_POINT_SEED_STRIDE = 10_000  # each b2 point gets a disjoint block

# lordif's own default R-squared-change cutoff (Choi, Gibbons & Crane, 2011), the third
# candidate boundary the demo's class-A headline must survive.
_LORDIF_CUTOFF = 0.02

# The SummEval demo's shipped ordinal-PO effect size (README, examples/audit_summeval.py).
_DEMO_R2_DELTA = 0.002


@dataclass(frozen=True)
class PointSummary:
    """Monte-Carlo summary of the R-squared change at one (metric, b2) design point."""

    metric: str
    b2: float
    odds_ratio: float
    n_reps: int
    n_converged: int
    mean_r2_delta: float
    sd_r2_delta: float
    mc_se_r2_delta: float


def _assemble_rows(params: DgpParams, seed: int) -> tuple[list[float], list[float], list[float]]:
    """Draw one DGP sample and lay it out the way ``logistic_dif`` does (external conditioner).

    Reuses the engine's own row emitter so the (score, group, conditioner) rows are
    byte-identical to the shipped analytic path; only the uniform/nonuniform split and the
    Brant diagnostic are skipped downstream (not needed for the R-squared change).
    """
    sample = simulate(params, seed=seed)
    mat: NDArray[np.float64] = sample.ratings.wide().to_numpy(dtype=float)
    focal_items = set(sample.ratings.strata()[FOCAL])
    scores: list[float] = []
    groups: list[float] = []
    cond_rows: list[float] = []
    for row_idx, item in enumerate(sample.ratings.items):
        row: NDArray[np.float64] = mat[row_idx]
        rated: NDArray[np.float64] = row[~np.isnan(row)]
        _emit_item_rows(
            rated,
            is_focal=item in focal_items,
            item_cond=float(sample.conditioner[item]),
            scores=scores,
            groups=groups,
            cond_rows=cond_rows,
        )
    return scores, groups, cond_rows


def _r2_delta(params: DgpParams, seed: int) -> tuple[float, bool]:
    """Shipped-engine Nagelkerke R-squared change for one draw. ``(r2_delta, converged)``."""
    scores, groups, cond_rows = _assemble_rows(params, seed)
    try:
        stats = _dif_stats(scores, groups, cond_rows, want_split=False)
    except ValueError:
        return float("nan"), False
    return stats.nagelkerke_r2_delta, stats.converged


def _point(
    *,
    metric: str,
    b2: float,
    n_items_per_group: int,
    n_raters: int,
    n_reps: int,
    base_seed: int,
) -> PointSummary:
    """Replicate one (metric, b2) point and summarize its R-squared-change distribution."""
    if metric == "ordinal":
        params = DgpParams(n_items_per_group=n_items_per_group, n_raters=n_raters, dif_uniform=b2)
    elif metric == "dichotomous":
        params = DgpParams(
            n_items_per_group=n_items_per_group,
            n_raters=n_raters,
            n_categories=2,
            thresholds=(0.0,),
            dif_uniform=b2,
        )
    else:
        raise ValueError(f"unknown metric {metric!r}")

    vals: list[float] = []
    for rep in range(n_reps):
        r2, converged = _r2_delta(params, seed=base_seed + rep)
        if converged and not np.isnan(r2):
            vals.append(r2)
    arr: NDArray[np.float64] = np.asarray(vals, dtype=float)
    n_conv = int(arr.size)
    mean = float(arr.mean()) if n_conv else float("nan")
    sd = float(arr.std(ddof=1)) if n_conv > 1 else float("nan")
    mc_se = float(sd / np.sqrt(n_conv)) if n_conv > 1 else float("nan")
    return PointSummary(
        metric=metric,
        b2=b2,
        odds_ratio=float(np.exp(b2)),
        n_reps=n_reps,
        n_converged=n_conv,
        mean_r2_delta=mean,
        sd_r2_delta=sd,
        mc_se_r2_delta=mc_se,
    )


def _sweep(
    *,
    metric: str,
    b2_grid: NDArray[np.float64],
    n_items_per_group: int,
    n_raters: int,
    n_reps: int,
    metric_seed: int,
) -> list[PointSummary]:
    """Run the full b2 grid for one metric on disjoint per-point seed blocks."""
    out: list[PointSummary] = []
    for i, b2 in enumerate(b2_grid):
        out.append(
            _point(
                metric=metric,
                b2=float(b2),
                n_items_per_group=n_items_per_group,
                n_raters=n_raters,
                n_reps=n_reps,
                base_seed=metric_seed + i * _POINT_SEED_STRIDE,
            )
        )
    return out


def _interp_at(x: float, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> float:
    """Linear interpolation of ``ys`` at ``x`` over an ascending ``xs`` grid."""
    return float(np.interp(x, xs, ys))


def _invert(target: float, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> float:
    """Smallest ``x`` where an ascending ``ys(x)`` first reaches ``target`` (linear interp).

    Returns NaN when ``target`` lies outside the swept range, so an out-of-range crossing
    is reported honestly rather than clamped to a grid endpoint.
    """
    if target < ys[0] or target > ys[-1]:
        return float("nan")
    return float(np.interp(target, ys, xs))


@dataclass(frozen=True)
class Derivation:
    """The derived ordinal-PO bands and the anchors that ground them."""

    # Anchor b2 (and OR) where the dichotomous Nagelkerke crosses the shipped JG bands.
    b2_anchor_ab: float
    b2_anchor_bc: float
    or_anchor_ab: float
    or_anchor_bc: float
    # Derived ordinal-PO band values: ordinal mean R2-delta read at those anchor b2's.
    ordinal_band_ab: float
    ordinal_band_bc: float
    ordinal_band_ab_se: float
    ordinal_band_bc_se: float
    # For reference: b2 where the ORDINAL curve itself crosses the shipped bands and the
    # lordif cutoff (what DIF magnitude each candidate threshold corresponds to, ordinally).
    b2_ordinal_at_035: float
    b2_ordinal_at_070: float
    b2_ordinal_at_lordif: float


def derive(ordinal: list[PointSummary], dichotomous: list[PointSummary]) -> Derivation:
    """Transport the Jodoin-Gierl anchors onto the ordinal-PO Nagelkerke curve."""
    b2 = np.asarray([p.b2 for p in ordinal], dtype=float)
    ord_mean = np.asarray([p.mean_r2_delta for p in ordinal], dtype=float)
    ord_se = np.asarray([p.mc_se_r2_delta for p in ordinal], dtype=float)
    dich_mean = np.asarray([p.mean_r2_delta for p in dichotomous], dtype=float)

    # Step 2: anchor b2 where the dichotomous Nagelkerke crosses the shipped JG bands.
    b2_ab = _invert(_JG_NEGLIGIBLE, b2, dich_mean)
    b2_bc = _invert(_JG_LARGE, b2, dich_mean)

    # Step 3: read the ordinal curve at those anchor magnitudes -> derived ordinal bands.
    band_ab = _interp_at(b2_ab, b2, ord_mean)
    band_bc = _interp_at(b2_bc, b2, ord_mean)
    band_ab_se = _interp_at(b2_ab, b2, ord_se)
    band_bc_se = _interp_at(b2_bc, b2, ord_se)

    return Derivation(
        b2_anchor_ab=b2_ab,
        b2_anchor_bc=b2_bc,
        or_anchor_ab=float(np.exp(b2_ab)),
        or_anchor_bc=float(np.exp(b2_bc)),
        ordinal_band_ab=band_ab,
        ordinal_band_bc=band_bc,
        ordinal_band_ab_se=band_ab_se,
        ordinal_band_bc_se=band_bc_se,
        b2_ordinal_at_035=_invert(_JG_NEGLIGIBLE, b2, ord_mean),
        b2_ordinal_at_070=_invert(_JG_LARGE, b2, ord_mean),
        b2_ordinal_at_lordif=_invert(_LORDIF_CUTOFF, b2, ord_mean),
    )


def _summaries_frame(points: list[PointSummary]) -> pd.DataFrame:
    return pd.DataFrame([vars(p) for p in points])


def _print_report(
    ordinal: list[PointSummary],
    dichotomous: list[PointSummary],
    derivation: Derivation,
    *,
    n_obs: int,
) -> None:
    print(f"\n=== Ordinal-PO A/B/C band derivation (n_obs = {n_obs}, seed {BAND_SEED}) ===\n")
    print("b2 (log-OR) -> mean Nagelkerke R2-delta [MC SE], by outcome metric:")
    print(f"{'b2':>6} {'OR':>6} {'ordinal':>18} {'dichotomous':>18}")
    for o, d in zip(ordinal, dichotomous, strict=True):
        print(
            f"{o.b2:>6.2f} {o.odds_ratio:>6.2f} "
            f"{o.mean_r2_delta:>10.4f} +-{o.mc_se_r2_delta:<5.4f} "
            f"{d.mean_r2_delta:>10.4f} +-{d.mc_se_r2_delta:<5.4f}"
        )

    dv = derivation
    print("\nJodoin-Gierl anchors (dichotomous Nagelkerke crossings):")
    print(f"  A/B (0.035): b2 = {dv.b2_anchor_ab:.3f}  (OR = {dv.or_anchor_ab:.2f})")
    print(f"  B/C (0.070): b2 = {dv.b2_anchor_bc:.3f}  (OR = {dv.or_anchor_bc:.2f})")
    print("\nDERIVED ordinal-PO bands (ordinal Nagelkerke at those anchor magnitudes):")
    print(f"  A/B: {dv.ordinal_band_ab:.4f}  (MC SE {dv.ordinal_band_ab_se:.4f})")
    print(f"  B/C: {dv.ordinal_band_bc:.4f}  (MC SE {dv.ordinal_band_bc_se:.4f})")
    print("\nWhere the candidate thresholds land on the ordinal magnitude axis:")
    for label, b2v in (
        ("shipped 0.035", dv.b2_ordinal_at_035),
        ("shipped 0.070", dv.b2_ordinal_at_070),
        ("lordif  0.020", dv.b2_ordinal_at_lordif),
    ):
        print(f"  {label} -> b2 = {b2v:.3f} (OR {np.exp(b2v):.2f})")

    print("\nSummEval demo class-A stability (observed ordinal R2-delta = 0.002):")
    for label, thr in (
        ("lordif 0.020", _LORDIF_CUTOFF),
        ("shipped 0.035", _JG_NEGLIGIBLE),
        ("derived A/B", dv.ordinal_band_ab),
    ):
        verdict = "A (negligible)" if thr > _DEMO_R2_DELTA else "NOT class A"
        margin = thr / _DEMO_R2_DELTA if _DEMO_R2_DELTA > 0 else float("inf")
        print(f"  vs {label:>14} = {thr:.4f}: {verdict}  ({margin:.1f}x margin)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="sim/results", help="Directory for the per-point CSVs.")
    parser.add_argument("--reps", type=int, default=400, help="Replications per (metric, b2).")
    parser.add_argument(
        "--items-per-group",
        type=int,
        default=800,
        help="Items per stratum; n_obs = 2 * items * raters (default 800 -> 4800).",
    )
    parser.add_argument("--raters", type=int, default=3, help="Raters per item.")
    args = parser.parse_args()

    # Grid dense around both crossings (0.035 near OR~2, 0.070 near OR~2.7).
    b2_grid: NDArray[np.float64] = np.array(
        [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6], dtype=float
    )
    n_obs = 2 * args.items_per_group * args.raters

    t0 = time.perf_counter()
    ordinal = _sweep(
        metric="ordinal",
        b2_grid=b2_grid,
        n_items_per_group=args.items_per_group,
        n_raters=args.raters,
        n_reps=args.reps,
        metric_seed=BAND_SEED,
    )
    dichotomous = _sweep(
        metric="dichotomous",
        b2_grid=b2_grid,
        n_items_per_group=args.items_per_group,
        n_raters=args.raters,
        n_reps=args.reps,
        metric_seed=BAND_SEED + _METRIC_SEED_STRIDE,
    )
    derivation = derive(ordinal, dichotomous)
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = _summaries_frame(ordinal + dichotomous)
    points.to_csv(out_dir / "band_derivation_points.csv", index=False)
    pd.DataFrame([vars(derivation)]).to_csv(out_dir / "band_derivation_summary.csv", index=False)

    _print_report(ordinal, dichotomous, derivation, n_obs=n_obs)
    print(f"\nWrote {out_dir}/band_derivation_points.csv and band_derivation_summary.csv")
    print(f"Total wall time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
