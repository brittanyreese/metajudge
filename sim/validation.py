"""Pre-canned study grid factories for DIF validation studies (Plan 3).

Each factory returns the list of DgpParams cells that define one study's design.
"""

from __future__ import annotations

from typing import Any

from sim.dgp import DgpParams


def type1_power_cells(n_items_per_group: int = 100, n_raters: int = 3) -> list[DgpParams]:
    """Standard null + DIF cells for Type-I control and power estimation.

    Null cells (dif_uniform=dif_nonuniform=0) at three impact levels test Type-I control.
    DIF cells (mu_focal=0) at three uniform magnitudes and one nonuniform magnitude test
    power. All cells share the default thresholds, trait_slope, rater_sd.
    """
    base: dict[str, Any] = {"n_items_per_group": n_items_per_group, "n_raters": n_raters}
    return [
        # Null cells: no DIF, three impact levels
        DgpParams(**base, mu_focal=0.0),  # 0: H0, no impact
        DgpParams(**base, mu_focal=-0.5),  # 1: H0, moderate impact
        DgpParams(**base, mu_focal=-1.0),  # 2: H0, large impact
        # Uniform DIF cells: no impact
        DgpParams(**base, dif_uniform=0.5),  # 3: B-band territory
        DgpParams(**base, dif_uniform=1.0),  # 4: C-band territory
        DgpParams(**base, dif_uniform=1.5),  # 5: strong C-band
        # Nonuniform DIF cell
        DgpParams(**base, dif_nonuniform=0.3),  # 6: nonuniform DIF
    ]


def band_calibration_cells(n_items_per_group: int = 100, n_raters: int = 3) -> list[DgpParams]:
    """DIF magnitude sweep for Jodoin-Gierl band calibration.

    Six cells at dif_uniform in {0, 0.25, 0.50, 0.75, 1.00, 1.50} span the A-band
    (no DIF through near-B threshold) up to strong C-band. No impact (mu_focal=0).
    The study confirms mean R2_delta increases monotonically and that the A/B/C band
    transitions track the JG thresholds (0.035 and 0.070) at the chosen N.
    """
    base: dict[str, Any] = {"n_items_per_group": n_items_per_group, "n_raters": n_raters}
    return [
        DgpParams(**base, dif_uniform=0.00),  # 0: no DIF (A-band)
        DgpParams(**base, dif_uniform=0.25),  # 1: A-to-B transition
        DgpParams(**base, dif_uniform=0.50),  # 2: B-band territory
        DgpParams(**base, dif_uniform=0.75),  # 3: B-to-C transition
        DgpParams(**base, dif_uniform=1.00),  # 4: C-band territory
        DgpParams(**base, dif_uniform=1.50),  # 5: strong C-band
    ]


def cluster_stress_cells() -> list[DgpParams]:
    """Null cells stressing the crossed dependence: n_raters x rater_sd x n_items.

    Study S2 (2026-07-02 plan): does the analytic Type-I inflate anywhere in this
    crossed item-level-group design, and does the item-block bootstrap track it? The
    grid stresses the rater crossing (more raters, larger rater variance) and the
    per-group item count. A "within-cluster grouping" variant is not representable
    here: group is an item-level property by design.
    """
    return [
        DgpParams(n_items_per_group=n, n_raters=r, rater_sd=sd)
        for r in (3, 8)
        for sd in (0.5, 1.5)
        for n in (25, 100)
    ]


def overlap_rest_cells() -> list[DgpParams]:
    """Null cells in the rest-score confounded regime: impact x panel size.

    Study S3, block 1. Under impact the rest-score conditioner correlates with the
    group; pooling per-rep (|conditioner_group_corr|, reject) pairs across these
    cells builds the false-DIF-rate-vs-correlation curve that calibrates (or
    replaces) the 0.7 overlap-weak convention.
    """
    return [
        DgpParams(n_items_per_group=100, n_raters=r, mu_focal=mu)
        for r in (3, 8)
        for mu in (0.0, -0.5, -1.0, -1.5, -2.0)
    ]


def overlap_external_degraded_cells() -> list[DgpParams]:
    """Null cells with a degraded external conditioner: reliability x impact.

    Study S3, block 2: the confound also arises when an external conditioner is
    noisy under impact (residual impurity), not only for rest scores.
    """
    return [
        DgpParams(n_items_per_group=100, n_raters=3, mu_focal=mu, conditioner_reliability=rel)
        for rel in (0.8, 0.6, 0.4)
        for mu in (-1.0, -2.0)
    ]


def po_robustness_cells() -> list[DgpParams]:
    """PO violation crossed with DIF: Type-I and power when proportional odds breaks.

    Study S4: quantifies the estimator's robustness to its own key assumption and
    the consequence of the advisory Brant flag.
    """
    return [
        DgpParams(n_items_per_group=100, n_raters=3, po_violation=v, dif_uniform=d)
        for v in (0.3, 0.6)
        for d in (0.0, 1.0)
    ]


def sample_size_cells() -> list[DgpParams]:
    """Power curve at dif_uniform=1.0 (C-band) over per-group item counts.

    Study S5, block 1.
    """
    return [DgpParams(n_items_per_group=n, n_raters=3, dif_uniform=1.0) for n in (25, 50, 100, 200)]


def unbalanced_cells() -> list[DgpParams]:
    """Null + DIF with 100 reference vs 33 focal items.

    Study S5, block 2: French & Finch make group balance a driver of the inflation
    sign and magnitude; this checks the 3:1 imbalance the SummEval demo approximates.
    """
    return [
        DgpParams(n_items_per_group=100, n_raters=3, n_items_focal=33, dif_uniform=d)
        for d in (0.0, 1.0)
    ]


def conditioner_comparison_cells(
    n_items_per_group: int = 100, n_raters: int = 3
) -> list[DgpParams]:
    """Null cells (dif_uniform=0) at three impact levels for conditioner type comparison.

    Each cell is run twice: once with conditioner='external' and once with
    conditioner='rest_score'. Under impact, the rest-score conditioner is contaminated
    (focal group scores lower across items, not due to DIF), inflating the Type-I rate.
    """
    base: dict[str, Any] = {
        "n_items_per_group": n_items_per_group,
        "n_raters": n_raters,
        "dif_uniform": 0.0,
    }
    return [
        DgpParams(**base, mu_focal=0.0),  # 0: no impact (reference condition)
        DgpParams(**base, mu_focal=-0.5),  # 1: moderate impact
        DgpParams(**base, mu_focal=-1.0),  # 2: large impact
    ]
