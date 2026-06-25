"""Pre-canned study grid factories for DIF validation studies (Plan 3).

Each factory returns the list of DgpParams cells that define one study's design.
Implementations land in Tasks 2-4; stubs allow Task 1's import test to pass first.
"""

from __future__ import annotations

from sim.dgp import DgpParams


def type1_power_cells(n_items_per_group: int = 100, n_raters: int = 3) -> list[DgpParams]:
    """Standard null + DIF cells for Type-I control and power estimation.

    Null cells (dif_uniform=dif_nonuniform=0) at three impact levels test Type-I control.
    DIF cells (mu_focal=0) at three uniform magnitudes and one nonuniform magnitude test
    power. All cells share the default thresholds, trait_slope, rater_sd.
    """
    base: dict[str, int] = {"n_items_per_group": n_items_per_group, "n_raters": n_raters}
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
    """DIF magnitude sweep for Jodoin-Gierl band calibration."""
    raise NotImplementedError


def conditioner_comparison_cells(
    n_items_per_group: int = 100, n_raters: int = 3
) -> list[DgpParams]:
    """Null cells with varying impact for external vs. rest-score conditioner comparison."""
    raise NotImplementedError
