"""Pre-canned study grid factories for DIF validation studies (Plan 3).

Each factory returns the list of DgpParams cells that define one study's design.
Implementations land in Tasks 2-4; stubs allow Task 1's import test to pass first.
"""

from __future__ import annotations

from sim.dgp import DgpParams


def type1_power_cells(n_items_per_group: int = 100, n_raters: int = 3) -> list[DgpParams]:
    """Standard null + DIF cells for Type-I control and power estimation."""
    raise NotImplementedError


def band_calibration_cells(n_items_per_group: int = 100, n_raters: int = 3) -> list[DgpParams]:
    """DIF magnitude sweep for Jodoin-Gierl band calibration."""
    raise NotImplementedError


def conditioner_comparison_cells(
    n_items_per_group: int = 100, n_raters: int = 3
) -> list[DgpParams]:
    """Null cells with varying impact for external vs. rest-score conditioner comparison."""
    raise NotImplementedError
