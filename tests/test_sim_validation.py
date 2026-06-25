"""Validation studies: Type-I/power, band calibration, conditioner comparison, bootstrap CI."""

from __future__ import annotations

from sim.validation import band_calibration_cells, conditioner_comparison_cells, type1_power_cells


def test_validation_module_imports() -> None:
    assert callable(type1_power_cells)
    assert callable(band_calibration_cells)
    assert callable(conditioner_comparison_cells)
