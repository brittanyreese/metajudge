"""Guards the teaching claim of `examples/audit_catches_bias.py`.

The example exists to show the report card catching a biased judge panel: two
panels whose reliability numbers look alike, where only the DIF pillar fires
class C. These tests pin that outcome so a refactor or a dependency bump cannot
quietly turn the demonstration into a boring (or wrong) one.
"""

from __future__ import annotations

import pytest
from examples.audit_catches_bias import build_cards

from metajudge import ReportCard


@pytest.fixture(scope="module")
def cards() -> tuple[ReportCard, ReportCard]:
    return build_cards()


def test_clean_panel_is_class_a(cards: tuple[ReportCard, ReportCard]) -> None:
    clean, _ = cards
    assert clean.dif.dif_class == "A"


def test_biased_panel_is_class_c(cards: tuple[ReportCard, ReportCard]) -> None:
    _, biased = cards
    assert biased.dif.dif_class == "C"


def test_biased_panel_matched_on_external_conditioner(
    cards: tuple[ReportCard, ReportCard],
) -> None:
    # Guards R5: the class-C verdict comes from the external quality axis, not a
    # stratum level shift leaking through the panel-relative rest score.
    _, biased = cards
    assert biased.dif.conditioner_source == "external"


def test_reliability_numbers_do_not_separate_the_panels(
    cards: tuple[ReportCard, ReportCard],
) -> None:
    # Covers R4: Krippendorff alpha looks alike across both panels; only DIF fires.
    clean, biased = cards
    assert abs(clean.alpha.alpha - biased.alpha.alpha) <= 0.05
