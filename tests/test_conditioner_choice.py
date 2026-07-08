"""Guards the teaching claim of `examples/audit_conditioner_choice.py`.

The example exists to show that the conditioner argument to `audit()` is not
cosmetic: on the exact same biased panel from `examples/audit_catches_bias.py`,
the external conditioner correctly lands class C while the panel-relative
rest-score conditioner is corrupted by the panel's own shared bias and lands
class A instead, missing the effect entirely. These tests pin that outcome so
a refactor cannot quietly turn the demonstration into a boring (or wrong) one.
"""

from __future__ import annotations

import pytest
from examples.audit_conditioner_choice import build_cards

from metajudge import ReportCard


@pytest.fixture(scope="module")
def cards() -> dict[str, dict[str, ReportCard]]:
    return build_cards()


def test_clean_panel_is_class_a_on_both_conditioners(
    cards: dict[str, dict[str, ReportCard]],
) -> None:
    clean = cards["clean"]
    assert clean["external"].dif.dif_class == "A"
    assert clean["rest-score"].dif.dif_class == "A"


def test_biased_panel_external_conditioner_is_class_c(
    cards: dict[str, dict[str, ReportCard]],
) -> None:
    biased = cards["biased"]
    assert biased["external"].dif.dif_class == "C"
    assert biased["external"].dif.conditioner_source == "external"


def test_biased_panel_rest_score_conditioner_misses_the_effect(
    cards: dict[str, dict[str, ReportCard]],
) -> None:
    # The teaching point: the SAME biased data, audited with the panel-relative
    # rest score instead of the external conditioner, lands class A -- not a
    # smaller effect, a missed one.
    biased = cards["biased"]
    assert biased["rest-score"].dif.dif_class == "A"
    assert biased["rest-score"].dif.conditioner_source == "rest_score"


def test_rest_score_overlap_diagnostic_flags_the_corrupted_conditioner(
    cards: dict[str, dict[str, ReportCard]],
) -> None:
    # The engine's own advisory should catch the corruption automatically: weak
    # only on the biased panel's rest-score conditioner, clean everywhere else.
    clean, biased = cards["clean"], cards["biased"]
    assert biased["rest-score"].dif.conditioner_overlap_weak is True
    assert clean["rest-score"].dif.conditioner_overlap_weak is False
    assert clean["external"].dif.conditioner_overlap_weak is False
    assert biased["external"].dif.conditioner_overlap_weak is False
