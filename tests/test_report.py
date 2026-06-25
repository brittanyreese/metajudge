# tests/test_report.py
from collections.abc import Hashable

import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.dif import DifResult, logistic_dif
from metajudge.reliability import AlphaResult, IccResult
from metajudge.report import ReportCard, audit


def _card(*, converged: bool, conditioner_source: str = "rest_score") -> ReportCard:
    """A minimal report card with a chosen convergence flag / conditioner source."""
    alpha = AlphaResult(
        alpha=0.5, ci_low=0.4, ci_high=0.6, level="ordinal", n_bootstrap=1000, n_effective=1000
    )
    ic = IccResult(icc1=0.3, icck=0.6, n_targets=40, n_raters=3)
    dif = DifResult(
        chi2_total=1.0,
        chi2_uniform=0.5,
        chi2_nonuniform=0.5,
        p_total=0.6,
        p_uniform=0.5,
        p_nonuniform=0.5,
        nagelkerke_r2_delta=0.01,
        dif_class="A",
        conditioner_source=conditioner_source,
        n_obs=120,
        reference_level="ref",
        focal_level="foc",
        converged=converged,
        po_violation=False,
    )
    return ReportCard(alpha=alpha, icc=ic, dif=dif)


def _card_with_alpha(alpha: AlphaResult) -> ReportCard:
    icc_result = IccResult(icc1=0.8, icck=0.9, n_targets=10, n_raters=3)
    dif = DifResult(
        chi2_total=1.0,
        chi2_uniform=0.5,
        chi2_nonuniform=0.5,
        p_total=0.6,
        p_uniform=0.5,
        p_nonuniform=0.5,
        nagelkerke_r2_delta=0.01,
        dif_class="A",
        conditioner_source="rest_score",
        n_obs=30,
        reference_level="ref",
        focal_level="foc",
        converged=True,
        po_violation=False,
    )
    return ReportCard(alpha=alpha, icc=icc_result, dif=dif)


def _ratings() -> Ratings:
    rows: list[dict[str, object]] = []
    groups = ["foc", "ref"]
    for i in range(40):
        grp = groups[i % 2]
        for r_idx in range(3):
            rows.append(
                {"item": f"i{i}", "rater": f"r{r_idx}", "score": (i + r_idx) % 5, "group": grp}
            )
    df = pd.DataFrame(rows)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")


def test_audit_returns_full_card() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    assert isinstance(card, ReportCard)
    assert card.dif.dif_class in {"A", "B", "C"}


def test_markdown_contains_all_pillars() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    md = card.to_markdown()
    assert "Krippendorff" in md
    assert "ICC" in md
    assert "DIF" in md
    assert card.dif.dif_class in md


def test_audit_forwards_external_conditioner() -> None:
    # Strong uniform DIF (reference high, focal low) with a conditioner spread across both
    # groups, so the external-conditioner result (class C) is far from the rest-score
    # result (class B) on the same data. The numeric equality asserts then ride on a
    # large value a silent rest-score fallback could not satisfy: the label assert alone
    # would pass even on a near-zero degenerate fixture, so the magnitude is the point.
    ref_scores = [
        [4, 5, 5],
        [5, 4, 5],
        [4, 4, 5],
        [5, 5, 4],
        [4, 5, 4],
        [5, 4, 4],
        [5, 5, 5],
        [4, 4, 4],
    ]
    foc_scores = [
        [1, 2, 1],
        [2, 1, 1],
        [1, 1, 2],
        [2, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
        [1, 1, 1],
        [2, 2, 2],
    ]
    rows: list[dict[str, object]] = []
    cond: dict[Hashable, float] = {}
    for idx, scores in enumerate(ref_scores):
        item = f"i{idx}"
        cond[item] = float(idx % 4)  # spread across the group, not collinear with it
        for r, v in enumerate(scores):
            rows.append({"item": item, "rater": f"r{r}", "score": v, "group": "ref"})
    for idx, scores in enumerate(foc_scores):
        item = f"i{idx + 8}"
        cond[item] = float(idx % 4)
        for r, v in enumerate(scores):
            rows.append({"item": item, "rater": f"r{r}", "score": v, "group": "foc"})
    df = pd.DataFrame(rows)
    ratings = Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")

    card = audit(ratings, focal="foc", reference="ref", level="ordinal", seed=1, conditioner=cond)
    direct = logistic_dif(ratings, focal="foc", reference="ref", conditioner=cond)
    rest = logistic_dif(ratings, focal="foc", reference="ref")

    assert card.dif.conditioner_source == "external"
    assert card.dif.dif_class == "C"
    assert card.dif.chi2_total > 5.0  # non-degenerate, so equality below is not noise == noise
    assert card.dif.chi2_total == pytest.approx(direct.chi2_total)
    assert card.dif.nagelkerke_r2_delta == pytest.approx(direct.nagelkerke_r2_delta)
    # the rest-score path on the SAME data is materially different, so the forwarded
    # external value could not have come from a silent fallback to the default path
    assert abs(rest.chi2_total - card.dif.chi2_total) > 1.0


def test_markdown_rest_score_is_panel_relative_and_warns() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    md = card.to_markdown()
    assert card.dif.conditioner_source == "rest_score"
    assert "panel-relative" in md
    assert "fairness clearance" in md


def test_markdown_external_is_instrument_level_no_warning() -> None:
    cond: dict[Hashable, float] = {f"i{i}": float(i % 5) for i in range(40)}
    card = audit(
        _ratings(), focal="foc", reference="ref", level="ordinal", seed=1, conditioner=cond
    )
    md = card.to_markdown()
    assert card.dif.conditioner_source == "external"
    assert "instrument-level" in md
    assert "panel-relative" not in md
    assert "fairness clearance" not in md


def test_markdown_warns_when_dif_fit_not_converged() -> None:
    md = _card(converged=False).to_markdown()
    assert "did not converge" in md


def test_markdown_no_convergence_warning_when_converged() -> None:
    md = _card(converged=True).to_markdown()
    assert "did not converge" not in md


def test_markdown_rest_score_caveat_precedes_statistics() -> None:
    # The panel-relative caveat must sit above the chi-square lines: a reader who excerpts
    # the headline numbers should not be able to drop the "not a fairness clearance" note.
    md = _card(converged=True, conditioner_source="rest_score").to_markdown()
    assert "fairness clearance" in md
    assert md.index("fairness clearance") < md.index("Uniform DIF")


def test_markdown_flags_dropped_bootstrap_replicates() -> None:
    # When resamples were dropped, the CI rests on fewer replicates than requested;
    # the card must surface the realized count so the CI is not read as full-strength.
    alpha = AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=1000,
        n_effective=120,
    )
    md = _card_with_alpha(alpha).to_markdown()
    assert "120" in md
    assert "1000" in md


def test_markdown_omits_caveat_when_all_replicates_realized() -> None:
    alpha = AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=1000,
        n_effective=1000,
    )
    md = _card_with_alpha(alpha).to_markdown()
    # No drop warning: the realized count equals the requested count.
    assert "of 1000" not in md


def test_report_warns_on_po_violation() -> None:
    from dataclasses import replace

    card = _card(converged=True)
    assert "proportional-odds" not in card.to_markdown().lower()

    violated = replace(card, dif=replace(card.dif, po_violation=True))
    md = violated.to_markdown()
    assert "proportional-odds" in md.lower()
