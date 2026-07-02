# tests/test_report.py
from collections.abc import Hashable

import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.dif import ClusterBootstrapDif, DifResult, logistic_dif
from metajudge.reliability import AlphaResult, IccResult
from metajudge.report import Flags, ReportCard, audit


def _bootstrap(
    *, r2_low: float, r2_high: float, n_effective: int = 500, ci_method: str = "percentile"
) -> ClusterBootstrapDif:
    """A deterministic ClusterBootstrapDif for card-rendering tests (no refits)."""
    dif = _card(converged=True).dif
    return ClusterBootstrapDif(
        base=dif,
        r2_delta_ci_low=r2_low,
        r2_delta_ci_high=r2_high,
        chi2_total_ci_low=0.0,
        chi2_total_ci_high=5.0,
        cluster="item",
        ci_level=0.95,
        n_boot=1000,
        n_effective=n_effective,
        ci_method=ci_method,
    )


def _card(*, converged: bool, conditioner_source: str = "rest_score") -> ReportCard:
    """A minimal report card with a chosen convergence flag / conditioner source."""
    alpha = AlphaResult(
        alpha=0.5, ci_low=0.4, ci_high=0.6, level="ordinal", n_bootstrap=1000, n_effective=1000
    )
    ic = IccResult(
        icc1=0.3,
        icck=0.6,
        n_targets=40,
        n_raters=3,
        icc1_ci_low=0.1,
        icc1_ci_high=0.5,
        icck_ci_low=0.25,
        icck_ci_high=0.75,
    )
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
        conditioner_group_corr=0.0,
    )
    return ReportCard(alpha=alpha, icc=ic, dif=dif)


def _card_with_alpha(alpha: AlphaResult) -> ReportCard:
    icc_result = IccResult(
        icc1=0.8,
        icck=0.9,
        n_targets=10,
        n_raters=3,
        icc1_ci_low=0.6,
        icc1_ci_high=0.9,
        icck_ci_low=0.7,
        icck_ci_high=0.95,
    )
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
        conditioner_group_corr=0.0,
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


def test_markdown_external_names_conditioner_and_warns_about_interpretation() -> None:
    cond: dict[Hashable, float] = {f"i{i}": float(i % 5) for i in range(40)}
    card = audit(
        _ratings(), focal="foc", reference="ref", level="ordinal", seed=1, conditioner=cond
    )
    md = card.to_markdown()
    assert card.dif.conditioner_source == "external"
    assert "external conditioner" in md
    assert "instrument-level interpretation" in md
    assert "valid, independent" in md
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


def test_markdown_warns_analytic_dif_pvalues_are_anti_conservative() -> None:
    # The card's DIF p-values are the analytic i.i.d. likelihood-ratio test, which pools
    # each (item, rater) cell as independent and is anti-conservative under the crossed
    # rater x item design. The caveat must sit above the chi-square lines and point to the
    # cluster bootstrap, so an excerpter cannot read the p-values as clustering-robust.
    md = _card(converged=True, conditioner_source="rest_score").to_markdown()
    assert "anti-conservative" in md
    assert "cluster_bootstrap_dif" in md
    assert md.index("anti-conservative") < md.index("Uniform DIF")


def test_audit_default_has_no_bootstrap() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    assert card.dif_bootstrap is None
    assert card.flags.dif_robustly_nonnegligible is None


def test_audit_robust_attaches_cluster_bootstrap() -> None:
    card = audit(
        _ratings(), focal="foc", reference="ref", level="ordinal", seed=1, robust=True, n_boot=200
    )
    assert isinstance(card.dif_bootstrap, ClusterBootstrapDif)
    assert card.dif_bootstrap.n_boot == 200


def test_markdown_fast_omits_robust_significance_and_leads_with_effect_size() -> None:
    # Fast card: no clustering-robust flag was computed, so significance is "not assessed"
    # and the analytic p-values are tagged; the effect-size + class line leads the DIF
    # block (the screen's decision variable), sitting above the analytic detail lines.
    md = _card(converged=True).to_markdown()
    assert "not assessed" in md
    assert "[analytic, unclustered]" in md
    assert md.index("Nagelkerke") < md.index("Uniform DIF")


def test_markdown_robust_shows_cluster_flag_with_effect_size_ci() -> None:
    card = ReportCard(
        alpha=_card(converged=True).alpha,
        icc=_card(converged=True).icc,
        dif=_card(converged=True).dif,
        dif_bootstrap=_bootstrap(r2_low=0.05, r2_high=0.12),
    )
    md = card.to_markdown()
    assert "Clustering-robust flag" in md
    assert "item-cluster bootstrap" in md
    assert "not assessed" not in md  # significance WAS assessed
    # The verdict rests on a percentile CI bound vs the JG boundary; the card must disclose
    # that the percentile method is least accurate at the R2-change's lower bound of 0.
    assert "percentile CI" in md
    assert "boundary" in md


def test_markdown_robust_bca_method_labels_and_caveats_bca() -> None:
    card = ReportCard(
        alpha=_card(converged=True).alpha,
        icc=_card(converged=True).icc,
        dif=_card(converged=True).dif,
        dif_bootstrap=_bootstrap(r2_low=0.05, r2_high=0.12, ci_method="bca"),
    )
    md = card.to_markdown()
    assert "BCa CI" in md  # the CI is labelled by its method
    assert "bias-corrected accelerated (BCa)" in md  # caveat names the method
    assert "percentile method is least accurate" not in md  # the percentile-only caveat


def test_flags_dif_robustly_nonnegligible_true_when_ci_clears_band() -> None:
    card = ReportCard(
        alpha=_card(converged=True).alpha,
        icc=_card(converged=True).icc,
        dif=_card(converged=True).dif,
        dif_bootstrap=_bootstrap(r2_low=0.05, r2_high=0.12),  # lower bound above 0.035
    )
    assert card.flags.dif_robustly_nonnegligible is True


def test_flags_dif_robustly_nonnegligible_false_when_ci_includes_band() -> None:
    card = ReportCard(
        alpha=_card(converged=True).alpha,
        icc=_card(converged=True).icc,
        dif=_card(converged=True).dif,
        dif_bootstrap=_bootstrap(r2_low=0.01, r2_high=0.09),  # lower bound in negligible band
    )
    assert card.flags.dif_robustly_nonnegligible is False


def test_flags_dif_robustly_nonnegligible_none_when_ci_unreliable() -> None:
    # Too few resamples survived: the CI is noise, so the robust flag is withheld (None),
    # not reported as a confident False.
    card = ReportCard(
        alpha=_card(converged=True).alpha,
        icc=_card(converged=True).icc,
        dif=_card(converged=True).dif,
        dif_bootstrap=_bootstrap(r2_low=0.05, r2_high=0.12, n_effective=40),
    )
    assert card.flags.dif_robustly_nonnegligible is None


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


def test_markdown_flags_thin_alpha_ci_even_without_drops() -> None:
    alpha = AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=50,
        n_effective=50,
    )
    md = _card_with_alpha(alpha).to_markdown()
    assert "indicative only" in md
    assert "50" in md


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


def _alpha(*, n_bootstrap: int, n_effective: int) -> AlphaResult:
    return AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=n_bootstrap,
        n_effective=n_effective,
    )


def test_alpha_ci_not_degraded_when_drops_within_tolerance() -> None:
    # 50/1000 dropped = exactly 5% (the boundary): the flag ignores it (strict >),
    # so a handful of degenerate resamples is not read as a degraded CI.
    card = _card_with_alpha(_alpha(n_bootstrap=1000, n_effective=950))
    assert card.flags.alpha_ci_degraded is False
    assert "of 1000" not in card.to_markdown()


def test_alpha_ci_degraded_when_drops_exceed_tolerance() -> None:
    # 51/1000 dropped > 5%: just over the boundary flips the flag on.
    flags = _card_with_alpha(_alpha(n_bootstrap=1000, n_effective=949)).flags
    assert flags.alpha_ci_degraded is True


def test_report_warns_on_po_violation() -> None:
    from dataclasses import replace

    card = _card(converged=True)
    assert "proportional-odds" not in card.to_markdown().lower()

    violated = replace(card, dif=replace(card.dif, po_violation=True))
    md = violated.to_markdown()
    assert "proportional-odds" in md.lower()


def test_flags_property_returns_flags_instance() -> None:
    card = _card(converged=True)
    assert isinstance(card.flags, Flags)


def test_dif_converged_false_when_dif_did_not_converge() -> None:
    card = _card(converged=False)
    assert card.dif.converged is False


def test_dif_converged_true_when_dif_converged() -> None:
    card = _card(converged=True)
    assert card.dif.converged is True


def test_flags_conditioner_is_external_when_source_is_external() -> None:
    card = _card(converged=True, conditioner_source="external")
    assert card.flags.conditioner_is_external is True
    assert card.dif.conditioner_is_external is True  # also accessible on DifResult


def test_flags_conditioner_is_external_false_when_rest_score() -> None:
    card = _card(converged=True, conditioner_source="rest_score")
    assert card.flags.conditioner_is_external is False
    assert card.dif.conditioner_is_external is False


def test_dif_po_violation_false_by_default() -> None:
    card = _card(converged=True)
    assert card.dif.po_violation is False


def test_dif_po_violation_true_when_set() -> None:
    from dataclasses import replace

    card = _card(converged=True)
    violated = replace(card, dif=replace(card.dif, po_violation=True))
    assert violated.dif.po_violation is True


def test_flags_alpha_ci_degraded_when_resamples_dropped() -> None:
    alpha = AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=1000,
        n_effective=120,
    )
    card = _card_with_alpha(alpha)
    assert card.flags.alpha_ci_degraded is True


def test_flags_alpha_ci_degraded_false_when_all_resamples_realized() -> None:
    alpha = AlphaResult(
        alpha=0.6,
        ci_low=0.4,
        ci_high=0.8,
        level="nominal",
        n_bootstrap=1000,
        n_effective=1000,
    )
    card = _card_with_alpha(alpha)
    assert card.flags.alpha_ci_degraded is False


def test_flags_alpha_ci_degraded_true_when_thin_bootstrap_no_drops() -> None:
    # n_effective == n_bootstrap (no drops) but both below _MIN_EFFECTIVE=100;
    # degraded triggers via `not ci_reliable`, not via the drops branch.
    alpha = AlphaResult(
        alpha=0.6, ci_low=0.4, ci_high=0.8, level="nominal", n_bootstrap=50, n_effective=50
    )
    card = _card_with_alpha(alpha)
    assert card.flags.alpha_ci_degraded is True


def test_flags_removed_converged_raises_with_migration_hint() -> None:
    card = _card(converged=True)
    with pytest.raises(AttributeError, match=r"card\.dif\.converged"):
        _ = card.flags.converged  # type: ignore[attr-defined]


def test_flags_removed_po_violation_raises_with_migration_hint() -> None:
    card = _card(converged=True)
    with pytest.raises(AttributeError, match=r"card\.dif\.po_violation"):
        _ = card.flags.po_violation  # type: ignore[attr-defined]
