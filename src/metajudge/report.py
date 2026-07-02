# src/metajudge/report.py
"""Two-pillar report card: reliability + DIF."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Never

from metajudge._constants import MIN_EFFECTIVE
from metajudge.data import Ratings
from metajudge.dif import (
    _JG_NEGLIGIBLE,  # pyright: ignore[reportPrivateUsage]  # one Jodoin-Gierl threshold, one source
    ClusterBootstrapDif,
    DifResult,
    cluster_bootstrap_dif,
    logistic_dif,
)
from metajudge.reliability import (
    AlphaResult,
    IccResult,
    LevelOfMeasurement,
    icc,
    krippendorff_alpha,
)

# Fraction of dropped bootstrap resamples below which the alpha CI is treated as fine: a
# few degenerate resamples out of a full run is sampling noise, not degradation.
_ALPHA_CI_DROP_TOLERANCE = 0.05


@dataclass(frozen=True)
class Flags:
    """Typed interpretation signals derived from a ReportCard.

    Each field answers a trustworthiness or scope question a caller
    (CLI, notebook, downstream renderer) might ask — without parsing strings.

    DIF convergence and proportional-odds signals live on ``ReportCard.dif`` directly
    (``dif.converged``, ``dif.po_violation``, ``dif.conditioner_is_external``); only
    the cross-pillar alpha CI quality signal belongs here.
    """

    conditioner_is_external: bool
    alpha_ci_degraded: bool
    dif_robustly_nonnegligible: bool | None
    """Clustering-robust DIF verdict, or ``None`` when it was not assessed.

    ``True`` when the item-cluster bootstrap's Nagelkerke R-squared-change CI lower bound
    clears the Jodoin-Gierl negligible band (robustly at least moderate DIF); ``False`` when
    the CI reaches into the negligible band. ``None`` when no bootstrap was run
    (``audit(robust=False)``) or the bootstrap CI is unreliable -- the analytic p-values on
    the card are anti-conservative under clustering and are not a robust significance test.
    """

    @property
    def converged(self) -> Never:
        raise AttributeError(
            "Flags.converged was removed in 0.1.0; use card.dif.converged directly."
        )

    @property
    def po_violation(self) -> Never:
        raise AttributeError(
            "Flags.po_violation was removed in 0.1.0; use card.dif.po_violation directly."
        )


@dataclass(frozen=True)
class ReportCard:
    alpha: AlphaResult
    icc: IccResult
    dif: DifResult
    dif_bootstrap: ClusterBootstrapDif | None = None

    @property
    def flags(self) -> Flags:
        return Flags(
            conditioner_is_external=self.dif.conditioner_is_external,
            alpha_ci_degraded=self._alpha_ci_degraded(),
            dif_robustly_nonnegligible=self._dif_robustly_nonnegligible(),
        )

    def _alpha_ci_degraded(self) -> bool:
        """Whether the alpha bootstrap CI is meaningfully degraded.

        True when the CI is not reliable (< MIN_EFFECTIVE resamples survived) or a
        non-trivial fraction of resamples was dropped. A handful of degenerate resamples out
        of a full run is normal sampling noise, not degradation, so the flag ignores drops
        below ``_ALPHA_CI_DROP_TOLERANCE``.
        """
        a = self.alpha
        if not a.ci_reliable:
            return True
        if a.n_bootstrap <= 0:
            return False
        dropped_frac = (a.n_bootstrap - a.n_effective) / a.n_bootstrap
        return dropped_frac > _ALPHA_CI_DROP_TOLERANCE

    def _dif_robustly_nonnegligible(self) -> bool | None:
        """Clustering-robust DIF verdict from the item-cluster bootstrap CI, or None.

        None when no bootstrap was run or its CI is unreliable (too few surviving resamples,
        or NaN bounds). Otherwise the verdict is whether the R-squared-change CI lower bound
        clears the Jodoin-Gierl negligible band.
        """
        bt = self.dif_bootstrap
        if bt is None or not bt.ci_reliable or math.isnan(bt.r2_delta_ci_low):
            return None
        return bt.r2_delta_ci_low >= _JG_NEGLIGIBLE

    def to_markdown(self) -> str:
        a = self.alpha
        ic = self.icc
        d = self.dif
        f = self.flags
        alpha_ci = (
            f"- Krippendorff's alpha ({a.level}): {a.alpha:.3f} "
            f"[95% CI {a.ci_low:.3f}, {a.ci_high:.3f}]"
        )
        if f.alpha_ci_degraded:
            details = [f"CI from {a.n_effective} of {a.n_bootstrap} bootstrap resamples"]
            dropped = a.n_bootstrap - a.n_effective
            if dropped:
                details.append(f"{dropped} degenerate resamples dropped")
            if not a.ci_reliable:
                details.append(
                    f"indicative only because fewer than {MIN_EFFECTIVE} resamples survived"
                )
            alpha_ci += f" ({'; '.join(details)})"
        if f.conditioner_is_external:
            dif_header = "## DIF (external conditioner)"
            notes = [
                "> Note: external-conditioner DIF supports instrument-level interpretation "
                "only when the conditioner is valid, independent, and appropriate for the "
                "quality construct being matched.",
                "",
            ]
        else:
            dif_header = "## DIF (panel-relative, rest-score conditioner)"
            notes = [
                "> Note: the rest-score conditioner cannot see bias shared across the "
                "entire rater panel, so this is panel-relative DIF, not an instrument-level "
                "fairness clearance. Pass a valid independent external quality conditioner "
                "for a stronger instrument-level analysis.",
                "",
            ]
        # Structural caveat, independent of conditioner source: each item belongs to exactly
        # one stratum, so group is an item-level property and the conditioner is matched
        # BETWEEN nested item sets rather than within items answered by both groups. When the
        # strata genuinely differ in quality, the conditioner correlates with the group and
        # the single linear match leaves residual confounding (DIF impurity) this screen does
        # not remove; at near-perfect confounding the engine refuses outright. See
        # docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md.
        # When this run's conditioner-group overlap is weak (conditioner_overlap_weak), the
        # blanket caveat above is replaced by a run-specific one: it names the actual
        # correlation and common-support numbers, because at that overlap the effect size can
        # absorb a real between-strata quality gap as apparent DIF rather than screen it out.
        if d.conditioner_overlap_weak:
            structural_note = [
                "> WARNING: residual-impurity regime. The conditioner correlates with the "
                f"group (correlation {d.conditioner_group_corr:.3f}, common support "
                f"{d.conditioner_common_support:.3f}) beyond the calibrated safe band "
                "(|corr| < 0.2), where the simulation study measured a materially elevated "
                "false B/C rate under no true DIF. The effect size below may absorb a real "
                "between-strata quality gap as apparent DIF instead of screening it out.",
                "",
            ]
        else:
            structural_note = [
                "> Note: strata nest items (each item is in one stratum), so this matches "
                "quality between nested item sets, not within shared items. If the strata "
                "differ in quality, the conditioner correlates with the group and residual "
                "confounding (DIF impurity) remains; read the effect size as screening "
                "evidence, not a confound-free fairness verdict.",
                "",
            ]
        notes = [*structural_note, *notes]
        # The convergence warning, the proportional-odds warning, and the panel-relative
        # note all sit ABOVE the statistics so a reader who excerpts the headline numbers
        # cannot drop them.
        if d.po_violation:
            notes = [
                "> WARNING: the proportional-odds assumption is violated (Brant test); "
                "the nonuniform-DIF test below may be unreliable, because a "
                "non-proportional response pattern can imitate a group x trait interaction "
                "(Harrell, 2015, Ch. 13).",
                "",
                *notes,
            ]
        if not d.converged:
            notes = [
                "> WARNING: the DIF model fit did not converge; the chi-square statistics "
                "and effect size below are unreliable and must not be acted on.",
                "",
                *notes,
            ]
        r2_str = "—" if math.isnan(d.nagelkerke_r2_delta) else f"{d.nagelkerke_r2_delta:.3f}"
        lines = [
            "# metajudge report card",
            "",
            "## Reliability",
            "> Note: high agreement (alpha, ICC) is not evidence the rubric measures the "
            "intended construct. It shows raters apply the scale consistently, not that the "
            "scale captures the quality you care about.",
            "",
            alpha_ci,
            f"- ICC(2,1): {ic.icc1:.3f} [95% CI {ic.icc1_ci_low:.3f}, {ic.icc1_ci_high:.3f}]; "
            f"ICC(2,k): {ic.icck:.3f} [95% CI {ic.icck_ci_low:.3f}, {ic.icck_ci_high:.3f}] "
            f"({ic.n_targets} targets x {ic.n_raters} raters)",
            "",
            dif_header,
            *notes,
            f"- {d.focal_level} vs {d.reference_level} "
            f"(conditioner: {d.conditioner_source}, n={d.n_obs})",
            # The screen's decision variable leads: effect-size magnitude and its A/B/C class.
            f"- Effect size (Nagelkerke R2 delta): {r2_str} (Jodoin-Gierl class {d.dif_class})",
            *self._robust_flag_lines(),
            # Analytic component tests: kept for the uniform/nonuniform shape, but explicitly
            # tagged, because the i.i.d. pooling makes their p-values anti-conservative under
            # the crossed rater x item design and NOT a clustering-robust significance test.
            f"- Uniform DIF: chi2(1)={d.chi2_uniform:.2f}, p={d.p_uniform:.4f} "
            "[analytic, unclustered]",
            f"- Nonuniform DIF: chi2(1)={d.chi2_nonuniform:.2f}, p={d.p_nonuniform:.4f} "
            "[analytic, unclustered]",
        ]
        return "\n".join(lines)

    def _robust_flag_lines(self) -> list[str]:
        """The clustering-robust DIF flag line(s): the bootstrap verdict, or 'not assessed'.

        When no bootstrap was run this states significance was not clustering-assessed and
        points to the robust path, so the tagged analytic p-values below are never read as a
        significance test. When a bootstrap is present it reports the R-squared-change CI and
        the verdict (or that the CI is indicative-only / unavailable).
        """
        bt = self.dif_bootstrap
        if bt is None:
            return [
                "- Clustering-robust significance: not assessed. The analytic p-values below "
                "are anti-conservative under the crossed rater x item design; run "
                "audit(robust=True) or cluster_bootstrap_dif() for a clustering-robust flag."
            ]
        pct = round(bt.ci_level * 100)
        counts = f"item-cluster bootstrap, {bt.n_effective}/{bt.n_boot} resamples"
        if math.isnan(bt.r2_delta_ci_low):
            return [
                f"- Clustering-robust flag: unavailable — all bootstrap resamples were "
                f"degenerate ({counts})."
            ]
        method = "BCa" if bt.ci_method == "bca" else "percentile"
        ci = (
            f"R2 delta {pct}% {method} CI [{bt.r2_delta_ci_low:.3f}, {bt.r2_delta_ci_high:.3f}], "
            f"{counts}"
        )
        verdict = self._dif_robustly_nonnegligible()
        if verdict is None:
            return [
                f"- Clustering-robust flag: indicative only — fewer than 100 resamples "
                f"survived, so the CI is not trustworthy ({ci}). Read the point estimate."
            ]
        label = (
            "robustly non-negligible DIF"
            if verdict
            else "no robust DIF (CI reaches the negligible band)"
        )
        if bt.ci_method == "bca":
            caveat = (
                "  Caveat: the flag compares a bias-corrected accelerated (BCa) CI bound to the "
                f"Jodoin-Gierl {_JG_NEGLIGIBLE:.3f} boundary. BCa corrects the percentile bias, "
                "but the bound is still an estimate; read the point estimate and CI alongside "
                "the label."
            )
        else:
            caveat = (
                "  Caveat: the flag rests on a percentile CI bound compared to the "
                f"Jodoin-Gierl {_JG_NEGLIGIBLE:.3f} boundary (BCa was undefined at this "
                "0-bounded statistic). The percentile method is least accurate at the boundary, "
                "so a verdict from a bound sitting near it is fragile; read the point estimate "
                "and CI, not the label alone."
            )
        return [f"- Clustering-robust flag: {label} ({ci}).", caveat]


def audit(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    level: LevelOfMeasurement | str = "ordinal",
    seed: int = 0,
    conditioner: Mapping[Hashable, float] | None = None,
    robust: bool = False,
    n_boot: int = 1000,
    po_alpha: float = 1e-3,
) -> ReportCard:
    """Build the two-pillar report card.

    ``robust=False`` (default) is the fast analytic screen: it reports the DIF effect size
    and A/B/C class but does not assess clustering-robust significance. ``robust=True`` runs
    the item-cluster bootstrap (``n_boot`` resamples) so the card can report a
    clustering-robust flag from the Nagelkerke R-squared-change CI, at the cost of the extra
    refits. The analytic p-values are anti-conservative under the crossed rater x item
    design either way, and the card tags them as such.
    """
    if robust:
        bootstrap = cluster_bootstrap_dif(
            ratings,
            focal=focal,
            reference=reference,
            conditioner=conditioner,
            n_boot=n_boot,
            seed=seed,
            po_alpha=po_alpha,
        )
        dif = bootstrap.base
    else:
        bootstrap = None
        dif = logistic_dif(
            ratings, focal=focal, reference=reference, conditioner=conditioner, po_alpha=po_alpha
        )
    return ReportCard(
        alpha=krippendorff_alpha(ratings, level=level, seed=seed),
        icc=icc(ratings),
        dif=dif,
        dif_bootstrap=bootstrap,
    )
