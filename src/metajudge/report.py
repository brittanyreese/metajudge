# src/metajudge/report.py
"""Two-pillar report card: reliability + DIF."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Never

from metajudge.data import Ratings
from metajudge.dif import DifResult, logistic_dif
from metajudge.reliability import (
    AlphaResult,
    IccResult,
    icc,
    krippendorff_alpha,
)


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

    @property
    def flags(self) -> Flags:
        return Flags(
            conditioner_is_external=self.dif.conditioner_is_external,
            alpha_ci_degraded=(
                self.alpha.n_effective < self.alpha.n_bootstrap or not self.alpha.ci_reliable
            ),
        )

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
                details.append("indicative only because fewer than 100 resamples survived")
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
        lines = [
            "# metajudge report card",
            "",
            "## Reliability",
            alpha_ci,
            f"- ICC(2,1): {ic.icc1:.3f}; ICC(2,k): {ic.icck:.3f} "
            f"({ic.n_targets} targets x {ic.n_raters} raters)",
            "",
            dif_header,
            *notes,
            f"- {d.focal_level} vs {d.reference_level} "
            f"(conditioner: {d.conditioner_source}, n={d.n_obs})",
            f"- Uniform DIF: chi2(1)={d.chi2_uniform:.2f}, p={d.p_uniform:.4f}",
            f"- Nonuniform DIF: chi2(1)={d.chi2_nonuniform:.2f}, p={d.p_nonuniform:.4f}",
            f"- Effect size (Nagelkerke R2 delta): "
            f"{'—' if math.isnan(d.nagelkerke_r2_delta) else f'{d.nagelkerke_r2_delta:.3f}'} "
            f"(Jodoin-Gierl class {d.dif_class})",
        ]
        return "\n".join(lines)


def audit(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    level: str = "ordinal",
    seed: int = 0,
    conditioner: Mapping[Hashable, float] | None = None,
) -> ReportCard:
    return ReportCard(
        alpha=krippendorff_alpha(ratings, level=level, seed=seed),
        icc=icc(ratings),
        dif=logistic_dif(ratings, focal=focal, reference=reference, conditioner=conditioner),
    )
