# src/metajudge/report.py
"""Two-pillar report card: reliability + DIF."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from metajudge.data import Ratings
from metajudge.dif import DifResult, logistic_dif
from metajudge.reliability import (
    AlphaResult,
    IccResult,
    icc,
    krippendorff_alpha,
)


@dataclass(frozen=True)
class ReportCard:
    alpha: AlphaResult
    icc: IccResult
    dif: DifResult

    def to_markdown(self) -> str:
        a = self.alpha
        ic = self.icc
        d = self.dif
        if d.conditioner_source == "external":
            dif_header = "## DIF (instrument-level, external conditioner)"
            notes: list[str] = []
        else:
            dif_header = "## DIF (panel-relative, rest-score conditioner)"
            notes = [
                "> Note: the rest-score conditioner cannot see bias shared across the "
                "entire rater panel, so this is panel-relative DIF, not an instrument-level "
                "fairness clearance. Pass an external quality conditioner to test for "
                "instrument-level bias.",
                "",
            ]
        # The convergence warning and the panel-relative note sit ABOVE the statistics so a
        # reader who excerpts the headline numbers cannot drop them.
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
            f"- Krippendorff's alpha ({a.level}): {a.alpha:.3f} "
            f"[95% CI {a.ci_low:.3f}, {a.ci_high:.3f}]",
            f"- ICC(2,1): {ic.icc1:.3f}; ICC(2,k): {ic.icck:.3f} "
            f"({ic.n_targets} targets x {ic.n_raters} raters)",
            "",
            dif_header,
            *notes,
            f"- {d.focal_level} vs {d.reference_level} "
            f"(conditioner: {d.conditioner_source}, n={d.n_obs})",
            f"- Uniform DIF: chi2(1)={d.chi2_uniform:.2f}, p={d.p_uniform:.4f}",
            f"- Nonuniform DIF: chi2(1)={d.chi2_nonuniform:.2f}, p={d.p_nonuniform:.4f}",
            f"- Effect size (Nagelkerke R2 delta): {d.nagelkerke_r2_delta:.3f} "
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
