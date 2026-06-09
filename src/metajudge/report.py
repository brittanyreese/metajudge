# src/metajudge/report.py
"""Two-pillar report card: reliability + DIF."""

from __future__ import annotations

from dataclasses import dataclass

from metajudge.data import Ratings
from metajudge.dif import DifResult, mantel_haenszel_dif
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
        return "\n".join(
            [
                "# metajudge report card",
                "",
                "## Reliability",
                f"- Krippendorff's alpha ({a.level}): {a.alpha:.3f} "
                f"[95% CI {a.ci_low:.3f}, {a.ci_high:.3f}]",
                f"- ICC(2,1): {ic.icc1:.3f}; ICC(2,k): {ic.icck:.3f} "
                f"({ic.n_targets} targets x {ic.n_raters} raters)",
                "",
                "## DIF",
                f"- {d.focal_level} vs {d.reference_level}: "
                f"common OR {d.common_odds_ratio:.3f}, MH-D {d.mh_delta:.2f}, "
                f"p={d.p_value:.4f}",
                f"- ETS classification: {d.ets_class}",
            ]
        )


def audit(
    ratings: Ratings,
    *,
    focal: str,
    reference: str,
    level: str = "ordinal",
    seed: int = 0,
) -> ReportCard:
    return ReportCard(
        alpha=krippendorff_alpha(ratings, level=level, seed=seed),
        icc=icc(ratings),
        dif=mantel_haenszel_dif(ratings, focal=focal, reference=reference),
    )
