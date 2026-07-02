"""metajudge: audit an LLM judge/rubric instrument before you trust its numbers.

Two pillars: inter-rater reliability (Krippendorff's alpha, ICC(2,1)/(2,k)) and
differential item functioning across output strata (ordinal logistic-regression DIF).
``audit`` renders both into a one-screen report card; the pillar functions and
:class:`~metajudge.data.Ratings` are usable standalone.
"""

from metajudge.data import Ratings
from metajudge.demo import load_demo
from metajudge.dif import (
    ClusterBootstrapDif,
    DifResult,
    DifSweep,
    cluster_bootstrap_dif,
    holm_adjust,
    logistic_dif,
    sweep,
)
from metajudge.reliability import (
    AlphaResult,
    IccResult,
    LevelOfMeasurement,
    icc,
    krippendorff_alpha,
)
from metajudge.report import Flags, ReportCard, audit

__version__ = "0.1.0"

__all__ = [
    "AlphaResult",
    "ClusterBootstrapDif",
    "DifResult",
    "DifSweep",
    "Flags",
    "IccResult",
    "LevelOfMeasurement",
    "Ratings",
    "ReportCard",
    "__version__",
    "audit",
    "cluster_bootstrap_dif",
    "holm_adjust",
    "icc",
    "krippendorff_alpha",
    "load_demo",
    "logistic_dif",
    "sweep",
]
