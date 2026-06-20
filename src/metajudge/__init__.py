"""metajudge: audit an LLM judge/rubric instrument."""

from metajudge.data import Ratings
from metajudge.demo import load_demo
from metajudge.dif import (
    ClusterBootstrapDif,
    DifResult,
    cluster_bootstrap_dif,
    logistic_dif,
)
from metajudge.reliability import AlphaResult, IccResult, icc, krippendorff_alpha
from metajudge.report import ReportCard, audit

__version__ = "0.1.0"

__all__ = [
    "AlphaResult",
    "ClusterBootstrapDif",
    "DifResult",
    "IccResult",
    "Ratings",
    "ReportCard",
    "__version__",
    "audit",
    "cluster_bootstrap_dif",
    "icc",
    "krippendorff_alpha",
    "load_demo",
    "logistic_dif",
]
