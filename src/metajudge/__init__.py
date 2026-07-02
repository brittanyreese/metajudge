"""metajudge: audit an LLM judge/rubric instrument."""

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
from metajudge.reliability import AlphaResult, IccResult, icc, krippendorff_alpha
from metajudge.report import Flags, ReportCard, audit

__version__ = "0.1.0"

__all__ = [
    "AlphaResult",
    "ClusterBootstrapDif",
    "DifResult",
    "DifSweep",
    "Flags",
    "IccResult",
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
