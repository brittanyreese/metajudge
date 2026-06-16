"""metajudge: audit an LLM judge/rubric instrument."""

from metajudge.data import Ratings
from metajudge.demo import load_demo
from metajudge.dif import DifResult, logistic_dif
from metajudge.reliability import AlphaResult, IccResult, icc, krippendorff_alpha
from metajudge.report import ReportCard, audit

__version__ = "0.1.0"

__all__ = [
    "AlphaResult",
    "DifResult",
    "IccResult",
    "Ratings",
    "ReportCard",
    "__version__",
    "audit",
    "icc",
    "krippendorff_alpha",
    "load_demo",
    "logistic_dif",
]
