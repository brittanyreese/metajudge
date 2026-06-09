# tests/test_public_api.py
import metajudge


def test_public_symbols_exported() -> None:
    for name in [
        "Ratings",
        "audit",
        "ReportCard",
        "krippendorff_alpha",
        "AlphaResult",
        "icc",
        "IccResult",
        "mantel_haenszel_dif",
        "DifResult",
    ]:
        assert hasattr(metajudge, name), name
