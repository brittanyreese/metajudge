# tests/test_public_api.py
import metajudge


def test_public_symbols_exported() -> None:
    for name in [
        "Ratings",
        "audit",
        "ReportCard",
        "Flags",
        "krippendorff_alpha",
        "AlphaResult",
        "icc",
        "IccResult",
        "logistic_dif",
        "DifResult",
        "cluster_bootstrap_dif",
        "ClusterBootstrapDif",
        "holm_adjust",
    ]:
        assert hasattr(metajudge, name), name


def test_removed_symbols_absent() -> None:
    for name in ["brant_test", "BrantResult"]:
        assert not hasattr(metajudge, name), f"{name} should not be in public API"
