import metajudge


def test_package_exposes_version() -> None:
    assert isinstance(metajudge.__version__, str)
    assert metajudge.__version__
