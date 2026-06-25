from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="slow; run with --run-slow to enable")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
