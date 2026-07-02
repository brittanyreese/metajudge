# tests/test_examples.py
"""Guards against examples/README-committed output drifting from what the code prints.

Regression test for the A2 review finding: README's demo block, examples/sample_output.txt,
and docs/interop-epic.md all fell behind main after a caveat-changing commit landed. This
runs the example and diffs its stdout against the committed sample, so that class of drift
fails CI instead of shipping silently.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_audit_summeval_output_matches_committed_sample(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(_EXAMPLES_DIR / "audit_summeval.py"), run_name="__main__")
    captured = capsys.readouterr()
    expected = (_EXAMPLES_DIR / "sample_output.txt").read_text()
    assert captured.out.strip() == expected.strip()
