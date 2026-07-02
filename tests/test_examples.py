# tests/test_examples.py
"""Guards against examples/README-committed output drifting from what the code prints.

Regression test for the A2 review finding: README's demo block, examples/sample_output.txt,
and docs/interop-epic.md all fell behind main after a caveat-changing commit landed. This
runs the example and diffs its stdout against the committed sample, so that class of drift
fails CI instead of shipping silently.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# The cluster bootstrap's per-resample convergence is sensitive to platform-level
# floating-point/BLAS differences (e.g. macOS vs Linux CI runners), so n_effective can
# drift by a resample or two even with a fixed seed; normalize it rather than pin an
# exact count. This mirrors the already-documented n_effective drift nit (200 -> 199
# between v0.1.0 and HEAD).
_N_EFFECTIVE_RE = re.compile(r"n_effective=\d+ of \d+")


def _normalize(text: str) -> str:
    return _N_EFFECTIVE_RE.sub("n_effective=<n> of <n_boot>", text.strip())


def test_audit_summeval_output_matches_committed_sample(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(_EXAMPLES_DIR / "audit_summeval.py"), run_name="__main__")
    captured = capsys.readouterr()
    expected = (_EXAMPLES_DIR / "sample_output.txt").read_text()
    assert _normalize(captured.out) == _normalize(expected)
