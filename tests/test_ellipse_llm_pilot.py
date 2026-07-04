# tests/test_ellipse_llm_pilot.py
"""Fast pinned test on the committed ELLIPSE LLM-judge pilot scores.

The pilot CSV (``examples/data/ellipse_llm_pilot_qwen2.5-7b.csv``) is a real qwen2.5:7b
scoring run committed for reproducibility, so the DIF audit reproduces with no GPU or model.
This test rebuilds the single-judge DIF for the pre-registered pilot traits straight from
that CSV (no example-module import, no network) and pins the analytic effect sizes and
Jodoin-Gierl classes, so a change to the committed scores or the DIF engine that would move
the reported human-vs-LLM contrast fails CI instead of shipping silently.

The literals are the values produced by the committed CSV; if the DIF engine's reference
oracle changes them, the reference wins and these are updated (the numerical-reference
convention), never loosened to pass.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from pathlib import Path

import pandas as pd
import pytest

from metajudge.data import Ratings
from metajudge.dif import logistic_dif

_CSV = Path(__file__).parent.parent / "examples" / "data" / "ellipse_llm_pilot_qwen2.5-7b.csv"
_FOCAL = "Asian/Pacific Islander"
_REFERENCE = "Hispanic/Latino"
_ANALYTIC = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]

# Pinned from the committed qwen2.5:7b pilot CSV: (Nagelkerke R2 delta, Jodoin-Gierl class).
# All three are class A (negligible) -- the pilot judge shows no consequential DIF on the
# Yamashita-flagged focal/reference pair.
_EXPECTED: dict[str, tuple[float, str]] = {
    "Vocabulary": (0.0042, "A"),
    "Syntax": (0.0009, "A"),
    "Overall": (0.0078, "A"),
}


def _conditioner(scores: pd.DataFrame, trait: str) -> dict[Hashable, float]:
    """Leave-one-trait-out external conditioner from the judge's own other-trait scores."""
    others = [t for t in _ANALYTIC if t != trait]
    values = scores[others].mean(axis=1)
    return dict(zip(scores["text_id_kaggle"].astype(str).tolist(), values.tolist(), strict=True))


def _ratings(scores: pd.DataFrame, trait: str) -> Ratings:
    long = pd.DataFrame(
        {
            "item": scores["text_id_kaggle"].astype(str),
            "rater": "llm_judge",
            "score": scores[trait],
            "stratum": scores["race_ethnicity"],
        }
    )
    return Ratings.from_long(long, item="item", rater="rater", score="score", stratum="stratum")


def test_pilot_csv_shape_is_balanced() -> None:
    scores = pd.read_csv(_CSV)
    assert len(scores) == 300
    assert scores["text_id_kaggle"].duplicated().sum() == 0
    counts = scores["race_ethnicity"].value_counts().to_dict()
    assert counts == {_FOCAL: 150, _REFERENCE: 150}


@pytest.mark.parametrize("trait", list(_EXPECTED))
def test_pilot_dif_is_pinned_and_negligible(trait: str) -> None:
    scores = pd.read_csv(_CSV)
    with warnings.catch_warnings():
        # n_obs=300 is in the 200-499 "indicative" band; the warning is expected here.
        warnings.simplefilter("ignore", UserWarning)
        result = logistic_dif(
            _ratings(scores, trait),
            focal=_FOCAL,
            reference=_REFERENCE,
            conditioner=_conditioner(scores, trait),
        )
    expected_r2, expected_class = _EXPECTED[trait]
    assert result.n_obs == 300
    assert result.converged
    assert result.dif_class == expected_class
    assert result.nagelkerke_r2_delta == pytest.approx(expected_r2, abs=5e-4)
