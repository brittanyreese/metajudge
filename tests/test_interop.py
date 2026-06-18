# tests/test_interop.py
"""Epic evaluation-instruments interop: frame_from_evals output -> Ratings.

The measurement frame is research-locked (see the interop ADR): rater = judge
instance, item = evaluated sample, score = one selected rubric criterion. Rubric
criteria are a separate facet, audited one at a time, never treated as raters.
"""

import pandas as pd
import pytest

from metajudge.data import Ratings


def _flat_judge_frame(scores: dict[str, list[int]], index: list[str]) -> pd.DataFrame:
    # Epic frame_from_evals "single score" shape: rows = samples, columns = criteria.
    return pd.DataFrame(scores, index=index)


def _detailed_judge_frame(
    by_criterion: dict[str, dict[str, list[object]]], index: list[str]
) -> pd.DataFrame:
    # Epic frame_from_evals "nested" shape: MultiIndex columns (criterion, field),
    # field in {class, score, notes}.
    data = {
        (crit, field): vals
        for crit, fields in by_criterion.items()
        for field, vals in fields.items()
    }
    return pd.DataFrame(data, index=index, columns=pd.MultiIndex.from_tuples(data.keys()))


def test_from_eval_instruments_flat_two_judges() -> None:
    j1 = _flat_judge_frame({"organization": [5, 3], "accuracy": [4, 2]}, ["s1", "s2"])
    j2 = _flat_judge_frame({"organization": [4, 3], "accuracy": [5, 1]}, ["s1", "s2"])

    r = Ratings.from_eval_instruments({"judge_a": j1, "judge_b": j2}, criterion="organization")

    assert r.n_items == 2
    assert r.n_raters == 2
    wide = r.wide()  # items x raters
    assert wide.loc["s1", "judge_a"] == 5
    assert wide.loc["s1", "judge_b"] == 4
    assert wide.loc["s2", "judge_a"] == 3
    assert wide.loc["s2", "judge_b"] == 3


def test_from_eval_instruments_detailed_selects_score_field() -> None:
    # Nested shape: the adapter must pull the (criterion, "score") column, not class/notes.
    j1 = _detailed_judge_frame(
        {
            "organization": {"class": ["strong", "weak"], "score": [5, 2], "notes": ["a", "b"]},
            "accuracy": {"class": ["weak", "strong"], "score": [2, 5], "notes": ["c", "d"]},
        },
        ["s1", "s2"],
    )
    j2 = _detailed_judge_frame(
        {
            "organization": {"class": ["moderate", "weak"], "score": [3, 2], "notes": ["e", "f"]},
            "accuracy": {"class": ["strong", "none"], "score": [5, 1], "notes": ["g", "h"]},
        },
        ["s1", "s2"],
    )

    r = Ratings.from_eval_instruments({"judge_a": j1, "judge_b": j2}, criterion="organization")

    wide = r.wide()
    assert wide.loc["s1", "judge_a"] == 5
    assert wide.loc["s2", "judge_b"] == 2


def test_from_eval_instruments_carries_stratum() -> None:
    j1 = _flat_judge_frame({"organization": [5, 3]}, ["s1", "s2"])
    j2 = _flat_judge_frame({"organization": [4, 2]}, ["s1", "s2"])

    r = Ratings.from_eval_instruments(
        {"judge_a": j1, "judge_b": j2},
        criterion="organization",
        stratum={"s1": "abstractive", "s2": "extractive"},
    )

    assert r.strata() == {"abstractive": ["s1"], "extractive": ["s2"]}


def test_from_eval_instruments_unknown_criterion_raises() -> None:
    j1 = _flat_judge_frame({"organization": [5, 3]}, ["s1", "s2"])
    with pytest.raises(ValueError, match="criterion"):
        Ratings.from_eval_instruments({"judge_a": j1}, criterion="not_a_column")


def test_from_eval_instruments_detailed_unknown_criterion_raises() -> None:
    j1 = _detailed_judge_frame(
        {"organization": {"class": ["strong", "weak"], "score": [5, 2], "notes": ["a", "b"]}},
        ["s1", "s2"],
    )
    with pytest.raises(ValueError, match="criterion"):
        Ratings.from_eval_instruments({"judge_a": j1}, criterion="not_a_column")


def test_from_eval_instruments_empty_frames_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Ratings.from_eval_instruments({}, criterion="organization")
