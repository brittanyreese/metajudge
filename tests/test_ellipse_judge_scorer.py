# tests/test_ellipse_judge_scorer.py
"""Unit tests for the ELLIPSE LLM-judge scorer's per-trait call design.

Pins the fix for the halo/anchoring bug found when scoring all 7 traits in one call
(confirmed on GPT-4o: batched, it collapsed every trait to one repeated value on most
essays; asked one trait at a time, it differentiated normally). These tests assert the
per-trait contract directly: one isolated call per trait, all-or-nothing aggregation, no
partial (some-traits-missing) result ever returned.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import _ellipse_judge as ej


def test_parse_score_accepts_int_string_and_clamped_float() -> None:
    assert ej.parse_score('{"Grammar": 3}', "Grammar") == 3
    assert ej.parse_score('{"Grammar": "4"}', "Grammar") == 4
    assert ej.parse_score('{"Grammar": 2.0}', "Grammar") == 2
    assert ej.parse_score('{"Grammar": 9}', "Grammar") == 5  # clamped to the 1-5 scale
    assert ej.parse_score('{"Grammar": 0}', "Grammar") == 1  # clamped to the 1-5 scale


def test_parse_score_rejects_missing_key_bool_and_bad_json() -> None:
    assert ej.parse_score('{"Syntax": 3}', "Grammar") is None  # wrong key
    assert ej.parse_score('{"Grammar": true}', "Grammar") is None  # bool, not an int
    assert ej.parse_score("not json", "Grammar") is None


def test_build_messages_isolates_the_single_trait() -> None:
    messages = ej.build_messages("some essay text", "Grammar")
    user = messages[1]["content"]
    assert "Grammar" in user
    # The other 6 traits' keyword rubrics must not leak into a single-trait prompt --
    # that leakage is exactly what the batched design did, and reintroducing it would
    # reopen the halo/anchoring bug this test file exists to pin against.
    for other_trait, keywords in ej.TRAIT_KEYWORDS.items():
        if other_trait != "Grammar":
            assert keywords not in user


def test_score_essay_makes_one_call_per_trait(monkeypatch: object) -> None:
    calls: list[str] = []

    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> str:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        calls.append(trait)
        return f'{{"{trait}": 3}}'

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", "some essay text")

    assert result.scores == dict.fromkeys(ej.RUBRIC_TRAITS, 3)
    assert result.attempts == len(ej.RUBRIC_TRAITS)
    assert calls == list(ej.RUBRIC_TRAITS)  # exactly one isolated call per trait, in order


def test_score_essay_fails_closed_with_no_partial_result(monkeypatch: object) -> None:
    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> str:
        return "unparseable"

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    config = ej.JudgeConfig(max_retries=1)
    result = ej.score_essay(config, "essay-1", "some essay text")

    assert result.scores is None
    assert result.attempts == config.max_retries + 1  # stops at the first trait, retried
