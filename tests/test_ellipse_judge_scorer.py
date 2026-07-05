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

# Bound once here (each needs the pyright ignore); every call site below uses the local
# name, so this is the only place reportPrivateUsage needs suppressing.
_ChatResponse = ej._ChatResponse  # type: ignore[reportPrivateUsage]
_build_payload = ej._build_payload  # type: ignore[reportPrivateUsage]


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

    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        calls.append(trait)
        return _ChatResponse(
            content=f'{{"{trait}": 3}}', system_fingerprint=None, prompt_tokens=None
        )

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", "some essay text")

    assert result.scores == dict.fromkeys(ej.RUBRIC_TRAITS, 3)
    assert result.attempts == len(ej.RUBRIC_TRAITS)
    assert calls == list(ej.RUBRIC_TRAITS)  # exactly one isolated call per trait, in order


def test_score_essay_fails_closed_with_no_partial_result(monkeypatch: object) -> None:
    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        return _ChatResponse(content="unparseable", system_fingerprint=None, prompt_tokens=None)

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    config = ej.JudgeConfig(max_retries=1)
    result = ej.score_essay(config, "essay-1", "some essay text")

    assert result.scores is None
    assert result.attempts == config.max_retries + 1  # stops at the first trait, retried


def test_build_messages_default_path_is_unchanged() -> None:
    # Reproducibility guard: the committed pilot CSV and its pinned effect sizes were
    # scored with the default prompt. The new opt-in flags must not perturb it, so the
    # default output must still carry the holistic SCALE_ANCHORS and no reasoning field.
    user = ej.build_messages("essay text", "Grammar")[1]["content"]
    assert ej.SCALE_ANCHORS in user
    assert "reasoning" not in user
    assert 'Return exactly this JSON shape with an integer value 1-5: {"Grammar": <int>}' in user


def test_trait_scoped_anchors_name_the_trait_and_differ_across_traits() -> None:
    # The root cause of cross-call collapse: the 360-char holistic SCALE_ANCHORS is
    # identical on all 7 calls and describes whole-essay proficiency, re-priming one
    # holistic score per trait. Trait-scoped anchors must name the trait and differ.
    grammar = ej.build_messages("essay", "Grammar", trait_scoped_anchors=True)[1]["content"]
    syntax = ej.build_messages("essay", "Syntax", trait_scoped_anchors=True)[1]["content"]
    assert "Grammar" in grammar and "Syntax" in syntax
    # The scale-anchor block itself must differ between the two traits, not just the
    # keyword line (which already differs in the default path).
    assert ej.trait_scale_anchors("Grammar") != ej.trait_scale_anchors("Syntax")
    assert ej.SCALE_ANCHORS not in grammar  # holistic block replaced, not appended


def test_reasoning_prompt_asks_for_reasoning_before_the_score() -> None:
    user = ej.build_messages("essay", "Vocabulary", reasoning=True)[1]["content"]
    system = ej.build_messages("essay", "Vocabulary", reasoning=True)[0]["content"]
    assert "reasoning" in user
    # Reasoning key must precede the score key so the model writes it before committing.
    assert user.index("reasoning") < user.index('"Vocabulary"')
    # The default system prompt forbids explanation; the reasoning variant must not.
    assert "no explanation" not in system


def test_parse_score_ignores_a_reasoning_key() -> None:
    # The reasoning variant returns {"reasoning": "...", "<trait>": <int>}; the parser
    # must still pull the trait's integer and ignore the extra key.
    assert ej.parse_score('{"reasoning": "many errors", "Grammar": 2}', "Grammar") == 2


def test_score_essay_threads_prompt_flags_from_config(monkeypatch: object) -> None:
    seen: list[str] = []

    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        seen.append(messages[1]["content"])
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        return _ChatResponse(
            content=f'{{"{trait}": 3}}', system_fingerprint=None, prompt_tokens=None
        )

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    config = ej.JudgeConfig(reasoning=True, trait_scoped_anchors=True)
    ej.score_essay(config, "essay-1", "some essay text")

    assert all("reasoning" in prompt for prompt in seen)
    assert all(ej.SCALE_ANCHORS not in prompt for prompt in seen)


def test_build_payload_pins_openrouter_to_openai_only_by_default() -> None:
    config = ej.JudgeConfig(base_url="https://openrouter.ai/api/v1", model="openai/gpt-4o")
    payload = _build_payload(config, [])
    assert payload["provider"] == ej.DEFAULT_OPENROUTER_PROVIDER


def test_build_payload_respects_explicit_provider_override() -> None:
    config = ej.JudgeConfig(base_url="https://openrouter.ai/api/v1", provider={"order": ["azure"]})
    payload = _build_payload(config, [])
    assert payload["provider"] == {"order": ["azure"]}


def test_build_payload_omits_provider_for_non_openrouter_backend() -> None:
    config = ej.JudgeConfig(base_url=ej.DEFAULT_BASE_URL)
    payload = _build_payload(config, [])
    assert "provider" not in payload


def test_score_essay_flags_fingerprint_change_across_calls(monkeypatch: object) -> None:
    fingerprints = iter(["fp_a", "fp_a", "fp_b", "fp_b", "fp_b", "fp_b", "fp_b"])

    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        return _ChatResponse(
            content=f'{{"{trait}": 3}}', system_fingerprint=next(fingerprints), prompt_tokens=500
        )

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", "x" * 100)

    assert result.fingerprint_changed is True
    assert result.system_fingerprint == "fp_b"  # last fingerprint seen


def test_score_essay_does_not_flag_fingerprint_change_when_stable(monkeypatch: object) -> None:
    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        return _ChatResponse(
            content=f'{{"{trait}": 3}}', system_fingerprint="fp_a", prompt_tokens=500
        )

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", "x" * 100)

    assert result.fingerprint_changed is False
    assert result.system_fingerprint == "fp_a"


def test_score_essay_flags_truncation_when_prompt_tokens_below_essay_estimate(
    monkeypatch: object,
) -> None:
    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        return _ChatResponse(content=f'{{"{trait}": 3}}', system_fingerprint="fp", prompt_tokens=5)

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    long_essay = "word " * 2000  # ~10,000 chars; prompt_tokens=5 is implausibly small
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", long_essay)

    assert result.truncated is True


def test_score_essay_not_truncated_when_prompt_tokens_plausible(monkeypatch: object) -> None:
    def fake_chat_once(config: ej.JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
        trait = next(t for t in ej.RUBRIC_TRAITS if t in messages[1]["content"])
        return _ChatResponse(
            content=f'{{"{trait}": 3}}', system_fingerprint="fp", prompt_tokens=9999
        )

    monkeypatch.setattr(ej, "_chat_once", fake_chat_once)  # type: ignore[attr-defined]
    result = ej.score_essay(ej.JudgeConfig(), "essay-1", "short essay")

    assert result.truncated is False
