"""Self-contained LLM-judge scorer for the ELLIPSE analytic rubric.

OpenAI-compatible and dependency-free (stdlib ``urllib`` only, no ``openai`` SDK, no
private infrastructure). It speaks the ``/v1/chat/completions`` schema, so it works
against ANY endpoint that does: a local ``mlx_lm.server``, Ollama's OpenAI-compatible
``/v1``, or a hosted API such as OpenRouter. ``base_url`` / ``model`` / ``api_key`` come
from arguments or environment variables (``ELLIPSE_JUDGE_BASE_URL`` /
``ELLIPSE_JUDGE_MODEL`` / ``ELLIPSE_JUDGE_API_KEY``), defaulting to a local Ollama server.

Determinism and provenance are pinned in this file: ``temperature=0``, a fixed ``seed``,
the keyword rubric, the scale anchors, and the full prompt template all live in the
committed code, so a score CSV is reproducible from a pinned (model, prompt, seed) triple.
For models that emit chain-of-thought, thinking is disabled two ways: a ``/no_think``
directive in the system prompt (honored by Qwen3 / GLM style models) and a defensive
``<think>...</think>`` strip on the raw completion.

Each essay is scored with ONE CHAT-COMPLETION CALL PER TRAIT, never all 7 traits in a single
call. Batching all 7 traits into one JSON response was tried first and induces halo/anchoring
in some judges (confirmed on GPT-4o: batched, it collapsed all 7 scores to one repeated value
on the majority of essays; asked one trait at a time, it differentiated normally). Per-trait
calls cost more tokens (~7x the calls) but that is the correctness floor, not a shortcut to
skip.

The judge asks for STRUCTURED JSON (one integer rubric score for the trait) per call and
parses it robustly, with a bounded per-trait retry on parse failure; an essay where any trait
still fails to parse after retries is returned as a ``JudgeResult`` with ``scores=None`` and
the offending raw text, never silently dropped or partially filled.

Per-backend reproducibility pins (`docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md`,
"Reproducibility protocol per backend"):

- **OpenRouter.** A model slug like ``openai/gpt-4o`` load-balances across upstream
  providers (OpenAI direct, Azure-hosted OpenAI were both observed serving it in a
  spot-check, each with a different ``system_fingerprint``). When ``base_url`` is
  OpenRouter and ``JudgeConfig.provider`` is not set, the request is auto-pinned to
  ``DEFAULT_OPENROUTER_PROVIDER`` (OpenAI-only route, fail closed if a provider would
  silently drop ``seed``).
- **OpenAI / OpenRouter fingerprint.** Every response's ``system_fingerprint`` is
  captured on ``JudgeResult``; if it changes across an essay's 7 per-trait calls,
  ``fingerprint_changed=True`` flags that essay's scores as non-comparable.
- **Ollama.** The OpenAI-compatible ``/v1/chat/completions`` endpoint does NOT accept a
  per-request ``options.num_ctx`` (verified against Ollama's own OpenAI-compatibility
  docs: the endpoint's supported request fields do not include ``options``); the context
  window can only be raised server-side, via the ``OLLAMA_CONTEXT_LENGTH`` env var on
  ``ollama serve`` or a ``PARAMETER num_ctx <n>`` line in a custom Modelfile. What CAN be
  checked client-side is the response's ``usage.prompt_tokens`` (Ollama's OpenAI-compat
  equivalent of ``prompt_eval_count``): ``JudgeResult.truncated`` flags a value
  implausibly small for the essay's own length as a truncation tripwire.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

# The seven ELLIPSE score dimensions: one holistic (Overall) plus six analytic traits.
RUBRIC_TRAITS: tuple[str, ...] = (
    "Overall",
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
)

# Keyword-style analytic descriptors distilled from the official ELLIPSE rubric
# (ELL_Rubrics.docx, Crossley et al. 2023). Keyword prompts -- a short list of the
# salient scoring cues per trait -- outperform long paragraph-rubric prompts for LLM
# automated essay scoring; each trait below is reduced to its rubric cues rather than the
# full multi-sentence descriptor.
TRAIT_KEYWORDS: Mapping[str, str] = {
    "Overall": (
        "holistic English proficiency: syntactic variety, range of words and phrases, "
        "text organization, grammar and conventions accuracy, overall communication clarity"
    ),
    "Cohesion": (
        "text organization; cohesive devices (reference and transitional words/phrases); "
        "logical sequencing; overlap of ideas across sentences and paragraphs"
    ),
    "Syntax": (
        "variety of sentence structures (simple/compound/complex); correctness of sentence "
        "formation; word order; clause coordination and dependency"
    ),
    "Vocabulary": (
        "range and precision of word choice; topic-related terms; less common words; "
        "accuracy of word use and word forms"
    ),
    "Phraseology": (
        "use of phrases -- idioms, collocations, lexical bundles; variety; precision; "
        "repetition or misuse of phrases"
    ),
    "Grammar": (
        "grammatical accuracy; word-class and morphology correctness; frequency of "
        "grammar and usage errors"
    ),
    "Conventions": (
        "spelling, capitalization, and punctuation accuracy; consistency of mechanical conventions"
    ),
}

# 1-5 scale anchors, condensed from the rubric's per-level holistic descriptors.
SCALE_ANCHORS: str = (
    "5 = native-like facility, rare negligible errors; "
    "4 = controlled with occasional errors that rarely impede communication; "
    "3 = limited to common structures and generic usage, errors sometimes impede; "
    "2 = inconsistent, many errors that impede communication in many instances; "
    "1 = limited range loosely strung together, pervasive errors that impede in most cases."
)

_ENV_BASE_URL = "ELLIPSE_JUDGE_BASE_URL"
_ENV_MODEL = "ELLIPSE_JUDGE_MODEL"
_ENV_API_KEY = "ELLIPSE_JUDGE_API_KEY"

# Local Ollama OpenAI-compatible endpoint and an installed model, as pilot defaults. No
# private URL: swap to a larger local model (qwen2.5:32b) or the reported run
# (mlx_lm.server, Llama-3.1-70B) with a base_url + model change. The default is the small
# Qwen sibling qwen2.5:7b: on the pilot hardware the 32b ran at 25-46 s/essay under memory
# pressure, so the committed pilot uses the 7b (same family, ~10 s/essay) to stay in-session.
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"

# OpenRouter provider-routing pin (see module docstring). Restricts a load-balanced model
# slug to the OpenAI-direct route and fails closed rather than silently dropping `seed`.
OPENROUTER_HOST = "openrouter.ai"
DEFAULT_OPENROUTER_PROVIDER: Mapping[str, object] = {
    "order": ["openai"],
    "allow_fallbacks": False,
    "require_parameters": True,
}

# ponytail: crude chars-per-token heuristic (no tokenizer dependency); revisit if it
# false-positives on a real judge's usage.prompt_tokens.
_CHARS_PER_TOKEN_ESTIMATE = 4

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeConfig:
    """Pinned connection and decoding settings for one scoring run.

    ``base_url`` / ``model`` / ``api_key`` default to environment variables, then to a
    local Ollama server. ``temperature`` and ``seed`` are pinned for determinism;
    ``max_retries`` bounds the reparse loop per essay. ``provider`` overrides the
    OpenRouter provider-routing object; leave it ``None`` to get the auto-pinned
    OpenAI-only route on an OpenRouter ``base_url`` (see module docstring).
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = "not-needed"
    temperature: float = 0.0
    seed: int = 0
    timeout_s: float = 120.0
    max_retries: int = 2
    provider: Mapping[str, object] | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> JudgeConfig:
        """Build a config from ``ELLIPSE_JUDGE_*`` env vars, then explicit overrides.

        Any keyword in ``overrides`` that is not ``None`` wins over the environment and
        the defaults; ``None`` overrides are ignored so a CLI can pass through unset flags.
        """
        base = {
            "base_url": os.environ.get(_ENV_BASE_URL, DEFAULT_BASE_URL),
            "model": os.environ.get(_ENV_MODEL, DEFAULT_MODEL),
            "api_key": os.environ.get(_ENV_API_KEY, "not-needed"),
        }
        for key, value in overrides.items():
            if value is not None:
                base[key] = value  # type: ignore[assignment]
        return cls(**base)  # type: ignore[arg-type]


@dataclass(frozen=True)
class JudgeResult:
    """One essay's scoring outcome.

    ``scores`` maps each rubric trait to an integer 1-5 on success, or is ``None`` when
    every attempt failed to parse; ``raw`` keeps the last raw completion for logging, and
    ``attempts`` counts how many requests were made. ``system_fingerprint`` is the last
    fingerprint seen across this essay's calls; ``fingerprint_changed`` is True if the 7
    per-trait calls did not all report the same one (non-comparable scores; see module
    docstring). ``prompt_tokens`` is the last reported ``usage.prompt_tokens``;
    ``truncated`` flags it as implausibly small for this essay's length.
    """

    essay_id: str
    scores: dict[str, int] | None
    raw: str
    attempts: int
    system_fingerprint: str | None = None
    fingerprint_changed: bool = False
    prompt_tokens: int | None = None
    truncated: bool = False


def build_messages(essay_text: str, trait: str) -> list[dict[str, str]]:
    """Assemble the chat messages to score ONE trait of one essay.

    The prompt template is committed here so the (model, prompt, seed) triple fully
    determines a score. ``/no_think`` in the system message disables chain-of-thought on
    models that honor it; non-thinking models ignore it. One call per trait, never all 7 in
    one prompt: batching induces halo/anchoring in some judges (see module docstring).
    """
    system = (
        "/no_think\n"
        "You are a trained rater scoring essays by English language learners on the "
        "ELLIPSE analytic rubric. Score strictly and consistently. Respond with ONLY a "
        "JSON object, no prose, no explanation, no chain-of-thought."
    )
    user = (
        f"Score this essay on ONE dimension only: {trait}. The score is an integer from "
        "1 to 5.\n\n"
        f"Scale anchors: {SCALE_ANCHORS}\n\n"
        f"Keyword rubric for {trait}: {TRAIT_KEYWORDS[trait]}\n\n"
        f'Return exactly this JSON shape with an integer value 1-5: {{"{trait}": <int>}}\n\n'
        f'Essay:\n"""\n{essay_text}\n"""'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_score(raw: str, trait: str) -> int | None:
    """Parse a completion into a single 1-5 int for ``trait``, or ``None`` on any failure.

    Strips any ``<think>`` block, extracts the first ``{...}`` span, JSON-decodes it, and
    requires ``trait`` present with a value coercible to an integer clamped to 1-5. A
    missing key, a non-numeric value, or malformed JSON all return ``None`` so the caller
    can retry or log.
    """
    cleaned = _THINK_BLOCK.sub("", raw).strip()
    match = _JSON_OBJECT.search(cleaned)
    if match is None:
        return None
    try:
        obj: object = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    fields = cast("dict[str, object]", obj)
    if trait not in fields:
        return None
    value = fields[trait]
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        digits = re.search(r"-?\d+", value)
        if digits is None:
            return None
        number = int(digits.group(0))
    else:
        return None
    return max(1, min(5, number))


@dataclass(frozen=True)
class _ChatResponse:
    """One parsed chat-completion response: content plus its provenance fields."""

    content: str
    system_fingerprint: str | None
    prompt_tokens: int | None


def _build_payload(config: JudgeConfig, messages: list[dict[str, str]]) -> dict[str, object]:
    """Assemble the request body, auto-pinning the OpenRouter provider route.

    When ``base_url`` is OpenRouter and ``config.provider`` was not set explicitly, pins
    ``provider`` to :data:`DEFAULT_OPENROUTER_PROVIDER` (see module docstring).
    """
    payload: dict[str, object] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "seed": config.seed,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    provider = config.provider
    if provider is None and OPENROUTER_HOST in config.base_url:
        provider = DEFAULT_OPENROUTER_PROVIDER
    if provider:
        payload["provider"] = provider
    return payload


def _looks_truncated(prompt_tokens: int | None, essay_text: str) -> bool:
    """Flag ``prompt_tokens`` implausibly small for ``essay_text`` -- likely context truncation."""
    if prompt_tokens is None:
        return False
    return prompt_tokens < len(essay_text) // _CHARS_PER_TOKEN_ESTIMATE


def _chat_once(config: JudgeConfig, messages: list[dict[str, str]]) -> _ChatResponse:
    """POST one chat-completion request and return the assistant message plus provenance.

    Uses stdlib ``urllib``; passes ``temperature`` and ``seed`` for determinism and asks
    for a JSON object via ``response_format`` (endpoints that ignore it still get the JSON
    instruction in the prompt). Raises ``urllib.error.URLError`` / ``OSError`` on transport
    failure and ``ValueError`` on an unparseable transport-level response body.
    """
    payload = _build_payload(config, messages)
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
        body: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("unexpected response body: not a JSON object")
    body_fields = cast("dict[str, object]", body)
    choices = body_fields.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"no choices in response: {body!r}")
    first = cast("object", choices[0])
    if not isinstance(first, dict):
        raise ValueError("malformed choice in response")
    choice_fields = cast("dict[str, object]", first)
    message = choice_fields.get("message")
    if not isinstance(message, dict):
        raise ValueError("malformed message in response")
    message_fields = cast("dict[str, object]", message)
    content = message_fields.get("content")
    if not isinstance(content, str):
        raise ValueError("non-string content in response")
    fingerprint = body_fields.get("system_fingerprint")
    usage = body_fields.get("usage")
    prompt_tokens = None
    if isinstance(usage, dict):
        usage_fields = cast("dict[str, object]", usage)
        raw_prompt_tokens = usage_fields.get("prompt_tokens")
        if isinstance(raw_prompt_tokens, int) and not isinstance(raw_prompt_tokens, bool):
            prompt_tokens = raw_prompt_tokens
    return _ChatResponse(
        content=content,
        system_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        prompt_tokens=prompt_tokens,
    )


def score_essay(config: JudgeConfig, essay_id: str, essay_text: str) -> JudgeResult:
    """Score one essay with one call per trait, retrying each up to ``config.max_retries``.

    Returns a :class:`JudgeResult`. All 7 traits must parse for a success; if any trait
    exhausts its retries, scoring stops there and ``scores=None`` is returned with that
    trait's raw text, so a partial (some-traits-missing) result is never returned. A
    transport error on an attempt is treated like a parse failure (retried, then surfaced
    the same way), so one flaky call never silently corrupts the essay's scores.

    Also tracks provenance across the 7 calls: ``fingerprint_changed`` if
    ``system_fingerprint`` was not stable, and ``truncated`` from the last
    ``prompt_tokens`` seen (see module docstring).
    """
    scores: dict[str, int] = {}
    raw = ""
    attempts = 0
    fingerprints_seen: list[str] = []
    last_prompt_tokens: int | None = None
    for trait in RUBRIC_TRAITS:
        messages = build_messages(essay_text, trait)
        trait_score: int | None = None
        for _ in range(config.max_retries + 1):
            attempts += 1
            try:
                response = _chat_once(config, messages)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                raw = f"<request-error: {exc}>"
                continue
            raw = response.content
            if response.system_fingerprint is not None and (
                not fingerprints_seen or fingerprints_seen[-1] != response.system_fingerprint
            ):
                fingerprints_seen.append(response.system_fingerprint)
            if response.prompt_tokens is not None:
                last_prompt_tokens = response.prompt_tokens
            trait_score = parse_score(raw, trait)
            if trait_score is not None:
                break
        if trait_score is None:
            return JudgeResult(
                essay_id=essay_id,
                scores=None,
                raw=raw,
                attempts=attempts,
                system_fingerprint=fingerprints_seen[-1] if fingerprints_seen else None,
                fingerprint_changed=len(set(fingerprints_seen)) > 1,
                prompt_tokens=last_prompt_tokens,
                truncated=_looks_truncated(last_prompt_tokens, essay_text),
            )
        scores[trait] = trait_score
    return JudgeResult(
        essay_id=essay_id,
        scores=scores,
        raw=raw,
        attempts=attempts,
        system_fingerprint=fingerprints_seen[-1] if fingerprints_seen else None,
        fingerprint_changed=len(set(fingerprints_seen)) > 1,
        prompt_tokens=last_prompt_tokens,
        truncated=_looks_truncated(last_prompt_tokens, essay_text),
    )
