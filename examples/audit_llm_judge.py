"""Worked example: audit a real LLM judge panel with metajudge.

Three LLM judges (distinct models) each score the coherence (1-5) of 16 short
summaries, stratified by system family (extractive vs abstractive). metajudge
then reports inter-judge reliability (Krippendorff's alpha, ICC) and differential
item functioning across the two families: whether the judge panel is
systematically tougher or noisier on abstractive summaries than on extractive
ones, conditional on quality. That is the question you ask before you trust an
LLM judge's scores as measurements.

Two modes, two providers:

    # Real LLM judges. --provider selects which panel and which key/SDK:
    #   gemini: needs GOOGLE_AI_API_KEY + `google-genai`. Needs a billed
    #                 (Tier 1+) project; the free tier's per-project RPM/RPD
    #                 quotas are too tight for a 48-call panel run.
    #   openrouter: needs OPENROUTER_API_KEY + `openai` (OpenRouter speaks
    #                 the OpenAI chat-completions schema). Free `:free` models,
    #                 but free-tier capacity is unpredictable call to call.
    # Both SDKs come with `pip install metajudge[examples]`.
    uv run python examples/audit_llm_judge.py --mode live --provider gemini
    uv run python examples/audit_llm_judge.py --mode live --provider openrouter

    # Reproducible simulation: no key, no network, runs on a fresh clone:
    uv run python examples/audit_llm_judge.py --mode offline

Live mode writes the real score matrix to `examples/llm_judge_scores.csv` and the
report card to `examples/sample_output_llm.txt`. Offline mode builds a SEEDED
SIMULATED panel so the report-card format is reproducible with zero setup; the
simulated numbers are not a model run and are labelled as such.

Items live in `examples/summaries_sample.jsonl`. The summaries are synthetic
(several abstractive ones carry deliberate coherence defects: vague pronouns,
a non-sequitur, word-salad, an unsupported claim) so the judges have something
real to disagree about.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from metajudge import audit, cluster_bootstrap_dif
from metajudge.data import Ratings

HERE = Path(__file__).parent
ITEMS_PATH = HERE / "summaries_sample.jsonl"
SCORES_PATH = HERE / "llm_judge_scores.csv"
OUTPUT_PATH = HERE / "sample_output_llm.txt"

# Three distinct judges per provider. Distinct models (not one model at
# temperature > 0) is what produces a genuine inter-rater question: do
# independent judges agree?
JUDGE_MODELS_BY_PROVIDER: dict[str, list[str]] = {
    # Stable (non-preview) Gemini models; preview/experimental models get
    # tighter rate limits regardless of billing tier.
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash"],
    # OpenRouter `:free` models from three distinct providers, chosen for
    # general instruction-following rather than code-specialized variants. If
    # one is consistently 429ing, swap it for another `:free` model; free
    # capacity varies model to model and hour to hour.
    "openrouter": [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-31b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ],
}

# Seconds to pause between calls, per provider. Gemini on a billed project has
# a high enough per-model RPM that this is mostly just being a polite client.
# OpenRouter's `:free` models share one flat 20 requests/minute account-wide
# cap regardless of billing tier; 5s (~12/min) leaves headroom so one retried
# call doesn't fill the rolling window and cascade 429s into the next model.
REQUEST_DELAY_S_BY_PROVIDER: dict[str, float] = {"gemini": 1.0, "openrouter": 5.0}

RUBRIC = """\
You are scoring the COHERENCE of a summary against its source text.
Coherence = does the summary read as a clear, well-organised, internally
consistent statement that follows from the source? Ignore brevity; judge only
coherence and consistency.

Score on this 1-5 scale:
5 = fully coherent and consistent with the source
4 = mostly coherent, a minor awkwardness
3 = understandable but noticeably disjointed or with a small inconsistency
2 = hard to follow, vague references, or a clear inconsistency/contradiction
1 = incoherent (word-salad, non-sequitur, or contradicts the source)

SOURCE:
{source}

SUMMARY:
{summary}

Respond with ONLY a JSON object: {{"coherence": <integer 1-5>}}"""


def load_items() -> list[dict[str, str]]:
    with ITEMS_PATH.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_score(text: str) -> int:
    """Pull an integer 1-5 out of a judge response; raise if none is present."""
    try:
        return int(json.loads(text)["coherence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        m = re.search(r'"coherence"\s*:\s*([1-5])', text) or re.search(r"\b([1-5])\b", text)
        if m is None:
            raise ValueError(f"no 1-5 score found in judge response: {text!r}") from None
        return int(m.group(1))


def _call_with_retry(
    call: Callable[[], int],
    *,
    error_types: tuple[type[Exception], ...],
    get_status: Callable[[Exception], int | None],
    label: str,
    max_attempts: int = 6,
) -> int:
    """Retry a single judge call with exponential backoff on 429/5xx.

    Transient 429/5xx responses (rate limits, "high demand" 503s) are retried
    so a capacity spike on one model does not discard the whole run. A model
    that stays down after all retries fails loudly with its name, so you know
    which entry in JUDGE_MODELS_BY_PROVIDER to swap.
    """
    import random
    import time

    retryable = {429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except error_types as exc:
            code = get_status(exc)
            if code not in retryable:
                raise  # 404 bad model, auth, etc.: fail fast, don't spin
            if attempt == max_attempts:
                raise SystemExit(
                    f"{label} kept returning {code} after {max_attempts} retries; it is "
                    f"capacity-constrained right now. Swap it for another model, or retry later."
                ) from exc
            wait = min(2**attempt, 30) + random.uniform(0, 1)
            print(f"  {label}: {code}, retry {attempt}/{max_attempts} in {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


_SCORE_COLUMNS = ["item", "rater", "score", "system_family"]


def _run_panel(
    items: list[dict[str, str]],
    judge_models: list[str],
    delay: float,
    judge: Callable[[str, str], int],
) -> pd.DataFrame:
    """Score each item with each judge, checkpointing to SCORES_PATH after each judge.

    A judge that crashes (deprecated model, exhausted retries) no longer costs
    the completed judges before it: their rows are already on disk, and a
    rerun skips any rater already present in SCORES_PATH instead of re-paying
    for calls that already succeeded.
    """
    import time

    existing = (
        pd.read_csv(SCORES_PATH) if SCORES_PATH.exists() else pd.DataFrame(columns=_SCORE_COLUMNS)
    )
    done_raters = set(existing["rater"])
    frames = [existing] if not existing.empty else []

    for model in judge_models:
        if model in done_raters:
            print(f"judging with {model} ... already scored, skipping (see {SCORES_PATH.name})")
            continue
        print(f"judging with {model} ...")
        rows: list[dict[str, object]] = []
        for item in items:
            prompt = RUBRIC.format(source=item["source"], summary=item["summary"])
            rows.append(
                {
                    "item": item["id"],
                    "rater": model,
                    "score": judge(model, prompt),
                    "system_family": item["system_family"],
                }
            )
            time.sleep(delay)
        frames.append(pd.DataFrame(rows))
        pd.concat(frames, ignore_index=True).to_csv(SCORES_PATH, index=False)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_SCORE_COLUMNS)


def _score_live_gemini(
    items: list[dict[str, str]], judge_models: list[str], delay: float
) -> pd.DataFrame:
    import os

    try:
        from google import genai
        from google.genai import errors as genai_errors
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "live mode needs the google-genai SDK: "
            "`uv add --optional examples google-genai` (or pip install google-genai)"
        ) from exc

    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise SystemExit("live mode needs GOOGLE_AI_API_KEY in the environment")
    client = genai.Client(api_key=api_key)

    def judge(model: str, prompt: str) -> int:
        def call() -> int:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
            return _parse_score(resp.text or "")

        return _call_with_retry(
            call,
            error_types=(genai_errors.APIError,),
            get_status=lambda exc: getattr(exc, "code", None),
            label=model,
        )

    return _run_panel(items, judge_models, delay, judge)


def _score_live_openrouter(
    items: list[dict[str, str]], judge_models: list[str], delay: float
) -> pd.DataFrame:
    import os

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "live mode needs the openai SDK: "
            "`uv add --optional examples openai` (or pip install openai)"
        ) from exc

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("live mode needs OPENROUTER_API_KEY in the environment")
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def judge(model: str, prompt: str) -> int:
        def call() -> int:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return _parse_score(resp.choices[0].message.content or "")

        return _call_with_retry(
            call,
            error_types=(openai.APIStatusError,),
            get_status=lambda exc: exc.status_code,  # type: ignore[attr-defined]
            label=model,
        )

    return _run_panel(items, judge_models, delay, judge)


def score_live(items: list[dict[str, str]], provider: str) -> pd.DataFrame:
    judge_models = JUDGE_MODELS_BY_PROVIDER[provider]
    delay = REQUEST_DELAY_S_BY_PROVIDER[provider]
    if provider == "gemini":
        return _score_live_gemini(items, judge_models, delay)
    return _score_live_openrouter(items, judge_models, delay)


def score_offline(items: list[dict[str, str]], provider: str) -> pd.DataFrame:
    """Seeded SIMULATED judge panel: NOT a model run, for reproducible structure.

    Latent coherence is derived deterministically from the item id and family
    (abstractive items are lower and more variable, matching the deliberate
    defects in the data). Each simulated judge adds a fixed leniency bias plus
    seeded noise, then scores are clipped to the 1-5 ordinal scale.
    """
    import numpy as np

    judge_models = JUDGE_MODELS_BY_PROVIDER[provider]
    rng = np.random.default_rng(0)
    family_mean = {"extractive": 4.3, "abstractive": 3.1}
    family_spread = {"extractive": 0.45, "abstractive": 1.15}
    judge_bias = {m: b for m, b in zip(judge_models, (0.25, -0.30, 0.05), strict=True)}

    latent: dict[str, float] = {}
    for item in items:
        fam = item["system_family"]
        latent[item["id"]] = family_mean[fam] + rng.normal(0.0, family_spread[fam])

    rows: list[dict[str, object]] = []
    for model in judge_models:
        for item in items:
            raw = latent[item["id"]] + judge_bias[model] + rng.normal(0.0, 0.45)
            rows.append(
                {
                    "item": item["id"],
                    "rater": model,
                    "score": max(1, min(5, round(raw))),
                    "system_family": item["system_family"],
                }
            )
    return pd.DataFrame(rows)


def report(scores: pd.DataFrame, *, simulated: bool, provider: str) -> str:
    judge_models = JUDGE_MODELS_BY_PROVIDER[provider]
    ratings = Ratings.from_long(
        scores, item="item", rater="rater", score="score", stratum="system_family"
    )
    card = audit(ratings, focal="abstractive", reference="extractive")
    cb = cluster_bootstrap_dif(
        ratings, focal="abstractive", reference="extractive", n_boot=200, seed=0
    )
    banner = (
        "SIMULATED PANEL (no model calls): run `--mode live` for real LLM judges"
        if simulated
        else f"LIVE LLM JUDGE PANEL ({provider}): {', '.join(judge_models)}"
    )
    lines = [
        "# metajudge: LLM judge panel audit",
        f"_{banner}_",
        f"Judges: {len(ratings.raters)} | Items: {len(ratings.items)} "
        f"(extractive vs abstractive) | Score: coherence 1-5",
        "",
        card.to_markdown(),
        "",
        "## Cluster-robust DIF check (abstractive vs extractive)",
        f"- Nagelkerke R2 delta: {cb.base.nagelkerke_r2_delta:.3f} "
        f"[95% cluster CI {cb.r2_delta_ci_low:.3f}, {cb.r2_delta_ci_high:.3f}]",
        f"- CI reliable: {cb.ci_reliable} (n_effective={cb.n_effective} of {cb.n_boot})",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini")
    args = parser.parse_args()

    items = load_items()
    if args.mode == "live":
        scores = score_live(items, args.provider)
        scores.to_csv(SCORES_PATH, index=False)
        text = report(scores, simulated=False, provider=args.provider)
        OUTPUT_PATH.write_text(text + "\n")
        print(text)
        print(f"\n(wrote {SCORES_PATH.name} and {OUTPUT_PATH.name})")
    else:
        scores = score_offline(items, args.provider)
        print(report(scores, simulated=True, provider=args.provider))


if __name__ == "__main__":
    main()
