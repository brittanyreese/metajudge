"""Isolate how much of the GPT-4o collapse fix each prompt-mitigation flag contributes.

Companion to ``examples/audit_ellipse_llm.py``. The paired ablation in
``docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md`` ("Residual collapse after the
per-trait fix") showed both ``reasoning`` and ``trait_scoped_anchors`` ON together drop
GPT-4o's cross-call score collapse from 60% (24/40, default prompt) to 8% (3/40, both
flags), on the same 40 essays (20 focal + 20 reference, seed 7), scored against
``openai/gpt-4o`` via OpenRouter with the provider pin. That ablation changed both knobs at
once, so it confirms the combined fix but not how much each flag contributes individually.
This script isolates that: it scores the same 40-essay sample under two arms, each with
exactly one flag on, and reports each arm's collapse rate against the known baselines
(60% default, 8% both-flags, 12-14% human-rater).

Two modes, same as ``audit_ellipse_llm.py``:

    # DRY RUN: prints the sample size and the number of API calls this would make. No
    # network, no data files, no API key required.
    uv run python examples/ablation_knob_attribution.py --dry-run

    # LIVE: scores the 40-essay sample under both arms against a real endpoint. Needs the
    # ELLIPSE corpus on disk (see examples/audit_ellipse.py's load_merged) and an
    # OpenAI-compatible endpoint (see examples/_ellipse_judge.py). Confirm cost with the
    # user before running -- this repo requires explicit sign-off before any paid API call.
    uv run python examples/ablation_knob_attribution.py \\
        --base-url https://openrouter.ai/api/v1 --model openai/gpt-4o
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from audit_ellipse import FOCAL, REFERENCE

HERE = Path(__file__).parent
DEFAULT_OUT_DIR = HERE / "data"

RUBRIC_TRAIT_COUNT = 7  # examples/_ellipse_judge.py: RUBRIC_TRAITS has 7 entries

ARMS = ("reasoning_only", "trait_scoped_anchors_only")

_ABLATION_COLUMNS = ["text_id_kaggle", "race_ethnicity", "Text"]

# Baselines from the combined ablation this script attributes (docs/decisions/2026-07-04-
# e07-ellipse-human-rater-dif.md, "Residual collapse after the per-trait fix").
BASELINE_DEFAULT_COLLAPSE_RATE = 24 / 40  # default prompt, neither flag
BASELINE_BOTH_FLAGS_COLLAPSE_RATE = 3 / 40  # both flags on
HUMAN_COLLAPSE_RATE_RANGE = (0.12, 0.14)  # rater_1 (14%), rater_2 (12%), same essays


def select_ablation_sample(
    merged: pd.DataFrame, *, n_per_group: int = 20, seed: int = 7
) -> pd.DataFrame:
    """Deterministically pick a balanced focal/reference sample with the essay text.

    Mirrors ``audit_ellipse_llm.select_pilot``'s sampling method (same columns, same
    per-stratum ``DataFrame.sample``), parameterized to the ablation's 20+20/seed-7 design
    instead of the 150+150/seed-0 pilot design, so the two scripts stay independently
    readable rather than sharing a cross-module private helper for a two-line difference.
    """
    parts: list[pd.DataFrame] = []
    for stratum in (FOCAL, REFERENCE):
        pool = merged.loc[merged["race_ethnicity"] == stratum, _ABLATION_COLUMNS].dropna(
            subset=["Text"]
        )
        parts.append(pool.sample(n=n_per_group, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def is_collapsed(scores: Mapping[str, int]) -> bool:
    """Whether every trait in ``scores`` was assigned the identical value.

    Matches the collapse definition in ``docs/decisions/2026-07-04-e07-ellipse-human-rater-
    dif.md``: an essay "collapses" when all 7 per-trait calls return the same integer.
    """
    return len(set(scores.values())) == 1


def estimate_call_count(*, n_per_group: int, n_arms: int = 2) -> int:
    """Total chat-completion calls this ablation would make (no network; dry-run math).

    ``n_per_group`` essays per stratum, 2 strata, ``n_arms`` prompt variants, one call per
    trait (``RUBRIC_TRAIT_COUNT``), never batched (examples/_ellipse_judge.py module
    docstring: batching induces halo/anchoring).
    """
    return n_per_group * 2 * n_arms * RUBRIC_TRAIT_COUNT
