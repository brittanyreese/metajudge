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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from audit_ellipse import FOCAL, REFERENCE

if TYPE_CHECKING:
    from _ellipse_judge import JudgeConfig, JudgeResult

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


def judge_config_for_arm(
    arm: str, *, base_url: str | None, model: str | None, api_key: str | None
) -> JudgeConfig:
    """Build the ``JudgeConfig`` for one ablation arm: exactly one mitigation flag on.

    Raises ``ValueError`` for any ``arm`` not in :data:`ARMS` -- a typo here must fail
    loudly, not silently score under the wrong prompt.
    """
    from _ellipse_judge import JudgeConfig

    if arm == "reasoning_only":
        reasoning, trait_scoped_anchors = True, False
    elif arm == "trait_scoped_anchors_only":
        reasoning, trait_scoped_anchors = False, True
    else:
        raise ValueError(f"unknown arm: {arm!r} (expected one of {ARMS})")
    return JudgeConfig.from_env(
        base_url=base_url,
        model=model,
        api_key=api_key,
        reasoning=reasoning,
        trait_scoped_anchors=trait_scoped_anchors,
    )


@dataclass(frozen=True)
class ArmSummary:
    """One ablation arm's collapse-rate outcome.

    ``collapse_rate`` is ``n_collapsed / n_scored``, NaN when nothing scored successfully
    (mirrors ``float("nan")`` rather than raising, since a fully-failed arm is a valid --
    if bad -- run outcome to report, not a programming error). ``n_fingerprint_changed``
    and ``n_truncated`` count essays flagged by ``JudgeResult.fingerprint_changed``/
    ``.truncated`` (examples/_ellipse_judge.py's provenance fields for this judge's
    documented fingerprint-drift and context-truncation risks), counted over ALL essays
    regardless of parse success -- a data-quality flag matters even on a failed essay.
    """

    arm: str
    n_essays: int
    n_scored: int
    n_failed: int
    n_collapsed: int
    collapse_rate: float
    n_fingerprint_changed: int
    n_truncated: int


def summarize_arm(arm: str, results: list[JudgeResult]) -> ArmSummary:
    """Aggregate one arm's per-essay ``JudgeResult`` list into an :class:`ArmSummary`."""
    scored = [r for r in results if r.scores is not None]
    failed = [r for r in results if r.scores is None]
    n_collapsed = sum(1 for r in scored if is_collapsed(r.scores))  # type: ignore[arg-type]
    n_scored = len(scored)
    collapse_rate = n_collapsed / n_scored if n_scored else float("nan")
    return ArmSummary(
        arm=arm,
        n_essays=len(results),
        n_scored=n_scored,
        n_failed=len(failed),
        n_collapsed=n_collapsed,
        collapse_rate=collapse_rate,
        n_fingerprint_changed=sum(1 for r in results if r.fingerprint_changed),
        n_truncated=sum(1 for r in results if r.truncated),
    )
