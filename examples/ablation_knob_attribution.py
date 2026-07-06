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

import argparse
import sys
from collections.abc import Callable, Mapping
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


_ROW_COLUMNS = [
    "text_id_kaggle",
    "race_ethnicity",
    "scored",
    "collapsed",
    "fingerprint_changed",
    "truncated",
    "attempts",
]


def _already_scored(out_path: Path) -> set[str]:
    """Essay ids already present in an arm's output CSV (resumable runs)."""
    if not out_path.exists():
        return set()
    done = pd.read_csv(out_path, usecols=["text_id_kaggle"])
    return {str(x) for x in done["text_id_kaggle"].tolist()}


def _append_row(out_path: Path, row: dict[str, object]) -> None:
    """Append one scored essay's summary row, writing the header only on first write."""
    frame = pd.DataFrame([row], columns=_ROW_COLUMNS)
    write_header = not out_path.exists()
    frame.to_csv(out_path, mode="a", header=write_header, index=False)


def run_arm(
    arm: str,
    config: JudgeConfig,
    sample: pd.DataFrame,
    out_path: Path,
    *,
    score_essay_fn: Callable[[JudgeConfig, str, str], JudgeResult] | None = None,
) -> ArmSummary:
    """Score every essay in ``sample`` under one ablation arm, resumable via ``out_path``.

    Essays already present in ``out_path`` are skipped (kill-safe: a rerun continues an
    interrupted ablation without re-scoring or double-billing). The returned
    :class:`ArmSummary` is built from the FULL ``out_path`` contents after this run, so a
    resumed run's summary reflects all essays scored across every invocation, not just the
    ones scored this call. ``score_essay_fn`` defaults to the real
    ``_ellipse_judge.score_essay`` (imported lazily, same reason ``audit_ellipse_llm.py``
    imports it lazily: keep the dry-run path importable with no endpoint configured);
    tests inject a fake to avoid any network call.

    Prints a warning for each essay whose ``fingerprint_changed`` or ``truncated`` flag is
    set, and for any essay whose ``system_fingerprint`` is new partway through this arm's
    run -- the same two checks ``audit_ellipse_llm.regenerate`` runs for this exact
    judge+endpoint's documented fingerprint-drift and context-truncation risks
    (examples/_ellipse_judge.py module docstring). A flagged essay still counts toward the
    collapse rate; the flag is a caveat on trusting that number, not an exclusion.
    """
    if score_essay_fn is None:
        from _ellipse_judge import score_essay as score_essay_fn  # type: ignore[assignment]

    done = _already_scored(out_path)
    todo = sample[~sample["text_id_kaggle"].astype(str).isin(done)]
    run_fingerprints: set[str] = set()
    for position, (_, essay) in enumerate(todo.iterrows(), start=1):
        essay_id = str(essay["text_id_kaggle"])
        result = score_essay_fn(config, essay_id, str(essay["Text"]))
        if result.fingerprint_changed:
            print(
                f"[{arm} {position}/{len(todo)}] WARNING {essay_id}: system_fingerprint "
                "changed across this essay's 7 calls -- scores are non-comparable"
            )
        if result.system_fingerprint is not None:
            if run_fingerprints and result.system_fingerprint not in run_fingerprints:
                print(
                    f"[{arm} {position}/{len(todo)}] WARNING {essay_id}: "
                    f"system_fingerprint {result.system_fingerprint!r} is new for this "
                    f"run (seen so far: {sorted(run_fingerprints)}) -- treat the run as "
                    "non-comparable"
                )
            run_fingerprints.add(result.system_fingerprint)
        if result.truncated:
            print(
                f"[{arm} {position}/{len(todo)}] WARNING {essay_id}: prompt_tokens="
                f"{result.prompt_tokens} looks truncated for this essay's length"
            )
        scored = result.scores is not None
        _append_row(
            out_path,
            {
                "text_id_kaggle": essay_id,
                "race_ethnicity": essay["race_ethnicity"],
                "scored": scored,
                "collapsed": bool(scored and is_collapsed(result.scores)),  # type: ignore[arg-type]
                "fingerprint_changed": result.fingerprint_changed,
                "truncated": result.truncated,
                "attempts": result.attempts,
            },
        )

    # Every row always writes a literal True/False for "scored", "collapsed",
    # "fingerprint_changed", and "truncated" (never an empty cell), so all four columns
    # round-trip through pd.read_csv as clean bool dtype -- no NaN-vs-empty-string
    # ambiguity to reason about when re-summarizing a resumed run.
    written = pd.read_csv(out_path)
    n_scored = int(written["scored"].sum())
    n_collapsed = int(written["collapsed"].sum())
    n_fingerprint_changed = int(written["fingerprint_changed"].sum())
    n_truncated = int(written["truncated"].sum())
    return ArmSummary(
        arm=arm,
        n_essays=len(sample),
        n_scored=n_scored,
        n_fingerprint_changed=n_fingerprint_changed,
        n_truncated=n_truncated,
        n_failed=len(sample) - n_scored,
        n_collapsed=n_collapsed,
        collapse_rate=n_collapsed / n_scored if n_scored else float("nan"),
    )


def format_report(summaries: Mapping[str, ArmSummary]) -> str:
    """Human-readable comparison of both arms against the known collapse-rate baselines."""
    lines = [
        "Knob-attribution ablation (40-essay sample, seed 7):",
        f"  default prompt (neither flag), from the prior ablation: "
        f"{BASELINE_DEFAULT_COLLAPSE_RATE:.2f} (24/40)",
        f"  both flags on, from the prior ablation:                "
        f"{BASELINE_BOTH_FLAGS_COLLAPSE_RATE:.2f} (3/40)",
        f"  human-rater baseline, from the prior ablation:         "
        f"{HUMAN_COLLAPSE_RATE_RANGE[0]:.2f}-{HUMAN_COLLAPSE_RATE_RANGE[1]:.2f}",
        "",
    ]
    for arm, summary in summaries.items():
        lines.append(
            f"  {arm}: collapse_rate={summary.collapse_rate:.2f} "
            f"({summary.n_collapsed}/{summary.n_scored} scored, "
            f"{summary.n_failed} failed to parse, "
            f"{summary.n_fingerprint_changed} fingerprint-changed, "
            f"{summary.n_truncated} truncated)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sample size and call count; no network, no data files",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="model id served at --base-url")
    parser.add_argument("--api-key", default=None, help="bearer token")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.dry_run:
        n_calls = estimate_call_count(n_per_group=args.n_per_group)
        print(
            f"Dry run: {args.n_per_group * 2} essays ({args.n_per_group} focal + "
            f"{args.n_per_group} reference), {len(ARMS)} arms, {n_calls} total API calls. "
            "Confirm current per-token pricing for the target model/endpoint before "
            "running live -- this repo requires explicit sign-off before any paid API call."
        )
        return

    from audit_ellipse import load_merged

    sample = select_ablation_sample(load_merged(), n_per_group=args.n_per_group, seed=args.seed)
    summaries: dict[str, ArmSummary] = {}
    for arm in ARMS:
        config = judge_config_for_arm(
            arm, base_url=args.base_url, model=args.model, api_key=args.api_key
        )
        out_path = args.out_dir / f"ablation_{arm}.csv"
        summaries[arm] = run_arm(arm, config, sample, out_path)
    print(format_report(summaries))


if __name__ == "__main__":
    sys.exit(main())
