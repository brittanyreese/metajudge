"""Audit an LLM judge scoring the ELLIPSE essays, mirroring the human-rater demo.

Companion to ``examples/audit_ellipse.py`` (the human two-rater panel). Here a single LLM
judge scores essays on the ELLIPSE analytic rubric, and metajudge asks the same DIF
question of that judge: does it score the focal race/ethnicity group differently from the
reference group, conditional on independent quality? This replicates the Yamashita (2025)
AES-fairness setup with a self-contained, reproducible open-model judge.

Two modes:

    # AUDIT (default): reproduces from the committed pilot CSV. No GPU, no model, no
    # network -- anyone can rerun the DIF audit.
    uv run python examples/audit_ellipse_llm.py

    # REGENERATE: re-score essays with an LLM judge and rewrite the CSV. Needs an
    # OpenAI-compatible endpoint (see examples/_ellipse_judge.py). The pilot default is a
    # local Ollama server + qwen2.5:7b. Resumable: rerun to continue an interrupted run.
    uv run python examples/audit_ellipse_llm.py --regenerate

The judge, prompt, seed, and decoding are pinned in ``examples/_ellipse_judge.py``; the
essay-selection seed and pilot size are pinned below. Together they make the committed CSV
reproducible from source.

PILOT vs REPORTED run:

- The committed CSV (``examples/data/ellipse_llm_pilot_qwen2.5-7b.csv``) is a PILOT: a
  balanced 300-essay sample (150 focal + 150 reference) scored by qwen2.5:7b on a local
  Ollama endpoint. It is a fast, fully local proof of the pipeline, not the headline model.
  (The pilot was specified as qwen2.5:32b; on the pilot hardware the 32b ran at 25-46
  s/essay under memory pressure, infeasible in-session, so its small Qwen sibling
  qwen2.5:7b was used instead -- a one-flag change, same model family. Rerun with
  ``--model qwen2.5:32b`` when the hardware allows.)
- The REPORTED judge is ``mlx-community/Meta-Llama-3.1-70B-Instruct-4bit`` served via
  ``mlx_lm.server`` (keyword-prompt AES literature finds Llama-3.1-70B the strongest open
  ELLIPSE config). Switching to it is a base_url + model change only, e.g.:

      mlx_lm.server --model mlx-community/Meta-Llama-3.1-70B-Instruct-4bit --port 8080
      uv run python examples/audit_ellipse_llm.py --regenerate \\
          --base-url http://127.0.0.1:8080/v1 \\
          --model mlx-community/Meta-Llama-3.1-70B-Instruct-4bit \\
          --out examples/data/ellipse_llm_llama31-70b.csv

  GPT-4o via OpenRouter is available as an optional hosted spot-check with the same flags
  (``--base-url https://openrouter.ai/api/v1 --model openai/gpt-4o`` plus an API key in
  ``ELLIPSE_JUDGE_API_KEY``).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# Sibling example modules (this file's directory is on sys.path when run as a script).
from audit_ellipse import ALL_TRAITS, ANALYTIC_TRAITS, FOCAL, REFERENCE, load_merged

from metajudge.data import Ratings
from metajudge.dif import DifResult, cluster_bootstrap_dif, logistic_dif

if TYPE_CHECKING:
    # Judge module is imported lazily at call time (keeps the default audit path
    # endpoint-free); this import exists only for the return-type annotation.
    from _ellipse_judge import JudgeConfig

HERE = Path(__file__).parent
DEFAULT_CSV = HERE / "data" / "ellipse_llm_pilot_qwen2.5-7b.csv"

# Pre-registered pilot design (fixed before any LLM scoring), mirroring audit_ellipse.py:
# focal vs reference stratum pair is Yamashita's flagged pair (imported from audit_ellipse).
PILOT_PER_GROUP = 150  # balanced: 150 focal + 150 reference
PILOT_SEED = 0
# Traits reported for the human-vs-LLM contrast: Vocabulary (human demo: pre-registered
# null) and Syntax (human demo: exploratory class-C, not robustly confirmed), plus the
# holistic Overall score.
PILOT_TRAITS = ("Vocabulary", "Syntax", "Overall")


def select_pilot(merged: pd.DataFrame) -> pd.DataFrame:
    """Deterministically pick a balanced focal/reference pilot with the essay text.

    Samples ``PILOT_PER_GROUP`` essays from each of the focal and reference strata with a
    fixed seed, keeping ``text_id_kaggle``, ``race_ethnicity``, and the essay ``Text``.
    """
    columns = ["text_id_kaggle", "race_ethnicity", "Text"]
    parts: list[pd.DataFrame] = []
    for stratum in (FOCAL, REFERENCE):
        pool = merged.loc[merged["race_ethnicity"] == stratum, columns].dropna(subset=["Text"])
        parts.append(pool.sample(n=PILOT_PER_GROUP, random_state=PILOT_SEED))
    return pd.concat(parts, ignore_index=True)


_CSV_COLUMNS = [
    "text_id_kaggle",
    "race_ethnicity",
    *ALL_TRAITS,
    "attempts",
    "system_fingerprint",
    "prompt_tokens",
    "truncated",
]


def _already_scored(out_path: Path) -> set[str]:
    """Essay ids already present in an existing scores CSV (for resumable runs)."""
    if not out_path.exists():
        return set()
    done = pd.read_csv(out_path, usecols=["text_id_kaggle"])
    return {str(x) for x in done["text_id_kaggle"].tolist()}


def _append_row(out_path: Path, row: dict[str, object]) -> None:
    """Append one scored essay to the CSV, writing the header only on first write."""
    frame = pd.DataFrame([row], columns=_CSV_COLUMNS)
    write_header = not out_path.exists()
    frame.to_csv(out_path, mode="a", header=write_header, index=False)


def judge_config_from_args(args: argparse.Namespace) -> JudgeConfig:
    """Build a ``JudgeConfig`` from parsed CLI args, threading the prompt-variant flags.

    ``--reasoning`` and ``--trait-scoped-anchors`` opt into the collapse-mitigation prompt
    (``docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md``); both default off, so a
    plain ``--regenerate`` reproduces the committed pilot CSV byte-for-byte. Imports
    ``JudgeConfig`` lazily for the same reason ``regenerate`` does: the default audit path
    never needs the judge module.
    """
    from _ellipse_judge import JudgeConfig

    return JudgeConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        reasoning=args.reasoning,
        trait_scoped_anchors=args.trait_scoped_anchors,
    )


def regenerate(args: argparse.Namespace) -> None:
    """Score the pilot essays with the LLM judge, appending to the scores CSV.

    Resumable and kill-safe: each scored essay is appended immediately, and a rerun skips
    essays already in the CSV, so the run survives interruption and can be chunked. Logs
    every essay that fails to parse after retries and reports the running parse-failure rate.
    """
    # Imported here so the default AUDIT path never imports the judge (keeps the audit
    # runnable with no endpoint) and an import error can't break reproduction.
    from _ellipse_judge import score_essay

    config = judge_config_from_args(args)
    print(
        f"Judge: model={config.model} base_url={config.base_url} temp={config.temperature} "
        f"seed={config.seed} reasoning={config.reasoning} "
        f"trait_scoped_anchors={config.trait_scoped_anchors}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pilot = select_pilot(load_merged())
    done = _already_scored(args.out)
    todo = pilot[~pilot["text_id_kaggle"].astype(str).isin(done)]
    print(
        f"Pilot: {len(pilot)} essays ({PILOT_PER_GROUP} focal + {PILOT_PER_GROUP} reference); "
        f"{len(done)} already scored, {len(todo)} to score."
    )

    scored = 0
    failures: list[str] = []
    run_fingerprints: set[str] = set()
    for position, (_, essay) in enumerate(todo.iterrows(), start=1):
        essay_id = str(essay["text_id_kaggle"])
        result = score_essay(config, essay_id, str(essay["Text"]))
        if result.fingerprint_changed:
            print(
                f"[{position}/{len(todo)}] WARNING {essay_id}: system_fingerprint changed "
                "across this essay's 7 calls -- scores are non-comparable"
            )
        if result.system_fingerprint is not None:
            if run_fingerprints and result.system_fingerprint not in run_fingerprints:
                print(
                    f"[{position}/{len(todo)}] WARNING {essay_id}: system_fingerprint "
                    f"{result.system_fingerprint!r} is new for this run (seen so far: "
                    f"{sorted(run_fingerprints)}) -- treat the run as non-comparable"
                )
            run_fingerprints.add(result.system_fingerprint)
        if result.truncated:
            print(
                f"[{position}/{len(todo)}] WARNING {essay_id}: prompt_tokens="
                f"{result.prompt_tokens} looks truncated for this essay's length"
            )
        if result.scores is None:
            failures.append(essay_id)
            print(
                f"[{position}/{len(todo)}] PARSE FAIL {essay_id} "
                f"(attempts={result.attempts}): {result.raw[:120]!r}"
            )
            continue
        row: dict[str, object] = {
            "text_id_kaggle": essay_id,
            "race_ethnicity": str(essay["race_ethnicity"]),
            "attempts": result.attempts,
            "system_fingerprint": result.system_fingerprint,
            "prompt_tokens": result.prompt_tokens,
            "truncated": result.truncated,
        }
        row.update(result.scores)
        _append_row(args.out, row)
        scored += 1
        if position % 25 == 0:
            print(f"[{position}/{len(todo)}] ok (this-run scored={scored}, failed={len(failures)})")

    total = len(_already_scored(args.out))
    print(f"\nScores CSV: {args.out} now holds {total}/{len(pilot)} pilot essays.")
    print(f"This run: scored {scored}, parse failures {len(failures)}.")
    if failures:
        print(f"Failed essay ids this run: {failures}")


def build_llm_conditioner(scores: pd.DataFrame, trait: str) -> dict[Hashable, float]:
    """Leave-one-trait-out external conditioner from the LLM's OWN other-trait scores.

    For an analytic trait, the mean of the other five analytic traits (Overall excluded);
    for ``"Overall"``, the mean of all six analytic traits. This keys quality "as the judge
    sees it" by essay, independent of the studied trait -- the same design as the human demo.
    """
    others = [t for t in ANALYTIC_TRAITS if t != trait]
    values = scores[others].mean(axis=1)
    return dict(zip(scores["text_id_kaggle"].tolist(), values.tolist(), strict=True))


def build_llm_ratings(scores: pd.DataFrame, trait: str) -> Ratings:
    """Single-judge Ratings for one trait: item = essay, rater = the LLM judge."""
    long = pd.DataFrame(
        {
            "item": scores["text_id_kaggle"],
            "rater": "llm_judge",
            "score": scores[trait],
            "stratum": scores["race_ethnicity"],
        }
    )
    return Ratings.from_long(long, item="item", rater="rater", score="score", stratum="stratum")


def audit_trait(scores: pd.DataFrame, trait: str, *, n_boot: int, seed: int) -> str:
    """Render the DIF result for one trait from the LLM scores as a markdown block.

    Uses the external-conditioner :func:`logistic_dif` (the single-judge path: one rating
    per essay, matched on the leave-one-trait-out quality proxy) plus a cluster bootstrap
    for a clustering-robust CI on the Nagelkerke R-squared change.
    """
    ratings = build_llm_ratings(scores, trait)
    conditioner = build_llm_conditioner(scores, trait)
    dif = logistic_dif(ratings, focal=FOCAL, reference=REFERENCE, conditioner=conditioner)
    boot = cluster_bootstrap_dif(
        ratings,
        focal=FOCAL,
        reference=REFERENCE,
        conditioner=conditioner,
        n_boot=n_boot,
        seed=seed,
    )
    r2 = "n/a" if math.isnan(dif.nagelkerke_r2_delta) else f"{dif.nagelkerke_r2_delta:.4f}"
    return "\n".join(
        [
            f"## {trait}",
            f"- {FOCAL} vs {REFERENCE} (external conditioner, single judge, n_obs={dif.n_obs})",
            f"- Effect size (Nagelkerke R2 delta): {r2} (Jodoin-Gierl class {dif.dif_class})",
            f"- Clustering-robust R2 delta {boot.ci_level:.0%} {boot.ci_method} CI "
            f"[{boot.r2_delta_ci_low:.4f}, {boot.r2_delta_ci_high:.4f}] "
            f"({boot.n_effective}/{boot.n_boot} resamples, reliable={boot.ci_reliable})",
            f"- Uniform DIF: chi2(1)={dif.chi2_uniform:.2f}, p={dif.p_uniform:.4f} "
            "[analytic, unclustered]",
            f"- Nonuniform DIF: chi2(1)={dif.chi2_nonuniform:.2f}, p={dif.p_nonuniform:.4f} "
            "[analytic, unclustered]",
            f"- Conditioner-group correlation: {dif.conditioner_group_corr:.3f} "
            f"(overlap_weak={dif.conditioner_overlap_weak})",
        ]
    )


def audit_from_csv(csv_path: Path, *, n_boot: int = 400, seed: int = 0) -> None:
    """Run the DIF audit for the pilot traits from the committed LLM scores CSV.

    Reliability (Krippendorff alpha, ICC) is deliberately not reported: a single LLM judge
    has no inter-rater agreement to measure. A multi-judge LLM panel (as in
    examples/audit_llm_judge.py) would restore that pillar; the fairness question here is DIF.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LLM scores CSV not found: {csv_path}. Run with --regenerate (needs an "
            "OpenAI-compatible judge endpoint) to create it."
        )
    scores = pd.read_csv(csv_path)
    counts = scores["race_ethnicity"].value_counts()
    print("# metajudge report card: ELLIPSE LLM-judge DIF (single judge)\n")
    print(f"LLM judge scores: {len(scores)} essays, per-stratum N: {counts.to_dict()}")
    print(f"Focal ({FOCAL}) vs reference ({REFERENCE}). Reliability pillar N/A (one judge).\n")
    for trait in PILOT_TRAITS:
        print(audit_trait(scores, trait, n_boot=n_boot, seed=seed))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate", action="store_true", help="re-score essays via the LLM judge"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="model id served at --base-url")
    parser.add_argument("--api-key", default=None, help="bearer token (unused by local servers)")
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="require a one-sentence justification before each trait score "
        "(collapse mitigation; off reproduces the committed pilot)",
    )
    parser.add_argument(
        "--trait-scoped-anchors",
        action="store_true",
        help="use per-trait scale anchors instead of the shared holistic block "
        "(collapse mitigation; off reproduces the committed pilot)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_CSV, help="scores CSV path (regenerate)"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="scores CSV to audit")
    args = parser.parse_args()

    if args.regenerate:
        regenerate(args)
    else:
        audit_from_csv(args.csv)


# Re-exported so a pinned test can build the same DIF result off the committed CSV without
# reaching into private names.
__all__ = [
    "DifResult",
    "build_llm_conditioner",
    "build_llm_ratings",
    "cluster_bootstrap_dif",
    "logistic_dif",
]


if __name__ == "__main__":
    sys.exit(main())
