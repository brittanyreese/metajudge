"""Worked example: audit the ELLIPSE human-rater panel for race/ethnicity DIF.

ELLIPSE (Crossley et al. 2023; scrosseye/ELLIPSE-Corpus, CC BY-NC-SA 4.0) is 6,482
English-language-learner essays, each scored by two trained HUMAN raters on a holistic
score plus six analytic traits (cohesion, syntax, vocabulary, phraseology, grammar,
conventions), 0-5 in whole-point steps in the raw per-rater file this script reads, with
per-essay race/ethnicity, gender, SES, and grade. Yamashita (2025, Language Testing)
reports that GPT-4o scoring these same essays showed significant race/ethnicity bias
(over-scoring Asian/Pacific Islander writers, under-scoring Hispanic/Latino writers) on a
many-facet Rasch model; this script asks the DIF question of the HUMAN rater panel instead
(no LLM calls here -- that is a deferred follow-up, sketched in the findings doc below).

Design, pre-registered before this script was ever run against the data (see
docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md for the full record):

- item = essay (``text_id_kaggle``); rater = one of two anonymized rating slots
  (``rater_1``, ``rater_2``) the raw file provides per essay -- not stable individual
  raters across essays (the corpus draws from a pool of 27 human raters; the raw file
  does not expose which one). This is the same slot convention already used for the
  SummEval example's ``expert_0``/``expert_1``/``expert_2`` (see scripts/prep_demo.py).
- score = the PRIMARY trait is Vocabulary, chosen a priori: word choice and lexical
  register are the analytic dimension most exposed in the L2-writing-assessment
  literature to rater judgments correlated with a writer's linguistic/cultural
  background, independent of whether Yamashita's GPT-4o study broke its bias finding
  out by trait (it did not; the published result is at the holistic-proficiency level).
- stratum = race/ethnicity, collapsed to the pair Yamashita's study flagged: focal =
  Asian/Pacific Islander (n=791 after cleaning), reference = Hispanic/Latino (n=4619).
  Both comfortably clear the >=500 Jodoin-Gierl calibration floor.
- conditioner = external, independent of the studied trait: the leave-one-trait-out
  mean of the other five analytic traits, pooled across both raters. Holistic Overall
  is deliberately excluded from every analytic trait's conditioner (kept for its own
  row in the exploratory sweep below) so the conditioner never contains the studied
  response.

A second, EXPLORATORY, NOT pre-registered pass sweeps all 7 score dimensions (Overall
plus the 6 analytic traits) with Holm correction, because a single a-priori trait can
miss where the real signal is. Read its result as hypothesis-generating, not
confirmatory: docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md is explicit about
which numbers are pre-registered and which were found by looking.

Headline result (full record in the findings doc): the pre-registered Vocabulary analysis
is a clean Jodoin-Gierl class A null, cluster-robust CI included. The exploratory sweep
flags Syntax with a large (class C) analytic point estimate that survives Holm correction
and sits in the safe conditioner-overlap band -- but the item-cluster bootstrap CI on
Syntax reaches down to the negligible boundary in two independent runs, so that finding is
NOT robustly confirmed by this script's own stronger check. Read the exploratory result as
"worth a dedicated confirmatory run," not as a settled fairness claim.

Data setup (NOT vendored -- CC BY-NC-SA 4.0, ~15MB, non-commercial license):

    git clone --depth 1 https://github.com/scrosseye/ELLIPSE-Corpus.git data/raw/ellipse
    cd data/raw/ellipse
    unzip -P ellipse_raw_data ellipsis_raw_rater_scores_anon_all_essay.zip -d extracted
    unzip -P ellipse_test ELLIPSE_Final_github_test.zip -d extracted

Then, from the repo root:

    uv run python examples/audit_ellipse.py

Expected output (a real run, not simulated) is committed as
`examples/sample_output_ellipse.txt`; unlike audit_summeval.py this script needs the
manual data-fetch step above, so a fresh clone cannot reproduce it without that fetch.
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import pandas as pd

from metajudge import audit
from metajudge.data import Ratings
from metajudge.dif import DifResult, cluster_bootstrap_dif, holm_adjust, logistic_dif

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "raw" / "ellipse"
RAW_PATH = DATA_DIR / "extracted" / "ellipsis_raw_rater_scores_anon_all_essay.csv"
TRAIN_PATH = DATA_DIR / "ELLIPSE_Final_github_train.csv"
TEST_PATH = DATA_DIR / "extracted" / "ELLIPSE_Final_github_test.csv"

ALL_TRAITS = [
    "Overall",
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]
ANALYTIC_TRAITS = [t for t in ALL_TRAITS if t != "Overall"]

FOCAL = "Asian/Pacific Islander"
REFERENCE = "Hispanic/Latino"
PRIMARY_TRAIT = "Vocabulary"  # pre-registered; see module docstring

_MISSING_DATA_MSG = f"""\
ELLIPSE data not found under {DATA_DIR}.

Fetch it first (CC BY-NC-SA 4.0, ~15MB, not vendored in this repo):

    git clone --depth 1 https://github.com/scrosseye/ELLIPSE-Corpus.git {DATA_DIR}
    cd {DATA_DIR}
    unzip -P ellipse_raw_data ellipsis_raw_rater_scores_anon_all_essay.zip -d extracted
    unzip -P ellipse_test ELLIPSE_Final_github_test.zip -d extracted
"""


def load_merged() -> pd.DataFrame:
    """Load the raw per-rater scores, merge in demographics, and drop bad rows.

    Merges on ``text_id_kaggle``. Drops essays outside the final reliable corpus (no
    demographic match -- about 0.2% of the raw file) and essays carrying a literal ``0``
    in any of the 14 raw score columns, a data-quality artifact (rows with malformed or
    missing ``text_id_kaggle``, or true score anomalies) rather than a real point on the
    documented 1-5 scale; 6 of 6468 merged essays are dropped this way.
    """
    if not (RAW_PATH.exists() and TRAIN_PATH.exists() and TEST_PATH.exists()):
        raise FileNotFoundError(_MISSING_DATA_MSG)
    raw = pd.read_csv(RAW_PATH)
    meta = pd.concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)], ignore_index=True)
    score_cols = [f"{trait}_{rater}" for trait in ALL_TRAITS for rater in (1, 2)]
    clean = raw[~(raw[score_cols] == 0).any(axis=1)].copy()
    merged = clean.merge(
        meta[["text_id_kaggle", "race_ethnicity"]], on="text_id_kaggle", how="inner"
    )
    return merged


def build_ratings(merged: pd.DataFrame, trait: str) -> Ratings:
    """Long-format Ratings for one score dimension: item x {rater_1, rater_2}."""
    parts: list[pd.DataFrame] = []
    for rater_idx in (1, 2):
        part = merged[["text_id_kaggle", "race_ethnicity", f"{trait}_{rater_idx}"]].rename(
            columns={f"{trait}_{rater_idx}": "score"}
        )
        part["rater"] = f"rater_{rater_idx}"
        parts.append(part)
    long = pd.concat(parts, ignore_index=True).rename(
        columns={"text_id_kaggle": "item", "race_ethnicity": "stratum"}
    )
    return Ratings.from_long(long, item="item", rater="rater", score="score", stratum="stratum")


def build_conditioner(merged: pd.DataFrame, trait: str) -> dict[Hashable, float]:
    """Leave-one-trait-out external conditioner: mean of the other analytic traits.

    Pooled across both raters. For an analytic trait this means the other 5 analytic
    traits (holistic Overall is never in an analytic trait's conditioner); for
    ``"Overall"`` itself (exploratory sweep only) it is the mean of all 6 analytic
    traits.
    """
    others = [t for t in ANALYTIC_TRAITS if t != trait]
    cond_cols = [f"{other}_{rater}" for other in others for rater in (1, 2)]
    values = merged[cond_cols].mean(axis=1)
    return dict(zip(merged["text_id_kaggle"].tolist(), values.tolist(), strict=True))


def main() -> None:
    merged = load_merged()
    counts = merged["race_ethnicity"].value_counts()
    print(f"Loaded {len(merged)} essays (final reliable corpus, cleaned).")
    print(f"Per-stratum N: {counts.to_dict()}")
    print(f"Focal ({FOCAL}) n={counts[FOCAL]}, reference ({REFERENCE}) n={counts[REFERENCE]}\n")

    # --- Primary, pre-registered analysis: Vocabulary -----------------------------
    print(f"## Primary (pre-registered): {PRIMARY_TRAIT}\n")
    ratings = build_ratings(merged, PRIMARY_TRAIT)
    conditioner = build_conditioner(merged, PRIMARY_TRAIT)
    card = audit(
        ratings,
        focal=FOCAL,
        reference=REFERENCE,
        conditioner=conditioner,
        robust=True,
        n_boot=200,
        seed=0,
    )
    print(card.to_markdown())

    # --- Secondary, EXPLORATORY sweep across all 7 score dimensions ---------------
    # Not pre-registered: run after seeing the primary result. Holm-corrected across
    # the family so a real signal elsewhere in the rubric can be told apart from
    # noise, but read any hit here as hypothesis-generating, not confirmatory.
    print("\n## Exploratory (NOT pre-registered): Holm-corrected sweep, all 7 dimensions\n")
    sweep_results: list[DifResult] = []
    for trait in ALL_TRAITS:
        trait_ratings = build_ratings(merged, trait)
        trait_conditioner = build_conditioner(merged, trait)
        sweep_results.append(
            logistic_dif(
                trait_ratings, focal=FOCAL, reference=REFERENCE, conditioner=trait_conditioner
            )
        )
    holm_p = holm_adjust([r.p_total for r in sweep_results])
    for trait, r, p_holm in zip(ALL_TRAITS, sweep_results, holm_p, strict=True):
        print(
            f"- {trait:12s} R2 delta={r.nagelkerke_r2_delta:.4f} class={r.dif_class} "
            f"p_total_holm={p_holm:.3g} conditioner_group_corr={r.conditioner_group_corr:.3f} "
            f"overlap_weak={r.conditioner_overlap_weak}"
        )

    # --- Confirmatory-style follow-up on the sweep's largest hit -------------------
    top_trait = max(
        zip(ALL_TRAITS, sweep_results, strict=True), key=lambda pair: pair[1].nagelkerke_r2_delta
    )[0]
    print(f"\n## Follow-up robust check on the sweep's largest hit: {top_trait}\n")
    print(
        "(post-hoc: this trait was picked BECAUSE the exploratory sweep flagged it, so "
        "the cluster-robust CI below de-risks a fluke but does not make this a "
        "confirmatory analysis; see the findings doc.)\n"
    )
    top_ratings = build_ratings(merged, top_trait)
    top_conditioner = build_conditioner(merged, top_trait)
    top_cb = cluster_bootstrap_dif(
        top_ratings,
        focal=FOCAL,
        reference=REFERENCE,
        conditioner=top_conditioner,
        n_boot=200,
        seed=0,
    )
    print(
        f"- {top_trait}: Nagelkerke R2 delta={top_cb.base.nagelkerke_r2_delta:.3f} "
        f"[{top_cb.ci_level:.0%} {top_cb.ci_method} CI {top_cb.r2_delta_ci_low:.3f}, "
        f"{top_cb.r2_delta_ci_high:.3f}], class={top_cb.base.dif_class}, "
        f"n_effective={top_cb.n_effective} of {top_cb.n_boot}"
    )


if __name__ == "__main__":
    main()
