"""Unit tests for the knob-attribution ablation harness (no network, no live LLM calls).

Isolates how much of the GPT-4o collapse fix (docs/decisions/2026-07-04-e07-ellipse-human-
rater-dif.md, "Residual collapse...") comes from `reasoning` vs `trait_scoped_anchors`
individually. Every test here is pure logic or CSV-resumability against synthetic data --
no essay corpus, no API key, no network required to run this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import ablation_knob_attribution as abl


def _synthetic_merged(n_focal: int, n_reference: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n_focal):
        rows.append(
            {
                "text_id_kaggle": f"foc{i}",
                "race_ethnicity": "Asian/Pacific Islander",
                "Text": f"focal essay body {i}",
            }
        )
    for i in range(n_reference):
        rows.append(
            {
                "text_id_kaggle": f"ref{i}",
                "race_ethnicity": "Hispanic/Latino",
                "Text": f"reference essay body {i}",
            }
        )
    return pd.DataFrame(rows)


def test_select_ablation_sample_returns_balanced_groups() -> None:
    merged = _synthetic_merged(n_focal=6, n_reference=6)
    sample = abl.select_ablation_sample(merged, n_per_group=2, seed=7)
    assert len(sample) == 4
    assert (sample["race_ethnicity"] == "Asian/Pacific Islander").sum() == 2
    assert (sample["race_ethnicity"] == "Hispanic/Latino").sum() == 2
    assert list(sample.columns) == ["text_id_kaggle", "race_ethnicity", "Text"]


def test_select_ablation_sample_is_deterministic_for_a_fixed_seed() -> None:
    merged = _synthetic_merged(n_focal=6, n_reference=6)
    first = abl.select_ablation_sample(merged, n_per_group=2, seed=7)
    second = abl.select_ablation_sample(merged, n_per_group=2, seed=7)
    assert first["text_id_kaggle"].tolist() == second["text_id_kaggle"].tolist()


def test_select_ablation_sample_differs_for_a_different_seed() -> None:
    merged = _synthetic_merged(n_focal=6, n_reference=6)
    seed_7 = abl.select_ablation_sample(merged, n_per_group=2, seed=7)
    seed_8 = abl.select_ablation_sample(merged, n_per_group=2, seed=8)
    assert seed_7["text_id_kaggle"].tolist() != seed_8["text_id_kaggle"].tolist()


def test_is_collapsed_true_when_all_trait_scores_equal() -> None:
    assert abl.is_collapsed({"Overall": 2, "Cohesion": 2, "Syntax": 2}) is True


def test_is_collapsed_false_when_trait_scores_differ() -> None:
    assert abl.is_collapsed({"Overall": 2, "Cohesion": 3, "Syntax": 2}) is False


def test_is_collapsed_true_for_a_single_trait() -> None:
    assert abl.is_collapsed({"Overall": 4}) is True


def test_estimate_call_count_matches_two_arms_seven_traits() -> None:
    # 20 essays/group * 2 groups * 2 arms * 7 traits = 560 (docs/decisions/2026-07-04-e07-
    # ellipse-human-rater-dif.md: the prior paired ablation was the same order, ~$1-2).
    assert abl.estimate_call_count(n_per_group=20) == 560


def test_estimate_call_count_scales_with_n_per_group() -> None:
    assert abl.estimate_call_count(n_per_group=1) == 28
