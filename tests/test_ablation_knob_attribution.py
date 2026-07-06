"""Unit tests for the knob-attribution ablation harness (no network, no live LLM calls).

Isolates how much of the GPT-4o collapse fix (docs/decisions/2026-07-04-e07-ellipse-human-
rater-dif.md, "Residual collapse...") comes from `reasoning` vs `trait_scoped_anchors`
individually. Every test here is pure logic or CSV-resumability against synthetic data --
no essay corpus, no API key, no network required to run this file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import ablation_knob_attribution as abl
from _ellipse_judge import JudgeResult


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


def test_judge_config_for_reasoning_only_arm_sets_reasoning_true() -> None:
    config = abl.judge_config_for_arm("reasoning_only", base_url=None, model=None, api_key=None)
    assert config.reasoning is True
    assert config.trait_scoped_anchors is False


def test_judge_config_for_trait_scoped_anchors_only_arm_sets_it_true() -> None:
    config = abl.judge_config_for_arm(
        "trait_scoped_anchors_only", base_url=None, model=None, api_key=None
    )
    assert config.reasoning is False
    assert config.trait_scoped_anchors is True


def test_judge_config_for_unknown_arm_raises() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        abl.judge_config_for_arm("both", base_url=None, model=None, api_key=None)


def test_judge_config_for_arm_forwards_base_url_and_model() -> None:
    config = abl.judge_config_for_arm(
        "reasoning_only",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o",
        api_key="test-key",
    )
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.model == "openai/gpt-4o"
    assert config.api_key == "test-key"


def test_summarize_arm_counts_collapsed_scored_and_failed() -> None:
    results = [
        JudgeResult(essay_id="a", scores={"Overall": 2, "Cohesion": 2}, raw="", attempts=1),
        JudgeResult(essay_id="b", scores={"Overall": 3, "Cohesion": 4}, raw="", attempts=1),
        JudgeResult(essay_id="c", scores=None, raw="<request-error>", attempts=3),
    ]
    summary = abl.summarize_arm("reasoning_only", results)
    assert summary.arm == "reasoning_only"
    assert summary.n_essays == 3
    assert summary.n_scored == 2
    assert summary.n_failed == 1
    assert summary.n_collapsed == 1
    assert summary.collapse_rate == pytest.approx(0.5)
    assert summary.n_fingerprint_changed == 0
    assert summary.n_truncated == 0


def test_summarize_arm_collapse_rate_is_nan_when_nothing_scored() -> None:
    results = [JudgeResult(essay_id="a", scores=None, raw="", attempts=3)]
    summary = abl.summarize_arm("reasoning_only", results)
    assert math.isnan(summary.collapse_rate)


def test_summarize_arm_counts_fingerprint_changed_and_truncated_regardless_of_parse_success() -> (
    None
):
    results = [
        JudgeResult(
            essay_id="a",
            scores={"Overall": 2},
            raw="",
            attempts=1,
            fingerprint_changed=True,
        ),
        JudgeResult(
            essay_id="b",
            scores=None,
            raw="<request-error>",
            attempts=3,
            truncated=True,
        ),
        JudgeResult(essay_id="c", scores={"Overall": 3}, raw="", attempts=1),
    ]
    summary = abl.summarize_arm("reasoning_only", results)
    assert summary.n_fingerprint_changed == 1
    assert summary.n_truncated == 1
