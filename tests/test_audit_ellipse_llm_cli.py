# tests/test_audit_ellipse_llm_cli.py
"""The regenerate CLI threads the collapse-mitigation prompt flags into the judge config.

``--reasoning`` and ``--trait-scoped-anchors`` opt into the prompt variant that drops
GPT-4o's cross-call score collapse (``docs/decisions/2026-07-04-e07-ellipse-human-rater-dif.md``);
both default off so a plain ``--regenerate`` reproduces the committed pilot CSV. These pin
that the argparse flags reach ``JudgeConfig``, and that the default stays the pilot prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import audit_ellipse_llm as all_mod


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "base_url": None,
        "model": None,
        "api_key": None,
        "reasoning": False,
        "trait_scoped_anchors": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_regenerate_config_is_the_pilot_prompt() -> None:
    config = all_mod.judge_config_from_args(_args())
    assert config.reasoning is False
    assert config.trait_scoped_anchors is False


def test_flags_thread_the_collapse_mitigation_prompt_into_the_config() -> None:
    config = all_mod.judge_config_from_args(_args(reasoning=True, trait_scoped_anchors=True))
    assert config.reasoning is True
    assert config.trait_scoped_anchors is True
