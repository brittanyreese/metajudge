"""Vendored SummEval demo corpus loader."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

from metajudge.data import Ratings

# The model_id -> system_family taxonomy (Fabbri et al. 2020) lives in
# scripts/prep_demo.py, which bakes the stratum column into demo.csv. The loader
# reads that column directly, so no copy of the mapping is needed here.


def load_demo() -> Ratings:
    """Return a Ratings object for the SummEval expert coherence subset.

    1600 items (100 articles x 16 summarization systems), 3 expert raters,
    coherence scores 1-5, stratum = system family (extractive / abstractive).
    Source: Fabbri et al. (2020) SummEval, MIT license. See data/SOURCE.md.

    DIF note: scores are 1-5 ordinal and audited with ordinal logistic-regression
    DIF (see logistic_dif). With no explicit conditioner the analysis matches on a
    leave-one-rater-out rest score across the 3 expert raters; pass an external
    quality conditioner for the stronger matching described in the DIF ADR.
    """
    resource = files("metajudge") / "data" / "demo.csv"
    with resource.open("r", encoding="utf-8") as handle:
        df = pd.read_csv(handle)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="stratum")
