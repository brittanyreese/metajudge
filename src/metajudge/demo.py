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

    DIF note: scores are 1-5 ordinal; the binary MH engine averages each item's
    raters, then dichotomizes those per-item means at their median. This discards
    ordinal variation and detects uniform DIF only. The result also depends on the
    n_match_bins default (see mantel_haenszel_dif). Polytomous-native methods
    (GMH, OLR) belong to E04.
    """
    resource = files("metajudge") / "data" / "demo.csv"
    with resource.open("r", encoding="utf-8") as handle:
        df = pd.read_csv(handle)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="stratum")
