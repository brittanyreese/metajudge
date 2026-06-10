"""Vendored SummEval demo corpus loader."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

from metajudge.data import Ratings

# Fabbri et al. (2020) model taxonomy. Must match SYSTEM_FAMILY in scripts/prep_demo.py.
# Included here for documentation; the stratum column in demo.csv already encodes the mapping.
SYSTEM_FAMILY: dict[str, str] = {
    "M0": "extractive",  # Lead-3
    "M1": "extractive",  # LexRank
    "M2": "abstractive",  # Seq2Seq
    "M5": "abstractive",  # Fast-RL
    "M6": "abstractive",  # rnn-ext+abs+RL+rerank
    "M8": "extractive",  # SummaRuNNer
    "M9": "abstractive",  # ML encoder-decoder
    "M10": "extractive",  # REFRESH
    "M11": "extractive",  # HBSS
    "M12": "extractive",  # GSF
    "M13": "abstractive",  # ROUGESal+Ent RL
    "M14": "abstractive",  # Bottom-Up
    "M15": "extractive",  # Ban-ETM
    "M17": "abstractive",  # Two-stage
    "M20": "abstractive",  # T5
    "M22": "abstractive",  # BART
    "M23": "abstractive",  # PEGASUS
}


def load_demo() -> Ratings:
    """Return a Ratings object for the SummEval expert coherence subset.

    1600 items (100 articles x 16 summarization systems), 3 expert raters,
    coherence scores 1-5, stratum = system family (extractive / abstractive).
    Source: Fabbri et al. (2020) SummEval, MIT license. See data/SOURCE.md.

    DIF note: scores are 1-5 ordinal; the binary MH engine dichotomizes at the
    per-item median. This discards ordinal variation and detects uniform DIF only.
    Polytomous-native methods (GMH, OLR) belong to E04.
    """
    resource = files("metajudge") / "data" / "demo.csv"
    with resource.open("r", encoding="utf-8") as handle:
        df = pd.read_csv(handle)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="stratum")
