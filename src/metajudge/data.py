# src/metajudge/data.py
"""Long-format ratings model shared by every pillar."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np
import pandas as pd


class Ratings:
    """Multi-rater scores for a set of judged items, optionally stratified."""

    def __init__(
        self,
        long: pd.DataFrame,
        *,
        item_col: str,
        rater_col: str,
        score_col: str,
        stratum_col: str | None,
    ) -> None:
        self._long = long
        self._item_col = item_col
        self._rater_col = rater_col
        self._score_col = score_col
        self._stratum_col = stratum_col
        self.items: list[Hashable] = sorted(long[item_col].unique().tolist())
        self.raters: list[Hashable] = sorted(long[rater_col].unique().tolist())

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        item: str,
        rater: str,
        score: str,
        stratum: str | None = None,
    ) -> Ratings:
        cols = [item, rater, score] + ([stratum] if stratum else [])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        if stratum is not None:
            per_item = df.groupby(item)[stratum].nunique()
            if (per_item > 1).any():
                bad = per_item[per_item > 1].index.tolist()
                raise ValueError(f"each item must map to one stratum; offenders: {bad}")
        return cls(
            df[cols].copy(),
            item_col=item,
            rater_col=rater,
            score_col=score,
            stratum_col=stratum,
        )

    @classmethod
    def from_eval_instruments(
        cls,
        frames: Mapping[Hashable, pd.DataFrame],
        *,
        criterion: str,
        stratum: Mapping[Hashable, str] | pd.Series | None = None,
    ) -> Ratings:
        """Build Ratings from per-judge eval-instrument outputs.

        Each value in `frames` is one judge's (or run's) `frame_from_evals` output
        (rows = evaluated samples, columns = rubric criteria), keyed by the judge
        id. The measurement frame is rater = judge, item = sample, score = the
        selected rubric `criterion`. Rubric criteria are a separate facet, audited
        one at a time, never treated as raters (a criteria-as-raters frame would
        measure internal consistency, not inter-rater reliability). See the interop
        ADR for the cited rationale.

        Pass `stratum` (a sample-id -> stratum-label mapping) to carry a DIF stratum
        through. Frames are not imported from the eval tool; only their DataFrame
        output is consumed, so this adds no dependency.
        """
        if not frames:
            raise ValueError("from_eval_instruments needs at least one judge frame")
        parts: list[pd.DataFrame] = []
        for rater_id, frame in frames.items():
            # Handles both frame_from_evals shapes: flat (columns = criteria) and
            # detailed ((criterion, field) MultiIndex, field in class/score/notes).
            if isinstance(frame.columns, pd.MultiIndex):
                col = (criterion, "score")
                if col not in frame.columns:
                    raise ValueError(
                        f"criterion {criterion!r} (score field) not found in eval frame"
                    )
                scores = frame[col]
            else:
                if criterion not in frame.columns:
                    raise ValueError(f"criterion {criterion!r} not found in eval frame")
                scores = frame[criterion]
            parts.append(
                pd.DataFrame(
                    {
                        "item": list(scores.index),
                        "rater": rater_id,
                        "score": scores.to_numpy(),
                    }
                )
            )
        long = pd.concat(parts, ignore_index=True)
        if stratum is not None:
            mapping = stratum.to_dict() if isinstance(stratum, pd.Series) else dict(stratum)
            long["stratum"] = long["item"].map(mapping)
            return cls.from_long(long, item="item", rater="rater", score="score", stratum="stratum")
        return cls.from_long(long, item="item", rater="rater", score="score")

    def wide(self) -> pd.DataFrame:
        wide = self._long.pivot_table(
            index=self._item_col,
            columns=self._rater_col,
            values=self._score_col,
            aggfunc="mean",
        )
        return wide.reindex(index=self.items, columns=self.raters)

    def coder_unit_matrix(self) -> np.ndarray:  # type: ignore[type-arg]
        return self.wide().to_numpy(dtype=float).T

    def strata(self) -> dict[str, list[Hashable]]:
        if self._stratum_col is None:
            raise ValueError("no stratum column was provided")
        out: dict[str, list[Hashable]] = {}
        pairs = self._long[[self._item_col, self._stratum_col]].drop_duplicates()
        for level, sub in pairs.groupby(self._stratum_col):
            out[str(level)] = sorted(sub[self._item_col].tolist())
        return out
