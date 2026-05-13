# src/metajudge/data.py
"""Long-format ratings model shared by every pillar."""

from __future__ import annotations

from collections.abc import Hashable

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

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_raters(self) -> int:
        return len(self.raters)

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
