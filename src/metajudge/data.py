# src/metajudge/data.py
"""Long-format ratings model shared by every pillar."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Never

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
        """Build Ratings directly from an already-validated long frame.

        This constructor does no validation (no duplicate-cell check, no stratum
        completeness check): it trusts ``long`` as-is. Prefer :meth:`from_long` or
        :meth:`from_eval_instruments`, which validate before construction.
        """
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
        """Build a validated Ratings from a long-format DataFrame.

        ``df`` carries one row per (item, rater) cell; ``item``, ``rater``, and ``score``
        name its columns. Pass ``stratum`` to carry a DIF stratum label through. Only the
        named columns are copied; the input frame is not mutated.

        Raises ``ValueError`` when a named column is missing, when duplicate (item, rater)
        cells are present (aggregate explicitly first; duplicates that disagree on the
        stratum label get their own message), when any stratum value is missing, or when
        one item maps to more than one stratum.
        """
        cols = [item, rater, score] + ([stratum] if stratum is not None else [])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        # score is an ordinal measurement; reject a non-numeric column at the boundary rather
        # than let it surface as an opaque failure deep inside a statistic. Coercibility, not
        # stored dtype, is the test, so numeric values held in an object column still pass.
        try:
            pd.to_numeric(df[score])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"score column {score!r} must be numeric/ordinal; got non-numeric values "
                f"(dtype {df[score].dtype}). Encode ordinal labels as numbers before "
                "building Ratings."
            ) from exc
        duplicate_cells = df.duplicated(subset=[item, rater], keep=False)
        if bool(duplicate_cells.any()):
            bad = df.loc[duplicate_cells, [item, rater]].drop_duplicates().to_dict("records")
            if stratum is not None and bool(
                df.loc[duplicate_cells].groupby([item, rater])[stratum].nunique().gt(1).any()
            ):
                raise ValueError(
                    f"duplicate item-rater cells with conflicting stratum labels: {bad}; "
                    "fix the stratum column first, then aggregate if needed"
                )
            raise ValueError(f"duplicate item-rater cells found; aggregate explicitly first: {bad}")
        if stratum is not None:
            if bool(df[stratum].isna().any()):
                bad_items = df.loc[df[stratum].isna(), item].drop_duplicates().tolist()
                raise ValueError(f"missing stratum values for items: {bad_items}")
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

        Each value in ``frames`` is one judge's (or run's) ``frame_from_evals`` output
        (rows = evaluated samples, columns = rubric criteria), keyed by the judge
        id. The measurement frame is rater = judge, item = sample, score = the
        selected rubric ``criterion``. Rubric criteria are a separate facet, audited
        one at a time, never treated as raters (a criteria-as-raters frame would
        measure internal consistency, not inter-rater reliability). See the interop
        ADR for the cited rationale.

        Pass ``stratum`` (a sample-id -> stratum-label mapping) to carry a DIF stratum
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

    @property
    def n_items(self) -> Never:
        raise AttributeError(
            "Ratings.n_items was removed in 0.1.0; use len(ratings.items) instead."
        )

    @property
    def n_raters(self) -> Never:
        raise AttributeError(
            "Ratings.n_raters was removed in 0.1.0; use len(ratings.raters) instead."
        )

    def wide(self) -> pd.DataFrame:
        """Items x raters score matrix, reindexed to ``self.items`` x ``self.raters``.

        Unrated cells are ``NaN``. Uses ``pivot`` (not ``pivot_table``) so duplicate
        item-rater cells raise rather than being silently averaged; ``from_long`` already
        rejects duplicates, but Ratings can be built directly, so this keeps the
        one-cell-per-pair invariant true on every path.
        """
        wide = self._long.pivot(
            index=self._item_col,
            columns=self._rater_col,
            values=self._score_col,
        )
        return wide.reindex(index=self.items, columns=self.raters)

    def coder_unit_matrix(self) -> np.ndarray:  # type: ignore[type-arg]
        return self.wide().to_numpy(dtype=float).T

    def strata(self) -> dict[str, list[Hashable]]:
        """Map each stratum label to the sorted list of items it contains.

        Raises ``ValueError`` when the Ratings was built without a stratum column.
        """
        if self._stratum_col is None:
            raise ValueError("no stratum column was provided")
        out: dict[str, list[Hashable]] = {}
        pairs = self._long[[self._item_col, self._stratum_col]].drop_duplicates()
        for level, sub in pairs.groupby(self._stratum_col):
            out[str(level)] = sorted(sub[self._item_col].tolist())
        return out
