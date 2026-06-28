# tests/test_data.py
import numpy as np
import pandas as pd
import pytest

from metajudge.data import Ratings


def _long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item": ["i1", "i1", "i2", "i2"],
            "rater": ["r1", "r2", "r1", "r2"],
            "score": [1, 1, 0, 1],
            "group": ["a", "a", "b", "b"],
        }
    )


def test_from_long_builds_matrix() -> None:
    r = Ratings.from_long(_long(), item="item", rater="rater", score="score")
    assert len(r.items) == 2
    assert len(r.raters) == 2
    m = r.coder_unit_matrix()
    assert m.shape == (2, 2)
    # rows = raters sorted (r1, r2); cols = items sorted (i1, i2)
    np.testing.assert_array_equal(m, np.array([[1.0, 0.0], [1.0, 1.0]]))


def test_missing_cell_is_nan() -> None:
    df = _long().iloc[:3]  # drop (i2, r2)
    r = Ratings.from_long(df, item="item", rater="rater", score="score")
    m = r.coder_unit_matrix()
    assert np.isnan(m[1, 1])


def test_strata_requires_stratum_column() -> None:
    r = Ratings.from_long(_long(), item="item", rater="rater", score="score")
    with pytest.raises(ValueError, match="no stratum"):
        r.strata()


def test_strata_maps_levels_to_items() -> None:
    r = Ratings.from_long(_long(), item="item", rater="rater", score="score", stratum="group")
    assert r.strata() == {"a": ["i1"], "b": ["i2"]}


def test_inconsistent_stratum_per_item_raises() -> None:
    df = _long()
    df.loc[1, "group"] = "b"  # i1 now has two stratum labels
    with pytest.raises(ValueError, match="one stratum"):
        Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")
