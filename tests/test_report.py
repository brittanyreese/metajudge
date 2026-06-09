# tests/test_report.py
import pandas as pd

from metajudge.data import Ratings
from metajudge.report import ReportCard, audit


def _ratings() -> Ratings:
    rows: list[dict[str, object]] = []
    groups = ["foc", "ref"]
    for i in range(40):
        grp = groups[i % 2]
        for r_idx in range(3):
            rows.append(
                {"item": f"i{i}", "rater": f"r{r_idx}", "score": (i + r_idx) % 5, "group": grp}
            )
    df = pd.DataFrame(rows)
    return Ratings.from_long(df, item="item", rater="rater", score="score", stratum="group")


def test_audit_returns_full_card() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    assert isinstance(card, ReportCard)
    assert card.dif.ets_class in {"A", "B", "C"}


def test_markdown_contains_all_pillars() -> None:
    card = audit(_ratings(), focal="foc", reference="ref", level="ordinal", seed=1)
    md = card.to_markdown()
    assert "Krippendorff" in md
    assert "ICC" in md
    assert "DIF" in md
    assert card.dif.ets_class in md
