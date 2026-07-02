from metajudge import audit
from metajudge.data import Ratings
from metajudge.demo import load_demo


def test_load_demo_returns_ratings() -> None:
    r = load_demo()
    assert isinstance(r, Ratings)


def test_load_demo_shape() -> None:
    r = load_demo()
    assert len(r.items) == 1600
    assert len(r.raters) == 3
    scores = r.wide().to_numpy()
    assert int(scores.min()) >= 1
    assert int(scores.max()) <= 5


def test_load_demo_strata() -> None:
    r = load_demo()
    strata = r.strata()
    assert set(strata.keys()) == {"extractive", "abstractive"}


def test_load_demo_stratum_sizes() -> None:
    r = load_demo()
    strata = r.strata()
    for label, items in strata.items():
        assert len(items) >= 200, f"stratum {label!r} has only {len(items)} items (ETS floor: 200)"


def test_load_demo_audit_runs() -> None:
    r = load_demo()
    card = audit(r, focal="abstractive", reference="extractive")
    assert card.dif.dif_class in {"A", "B", "C"}
