"""Worked example: the conditioner argument to `audit()` is not cosmetic.

`examples/audit_catches_bias.py` shows the report card catching a biased judge
panel, matched on an external, ground-truth-derived quality score. This example
asks the next question: what happens on the exact same biased panel if you
instead use the default leave-one-rater-out rest score?

The answer is not "a smaller effect." It is a missed one. The biased panel's
true DIF effect (Nagelkerke R2 delta ~0.18, comfortably class C) is corrupted
by the rest score itself: when the whole panel shares the bias, the rest score
is depressed for the focal family too, so it partially absorbs the very effect
it is supposed to help detect. On this panel that corruption is severe enough
to fully erase the verdict -- class C becomes class A, not class B.

The report card's own `conditioner_overlap_weak` diagnostic catches this
automatically: it fires only on the corrupted rest-score run, never on the
external-conditioner run or on either conditioner against the clean panel.
That flag is the actionable version of the panel-relative-DIF caveat stated
elsewhere in this repo -- here it is a number, not a warning in prose.

SIMULATED, not a model run. Reuses the exact clean/biased panels from
`examples/audit_catches_bias.py` (same seed, same `dif_uniform` values), so
"same data, different conditioner" is literal, not approximate.

Run it (no key, no network). Imports `sim.dgp` and `examples.audit_catches_bias`,
so run it as a module so the repo root is on the import path:

    uv run python -m examples.audit_conditioner_choice
"""

from __future__ import annotations

from examples.audit_catches_bias import BIASED_DIF_UNIFORM, simulate_panel
from sim.dgp import FOCAL, REFERENCE, SimSample

from metajudge import ReportCard, audit

PANEL_DIF_UNIFORM = {
    "clean": 0.0,
    "biased": BIASED_DIF_UNIFORM,
}


def _audit_both_conditioners(sample: SimSample) -> dict[str, ReportCard]:
    """Audit one panel twice: external (ground-truth) and rest-score (default)."""
    external = audit(
        sample.ratings,
        focal=FOCAL,
        reference=REFERENCE,
        conditioner=sample.conditioner,
    )
    rest_score = audit(sample.ratings, focal=FOCAL, reference=REFERENCE)
    return {"external": external, "rest-score": rest_score}


def build_cards() -> dict[str, dict[str, ReportCard]]:
    """Return {"clean": {"external":, "rest-score":}, "biased": {...}}."""
    return {
        name: _audit_both_conditioners(simulate_panel(dif_uniform))
        for name, dif_uniform in PANEL_DIF_UNIFORM.items()
    }


def render(cards: dict[str, dict[str, ReportCard]]) -> str:
    """Assemble the four cards and the decision narration into one report."""
    biased_external = cards["biased"]["external"].dif
    biased_rest = cards["biased"]["rest-score"].dif
    lines = [
        "# metajudge: why the conditioner argument matters",
        "_SIMULATED PANELS (no model calls). Same clean/biased panels as "
        "audit_catches_bias.py, each audited twice: once with the external "
        "(ground-truth-derived) conditioner, once with the default rest score._",
        "",
        "## Clean panel, external conditioner",
        cards["clean"]["external"].to_markdown(),
        "",
        "## Clean panel, rest-score conditioner",
        cards["clean"]["rest-score"].to_markdown(),
        "",
        "## Biased panel, external conditioner",
        cards["biased"]["external"].to_markdown(),
        "",
        "## Biased panel, rest-score conditioner",
        cards["biased"]["rest-score"].to_markdown(),
        "",
        "## What just happened",
        f"On the clean panel, both conditioners agree: class "
        f"{cards['clean']['external'].dif.dif_class} either way. On the biased "
        f"panel, they disagree. The external conditioner reports class "
        f"{biased_external.dif_class} (R2 delta {biased_external.nagelkerke_r2_delta:.3f}), "
        f"matching the true, planted effect. The rest-score conditioner, run on "
        f"the identical data, reports class {biased_rest.dif_class} (R2 delta "
        f"{biased_rest.nagelkerke_r2_delta:.3f}): not a weaker signal, a missed one. "
        f"`conditioner_overlap_weak` is `{biased_rest.conditioner_overlap_weak}` on "
        f"that run and `False` on every other run in this example -- the engine "
        f"flags the corrupted conditioner without anyone having to notice the gap "
        f"by hand.",
        "",
        "The decision this changes: when a judge panel might share a bias against "
        "one output family, the default rest-score conditioner cannot be trusted "
        "to surface it. Pass an external, independent quality signal via "
        "`conditioner=` whenever one is available, and treat a rest-score-only "
        "class A on a panel you don't fully trust as inconclusive, not clean.",
    ]
    return "\n".join(lines)


def main() -> None:
    cards = build_cards()
    print(render(cards))


if __name__ == "__main__":
    main()
