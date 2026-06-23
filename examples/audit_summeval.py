"""Worked example: audit the SummEval expert panel with metajudge.

Runnable end to end on a fresh clone, no data setup:

    uv run python examples/audit_summeval.py

What this audits: the SummEval (Fabbri et al. 2020) EXPERT coherence annotations,
3 human expert raters scoring 1600 summaries (100 articles x 16 systems), stratified
by system family (extractive vs abstractive). The raters here are humans, not an LLM
judge; metajudge's statistics are rater-agnostic, so the report card looks the same
whichever the raters are. The numbers below illustrate the report-card format on a
real corpus; they are not a published claim about SummEval.

The expected output is committed alongside this script as `sample_output.txt`.
"""

from __future__ import annotations

from metajudge import audit, cluster_bootstrap_dif, load_demo


def main() -> None:
    ratings = load_demo()

    # The one-screen report card: reliability (alpha, ICC) + DIF across strata.
    card = audit(ratings, focal="abstractive", reference="extractive")
    print(card.to_markdown())

    # Cluster-robust CI for the DIF effect size, for when an analytic p-value lands
    # near a decision threshold. n_boot is kept small here so the example runs fast;
    # raise it for a real audit (the default is 1000).
    print("\n## Cluster-robust DIF check")
    cb = cluster_bootstrap_dif(
        ratings, focal="abstractive", reference="extractive", n_boot=200, seed=0
    )
    print(
        f"- Nagelkerke R2 delta: {cb.base.nagelkerke_r2_delta:.3f} "
        f"[95% cluster CI {cb.r2_delta_ci_low:.3f}, {cb.r2_delta_ci_high:.3f}]"
    )
    print(f"- CI reliable: {cb.ci_reliable} (n_effective={cb.n_effective} of {cb.n_boot})")


if __name__ == "__main__":
    main()
