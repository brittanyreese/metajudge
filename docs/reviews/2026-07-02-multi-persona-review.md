# Multi-persona project review — 2026-07-02

Status: Tier 1 findings were closed in `28df945` (PR #7); Tier 2 work is in
progress and tracked in [CHANGELOG.md](../../CHANGELOG.md) under Unreleased.
This document is the audit record that drove those fixes, kept as written.

Rigorous whole-project review of metajudge before resume-linking (target roles: AI/LLM
evaluation, measurement and outcomes research, psychometrics). Two phases:

- **Phase A** — five persona code reviewers (psychometrician, numerical correctness,
  Python API design, research integrity/reproducibility, portfolio communication), run
  as parallel subagents; findings adversarially spot-checked against source before
  inclusion. Reviewed at commit `fa7d976` on `main` (clean tree).
- **Phase B** — simulated journal panel (Editor-in-Chief, methodology, domain,
  cross-disciplinary perspective, Devil's Advocate) over SPEC.md + ADRs + docs as a
  methods manuscript, per the academic-paper-reviewer protocol. Reviewers were
  independent: no cross-referencing, no Phase A findings shared.

Verification battery run alongside: pytest green (98.29% coverage, ≥95% gate), pyright
strict clean, ruff clean, `examples/audit_summeval.py` runs offline, secret scan of the
full git history and tracked files clean, PyPI status checked live.

---

## Headline verdict

**The statistics are sound; the shell has drifted.** Two opus reviewers independently
verified the math (ICC CI algebra against R `irr`, PO/LR/Brant/BCa/Holm against pinned
oracles) and found **no numerical errors**. Every blocker/major finding is in the
documentation, release, and CI layer — and all are mechanical fixes, together well under
a day of work. Not ready to resume-link today; ready after the fix list below.

---

## Phase A — consolidated findings (severity-ranked)

All blocker/major findings below were independently confirmed by a second check
(spot-check command or verification battery) before inclusion.

### Blockers (break the first 15 minutes of a reviewer's visit)

| # | Finding | Evidence | Fix |
|---|---------|----------|-----|
| A1 | **`pip install metajudge` is broken — package unpublished.** README leads with it (README.md:21, :98) and carries a PyPI badge; no git-install fallback anywhere. | `pypi.org/pypi/metajudge/json` → HTTP 404 (checked live 2026-07-02). | Publish to PyPI (publish.yml + OIDC already exist) or add `pip install git+https://github.com/breesemarkides/metajudge@<tag>` fallback and drop/replace the badge until published. |
| A2 | **Demo outputs shown to readers are stale against `main`.** README demo block, `examples/sample_output.txt`, and `docs/interop-epic.md` all predate commit `fa7d976`, which added the nested-strata note (and, for interop-epic, a residual-impurity warning that changes how the shown class-C effect should be read). README claims "these are the live demo numbers, not a mock-up". | `diff examples/sample_output.txt <(uv run python examples/audit_summeval.py)` → live output has the `> Note: strata nest items…` block; `grep 'strata nest items' README.md examples/sample_output.txt docs/interop-epic.md` → no hits. | Regenerate all three from current main; add a CI step that runs the example and diffs against the committed sample so this class of drift can't recur. |

### Major

| # | Finding | Evidence | Fix |
|---|---------|----------|-----|
| A3 | **The inferential-calibration layer never runs in CI.** Type-I control (0.025–0.075), power (>0.80), band monotonicity, and bootstrap-coverage tests are all `@pytest.mark.slow`; `conftest.py` skips slow by default and `ci.yml` runs bare `uv run pytest` with no `--run-slow` and no scheduled job. CI enforces only wide quick bands (Type-I < 0.15, power > 0.30) — a calibration regression to Type-I ≈ 0.10 would pass CI. | grep of `.github/workflows/` for `run-slow\|setup-r\|schedule`: no hits. | Scheduled workflow (or pre-tag release gate) running `uv run pytest --run-slow`; provenance-note each reported operating characteristic to a commit where that leg ran green. |
| A4 | **R `MASS::polr` oracle never verified in CI.** `tests/test_oracle_provenance.py` skips when R is absent; CI installs no R. The ordinal (K>2) PO fit's only CI-enforced external anchor is the binary-limit statsmodels check plus pinned literals — README.md:119 and roadmap.md:40 advertise the R check as a standing guarantee. | `test_oracle_provenance.py:30` skip; ci.yml has no R setup step. | Add `r-lib/actions/setup-r` + MASS to one matrix leg (3.11 only, keeps CI fast), or reword README/PROVENANCE to "local/manual provenance gate, not run in CI" — the honest version of the current claim. |
| A5 | **docs/DECISIONS.md indexes 4 of 7 ADRs** while README.md:133 sends readers there as *the* curated index. Missing: cluster-bootstrap, nested-strata-confound, report-card-inference-caveat — three of the most substantive records. | `ls docs/decisions/` vs grep of DECISIONS.md (spot-checked). | Add the three entries in the existing format. |
| A6 | **Phantom ADR cross-reference.** `2026-07-01-e07-dif-nested-strata-confound.md:4` cites `2026-06-22-e07-dif-method-and-matching.md`, which was never created (full-history `git log --diff-filter=A` check: no rename, no add). The matching content actually lives in `2026-06-22-e07-dif-ordinal-logistic-regression.md:32-66`. | grep + git history (spot-checked). | Point the citation at the ordinal-logistic-regression ADR. |
| A7 | **Orphan citation.** `2026-06-23-e07-dif-cluster-bootstrap.md:16` cites "Finch et al. (2025)" as authority for the principled mixed/GEE fix; the citation appears in no References section and not in docs/REFERENCES.md. | grep (spot-checked): only French & Finch (2010) exists. | Add the full citation to the ADR references + REFERENCES.md, or correct/remove if a drafting slip. |
| A8 | **Reported numbers not traceable to a tag** — violates AGENTS.md's own binding rule. README's ICC CIs (`0.573 [0.449, 0.664]`, `0.801 [0.710, 0.856]`) come from McGraw–Wong CI code added in `054f2f3`, after the only tag `v0.1.0` (whose `IccResult` had no CI fields). CHANGELOG "Unreleased" holds BCa, `holm_adjust`, sweep, overlap diagnostic. | `git show v0.1.0:...` diffs (integrity reviewer). | Cut the next tag (v0.2.0 — CHANGELOG's Removed section makes it breaking) after A2's regeneration lands, and tidy the doubled Added/Changed headers in Unreleased first. |
| A9 | **`level` parameter is an unvalidated `str` passthrough.** `krippendorff_alpha(ratings, level="bogus")` → `TypeError: 'str' object is not callable` from inside the krippendorff package; `reliability.py:73` does a `cast()` (no runtime check) and the `Literal` type is module-private, so strict-pyright users get no static help. Same passthrough on `audit()`. | Probe by API reviewer; cast confirmed at reliability.py:73. | Export `LevelOfMeasurement` Literal, type both params with it, add a membership check with a clear `ValueError`. |
| A10 | **Raw `Ratings()` constructor bypasses all validation.** Duplicate item-rater cells construct fine and only explode later in `.wide()` with a generic pandas reshape error; `__init__` has no docstring steering users to `from_long`. | Probe by API reviewer. | Docstring warning pointing at `from_long`/`from_eval_instruments`, or underscore the constructor. |

### Minor

- **M1** (psychometrician) `dif.py:1-17` — module docstring never states the design
  inversion (group = item-level property, disjoint strata, between-set conditioning);
  the ADR and report card do. One sentence + ADR pointer.
- **M2** (psychometrician) `dif.py:34-36, 212-227` — Jodoin–Gierl 0.035/0.070 thresholds
  were calibrated on dichotomous items; the ordinal transfer follows lordif (Choi 2011)
  but that caveat is undisclosed in `_classify_jodoin_gierl`'s docstring (sample-size
  caveat is disclosed; this one isn't).
- **M3** (psychometrician) `reliability.py:57-71` — alpha CI is a naive percentile
  bootstrap (disclosed) while DIF gets cluster+BCa; internal asymmetry worth a note or a
  BCa upgrade.
- **M4** (numerics) — the sim's quick Type-I bound (<0.15) cannot catch a df off-by-one
  (χ²₂ scored as χ²₁ gives ≈0.146); the pinned p-value oracle in `test_dif.py:281-283`
  is the real guard. Don't credit the sim with df discrimination at CI settings.
- **M5** (numerics) `dif.py:184-209` — no DIF direction reported: LR/Nagelkerke is
  sign-invariant, so which group is disadvantaged is never in the output; ETS practice
  reports it. Consider surfacing the group coefficient's sign.
- **M6** (API) `data.py:152` — `coder_unit_matrix` returns bare `np.ndarray` with a
  type-ignore; everything else uses `NDArray[np.float64]`.
- **M7** (API) — `DifResult.dif_class`, `.conditioner_source`,
  `ClusterBootstrapDif.ci_method`, `.cluster` are unconstrained `str`; Literal-type them.
- **M8** (portfolio) CHANGELOG "Unreleased" — no entry for the `fa7d976`
  overlap-diagnostic feature; also doubled Added/Changed headers (see A8).

### Nits

`f_u`/`f_l` named backwards vs the McGraw–Wong/`irr` convention (bounds assigned
correctly — verified); `_common_support` is range-overlap, not distributional overlap
(docstring could say so); missing docstrings on `Ratings.from_long`/`__init__`;
`test_brant.py` rel=0.02 is the loosest pin in the suite (defensible — advisory-only
Wald stat — keep the rationale documented); non-converged fits return numbers with
`converged=False` (disclosed); bootstrap drops degenerate resamples (disclosed via
`n_effective`/`ci_reliable`); demo `n_effective` drifted 200→199 between v0.1.0 and HEAD
without a CHANGELOG note; CITATION.cff name vs commit-author name mismatch (cosmetic).

### Verified clean

- **No numerical errors.** ICC(2,1)/(2,k) + McGraw–Wong Satterthwaite CIs reproduce R
  `irr` algebraically; ordinal LR-DIF reproduces `MASS::polr` to six figures; Brant
  covariance, BCa (cancellation-invariant jackknife), and Holm are textbook-correct; the
  PO parameterization (σ(t_k−xβ), softplus-ordered thresholds, telescoping
  uniform+nonuniform decomposition, LR divergence guard) is exactly right.
- Cluster bootstrap genuinely resamples whole item blocks stratified within
  focal/reference; RNG plumbed explicitly, no global state.
- Six external oracles run in CI at tight tolerances (krippendorff 1e-9, pingouin 1e-6,
  statsmodels Logit 1e-6, Holm 1e-12, scipy BCa, Shrout–Fleiss literal 1e-5).
- No statsmodels/pingouin runtime imports; dependency rule airtight. No bare except, no
  mutable defaults, no `import *`. `examples/` absent from sdist (verified via build).
- ADR-vs-code spot-checks all pass: `icc()` refuses incomplete data citing ten Hove;
  report card leads with the clustering-robust flag and tags analytic p-values
  `[analytic, unclustered]`; `conditioner_overlap_weak` exists and is computed.
- Secret scan clean across full history and tracked files (the TODO.md rotate-keys item
  is moot for the repo itself).
- CI runs exactly what AGENTS.md/README claim, across the 3.11/3.12/3.13 matrix.
- No overclaiming language anywhere; hedging discipline unusually good.

### Strengths the reviewers agreed on

1. **Oracle discipline is exemplary** — every pillar pinned to an independent external
   reference, and the numerical-reference TDD convention is actually followed, not just
   stated.
2. **The ADRs are the differentiator** — each states rejected alternatives and honest
   limits; the recorded circular-Mantel-Haenszel catch (matching variable contained the
   studied response) is a real methodological bug caught and documented, the single
   strongest "understands measurement invariance" signal in the repo.
3. **Caveats ride above the numbers by design** — `to_markdown()` deliberately renders
   warnings before statistics so excerpts can't drop them; analytic p-values are demoted,
   not headlined; refusals (ICC incomplete data, identifiability) instead of silent bias.
4. **A real simulation-validation layer** (DGP recovery, Type-I/power/monotonicity,
   R-generated fixtures) on top of unit oracles — rare for a solo project, and currently
   invisible from the README (see portfolio recommendations).
5. **Honest self-downgrading interpretation** — the demo reports a significant-but-
   negligible DIF as "signal detectable, magnitude not" instead of stopping at p<0.05.

---

## Phase B — journal panel

Panel: EIC (measurement-methods venue) + R1 methodology (DIF specialist) + R2 domain
(LLM-as-judge literature, incl. live prior-art search per SPEC SC2) + R3 perspective
(computational reproducibility / research software) + Devil's Advocate. Independent
reviews; synthesis below by the session lead.

### Panel verdicts at a glance

| Reviewer | Recommendation | Single most important revision |
|---|---|---|
| EIC | Major revision | Resolve the unmet R20/SC1 preprint commitment (draft the minimal v1 or formally amend SPEC.md to record the gate as open) |
| R1 Methodology | Major revision | Run and **report** a confounded-regime + cluster-structure operating-characteristics study; calibrate or replace the 0.7 overlap flag from it |
| R2 Domain | Minor revision | Reconcile REFERENCES.md with SPEC.md's competitive-landscape claims (verify-or-drop "fl-IRT-ing"; add Fair-IRT, AutoRubric, arXiv:2602.00521, Wang 2023, Bavaresco 2024/25) |
| R3 Perspective | Minor revision | Close the tag/HEAD gap and add an automated example-output-sync test |
| Devil's Advocate | 1 CRITICAL, 4 MAJOR | Efficacy evidence for the lead pillar: report the sim operating characteristics + one worked corpus where DIF is non-negligible and the confound discharged |

Panel rubric medians (0–10): framing coherence 8, statistical rigor 7, domain
contribution 8, reproducibility 7, validation adequacy 5, scope honesty 5–8 (split:
excellent inside ADRs/code, undercut at the outward-facing layer).

### Editorial decision: **Major revision**

The Devil's Advocate CRITICAL bars acceptance under the panel protocol, and it names
the same gap R1's five major findings triangulate independently: **the lead, novel
pillar (DIF across output strata) ships validated for numerical correctness but not for
efficacy.** The sim harness that would supply Type-I/power operating characteristics
exists, is unit-tested, and even contains confounded-regime cells
(`conditioner_comparison_cells`, impact cells) — but its full-precision results are
CI-excluded and reported nowhere; the only empirical demonstration is a null (SummEval,
class A, R²Δ=0.002) on human raters rather than the titular LLM-judge panel. The DA's
venue calibration matters: under a JOSS-style software-artifact framing this downgrades
to MAJOR; under a methods-venue framing (which SPEC R20's preprint implies) it blocks.

**Consensus findings (≥3 reviewers independently):**
1. Operating characteristics unreported / rigorous sim cells never run in the pipeline
   (R1, DA, EIC; matches Phase A A3).
2. Outward-facing layer contradicts repo reality: PyPI 404 vs README install path;
   tagged v0.1.0 lacks the confound caveat that HEAD's card prints — a real installer
   gets the *under-caveated* card (EIC, R3, DA; matches Phase A A1/A2/A8).
3. Flagship evidence mismatch: the printed demo is human raters on 2018-19 systems; the
   real committed live LLM-judge run (16 items, 3 Gemini judges) is a side pointer
   (EIC, R2, DA).

**Notable disagreement, arbitrated:** the psychometrician (Phase A) judged the design
inversion honestly handled ("buries the bodies in the open"); the DA argued the "DIF"
label itself imports an invariance/fairness reading the between-disjoint-sets estimand
cannot support. Both agree the disclosure is real and substantive — the DA explicitly
conceded the confound ADR defense holds and the cherry-picking attack fails against the
ADR record. Arbitration: keep the method, sharpen the name — state up front (README +
`dif.py` docstring) that the quantity is a DIF-family *screen* over a between-set
conditional comparison, not textbook item-level invariance.

**Panel findings unique but adopted:**
- Reliability block lacks a validity caveat on the card face — "high agreement is not
  evidence the rubric measures the intended construct" (DA; the authors cite exactly
  this literature in ADRs, the knowledge just isn't on the artifact).
- SPEC R20 preprint gate unmet: `research/2026-06-24-metajudge-preprint-outline.md`
  self-declares "not a draft. No prose body is written" (EIC).
- SC2 prior-art re-verification never logged; R2's live search finds **SC2 does not
  trip** — closest work (arXiv:2602.00521, IRT/GRM judge-reliability) explicitly defers
  DIF to future work — but "fl-IRT-ing" cited twice in SPEC.md could not be located
  anywhere and must be verified or dropped (R2).
- MFRM archival rationale category slip: py-irt/tinyBenchmarks/HELM-IRT do ground-truth
  IRT, not rater-facet MFRM — rest the archival on scope/effort alone (DA).
- `SOURCE.md:54` still describes the pre-June-22 Mantel-Haenszel conditioner (R3).
- Null-result alternative explanation untested: the rest-score conditioner may absorb
  the real extractive/abstractive gap; the external-conditioner path would test it (DA).
- No `audit()`-level multi-strata sweep wiring; k strata = k manual calls + manual Holm
  bookkeeping despite `sweep()`/`holm_adjust()` existing (R3).

---

## Prioritized fix list

### Tier 1 — before resume-linking (mechanical, ~half a day total)

1. **Fix the front door** (A1): publish to PyPI (publish.yml + OIDC exist) or add a
   `pip install git+…@<tag>` fallback and align the badge. Nothing else matters if the
   first command fails.
2. **Regenerate stale outputs** (A2): README demo block, `examples/sample_output.txt`,
   `docs/interop-epic.md` from current main; add the ~5-line pytest that runs each
   example and diffs against its committed sample (R3) so the drift class dies.
3. **Cut the next tag** (A8/R3): v0.2.0 (CHANGELOG "Removed" makes it breaking) after
   items 1–2 land — this also closes the safety-relevant skew where an installer gets
   the card without the confound caveat. Tidy the doubled CHANGELOG headers first (M8).
4. **Docs integrity sweep** (A5, A6, A7, R3): index the 3 missing ADRs in DECISIONS.md;
   repoint the phantom cross-reference; fix or drop "Finch et al. (2025)"; update
   `SOURCE.md:54`.
5. **Bibliography reconciliation** (R2): verify-or-drop "fl-IRT-ing" from SPEC.md; add
   Fair-IRT, AutoRubric, arXiv:2602.00521, Wang et al. 2023, Bavaresco et al. 2024/25
   to REFERENCES.md; log the SC2 re-check as a dated note (result: does not trip).
6. **Small code fixes** (A9, A10, DA, M1, M2): export + validate `LevelOfMeasurement`
   with a clear ValueError; docstring-guard the raw `Ratings()` constructor; one
   validity-caveat line on the reliability block of the card; one design-inversion
   sentence in `dif.py`'s module docstring; JG dichotomous-origin clause in
   `_classify_jodoin_gierl`.

### Tier 2 — before the preprint ships (the substantive work, days)

7. **Run and report the operating-characteristics study** (R1's headline, defuses the
   DA CRITICAL): sweep strata separation (mu_focal) × conditioner reliability ×
   panel size/rater_sd × group balance; PO-violation × {null, DIF} cells; N sweep +
   unbalanced cell; analytic vs bootstrap Type-I side by side; false-DIF-rate surface
   that calibrates or replaces the 0.7 overlap constant. Publish the table at a tagged
   commit per AGENTS.md's own rule. If analytic inflation never appears in this crossed
   design, report that too — it's a finding.
8. **CI-gate the rigor** (A3, A4): scheduled (or pre-tag) workflow running
   `pytest --run-slow` plus an R-enabled leg (3.11 only) for the polr provenance test;
   or, minimum, reword README/PROVENANCE to "local/manual gate" honestly.
9. **Fix the evidence asymmetry** (EIC/R2/DA): promote the live LLM-judge report card
   into the README body beside SummEval (or reframe headline as rater-agnostic-first);
   run the external-conditioner variant on the SummEval demo to test the
   rest-score-absorption alternative explanation of the null.
10. **Close the R20 gate** (EIC): write the minimal preprint v1 prose the requirement
    was scoped to make cheap — or formally amend SPEC.md recording the gate open.
    Tier-2 item 7's results table is the preprint's missing evidence section; do them
    together.
11. **Disclose the second crossing** (R1): one sentence on the card/ADR that the item
    bootstrap leaves cross-item within-rater dependence unhandled (SEs still mildly
    optimistic); quantify in item 7's sweep.

### Tier 3 — optional polish (post-preprint)

BCa for the alpha CI (or a documented why-not, M3); surface the DIF direction sign
(M5); wire `sweep()`+Holm into `audit()` for multi-strata runs (R3); Literal-type the
four string fields (M7) and `NDArray` the matrix return (M6); JOSS-style Statement of
Need + one-line focal/reference glossary for ML readers (R3); rest the MFRM archival on
scope/effort alone (DA); README surfacing of the two buried differentiators — the
Mantel-Haenszel circularity catch and the `sim/` oracle harness (portfolio reviewer:
"stronger evidence of psychometric judgment than the headline numbers").

---

## Review provenance

Phase A personas: psychometrician (opus), numerical-correctness (opus), API design
(sonnet), research integrity (sonnet), portfolio (sonnet); synthesis by the session
lead with adversarial spot-checks. Phase B panel per academic-paper-reviewer v1.10
protocol: EIC (sonnet), methodology (opus), domain incl. live prior-art search
(sonnet), perspective (sonnet), Devil's Advocate (opus); independent reviews, editorial
synthesis by the session lead. Reviewed at `main` @ `fa7d976`, 2026-07-02.
