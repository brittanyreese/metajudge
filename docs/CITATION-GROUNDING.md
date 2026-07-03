# Citation Grounding Report

Date: 2026-07-02
Method: [citepipe](https://github.com/) ingestion pipeline (this project's sibling
repo, `data/sitesucker`), Zotero-mediated acquisition, and per-document claim-fidelity
audit against extracted source text.
Corpus paths (in the citepipe repo): `data/judge-audit-corpus/` (staged PDFs +
sidecars), `data/processed-judge-audit/` (`verified_chunks.jsonl`,
`quarantine_chunks.jsonl`, `manifest.json`, `corpus.db`), `data/judge-audit-corpus/verdicts.jsonl`
(claim-level verdicts), `data/judge-audit-corpus.manifest.json` (acquisition manifest).

## Summary

| Metric | Count |
|---|---|
| Total bibliography entries (`docs/REFERENCES.md`) | 48 |
| Software entries (existence-only, not a paper) | 1 (AutoRubric) |
| Acquisition targets (real papers) | 47 |
| Full text staged and citepipe-verified | 37 |
| Staged but quarantined (scanned PDF, no text layer) | 1 (Cortina, 1993) |
| No accessible full text (paywalled / no OA copy) | 8 |
| Book, no OA PDF expected by design | 1 (Krippendorff, 2004) |
| Total claims extracted from usage sites (ADRs, SPEC.md, research notes, REFERENCES.md rationale) | 66 |
| Claims verdict: **supported** | 46 |
| Claims verdict: **not-supported** | 0 |
| Claims verdict: **can't-verify** (no local text or evidence truncation) | 20 |

**Every claim that could be checked against real, extracted paper text held up.** No
citation was found to misrepresent, overstate, or fabricate what its source paper
says. The `can't-verify` claims are entirely a function of acquisition gaps (8
paywalled papers with no accessible full text, plus one truncated large textbook
excerpt) — not fidelity failures.

## Method

1. **Acquisition** (Zotero collection `metajudge`, key `BS995K9X`): each of the 47
   real citations in `docs/REFERENCES.md` was looked up and, where an accessible copy
   existed, imported and PDF-attached. 38 got a staged PDF; 8 remain `imported_no_pdf`
   (paywalled psych/education journal articles with no open-access copy found via
   Unpaywall/CORE/BASE); 1 (Krippendorff 2004) is a Sage book with no OA PDF possible
   by design.
2. **Ingestion**: `citepipe run --corpus data/judge-audit-corpus --output
   data/processed-judge-audit` — format-specific extraction with source anchors
   (page/bbox), SHA-256 tracking, and Jaccard quote-match verification (>= 0.7)
   against the source PDF. 100% of non-quarantined chunks verified; one file
   (Cortina 1993) quarantined as a scanned image with no text layer (citepipe does
   not OCR).
3. **Claim extraction**: read every usage site — the 7 ADRs in `docs/decisions/`,
   `SPEC.md`, the 3 `research/*.md` notes, and `docs/REFERENCES.md`'s own
   parenthetical rationale annotations — for specific, checkable assertions
   attributed to a citation (a finding, a threshold, a methodological warrant), not
   bare bibliography-list entries.
4. **Verdict**: each claim was checked against the cited paper's actual extracted
   text (deduplicated; the raw pipeline output carried ~5x chunk redundancy from
   per-worker re-chunking, collapsed before comparison) and marked `supported`,
   `not-supported`, or `can't-verify`. A claim whose citation has no local text was
   auto-verdicted `can't-verify` without further investigation.

## Per-citation table

| Key | Authors | Year | Acquisition | Verification | Claim verdicts |
|---|---|---|---|---|---|
| krippendorff-2004 | Krippendorff, K. | 2004 | imported, no PDF expected (book) | n/a | no claims found referencing it |
| hayes-krippendorff-2007 | Hayes, A. F., Krippendorff, K. | 2007 | staged | verified | supported (1) |
| shrout-fleiss-1979 | Shrout, P. E., Fleiss, J. L. | 1979 | staged | verified | supported (1), can't-verify (1) |
| mcgraw-wong-1996 | McGraw, K. O., Wong, S. P. | 1996 | staged | verified | supported (2) |
| koo-li-2016 | Koo, T. K., Li, M. Y. | 2016 | staged | verified | supported (1) |
| ebel-1951 | Ebel, R. L. | 1951 | no PDF found | n/a | can't-verify (1) |
| nakagawa-schielzeth-2010 | Nakagawa, S., Schielzeth, H. | 2010 | no PDF found | n/a | no claims found referencing it |
| brueckl-heuer-2022 | Brueckl, M., Heuer, F. | 2022 | staged | verified | supported (1) |
| tenhove-etal-2024 | ten Hove, D., Jorgensen, T. D., van der Ark, L. A. | 2024 | staged | verified | supported (1) |
| tenhove-etal-2025 | ten Hove, D., Jorgensen, T. D., van der Ark, L. A. | 2025 | staged | verified | supported (1) |
| mccullagh-1980 | McCullagh, P. | 1980 | staged | verified | supported (1) |
| zumbo-1999 | Zumbo, B. D. | 1999 | staged | verified | supported (2) |
| swaminathan-rogers-1990 | Swaminathan, H., Rogers, H. J. | 1990 | staged | verified | supported (2) |
| crane-etal-2006 | Crane, P. K., Gibbons, L. E., Jolley, L., van Belle, G. | 2006 | staged | verified | supported (2) |
| choi-gibbons-crane-2011 | Choi, S. W., Gibbons, L. E., Crane, P. K. | 2011 | staged | verified | supported (2), can't-verify (1) |
| allahyari-etal-2016 | Allahyari, E., Jafari, P., Bagheri, Z. | 2016 | staged | verified | supported (1) |
| nagelkerke-1991 | Nagelkerke, N. J. D. | 1991 | staged | verified | no claims found referencing it |
| jodoin-gierl-2001 | Jodoin, M. G., Gierl, M. J. | 2001 | staged | verified | supported (2) |
| gomezbenito-etal-2009 | Gomez-Benito, J., Hidalgo, M. D., Padilla, J. L. | 2009 | staged | verified | supported (1) |
| holland-thayer-1986 | Holland, P. W., Thayer, D. T. | 1986 | staged | verified | supported (1) |
| dorans-holland-1992 | Dorans, N. J., Holland, P. W. | 1992 | staged | verified | supported (2) |
| clauser-etal-1993 | Clauser, B. E., Mazor, K. M., Hambleton, R. K. | 1993 | no PDF found (ILL requested) | n/a | can't-verify (3) |
| french-maller-2007 | French, B. F., Maller, S. J. | 2007 | staged | verified | supported (2) |
| magis-etal-2010 | Magis, D., Beland, S., Tuerlinckx, F., De Boeck, P. | 2010 | staged | verified | supported (1) |
| brant-1990 | Brant, R. | 1990 | staged | verified | supported (1) |
| harrell-2015 | Harrell, F. E. | 2015 | staged | verified | can't-verify (2, truncated) |
| liang-zeger-1986 | Liang, K.-Y., Zeger, S. L. | 1986 | staged | verified | supported (1) |
| dennoortgate-deboeck-2005 | den Noortgate, W. V., De Boeck, P. | 2005 | staged | verified | supported (1) |
| french-finch-2010 | French, B. F., Finch, W. H. | 2010 | staged | verified | supported (2) |
| musca-etal-2011 | Musca, S. C., et al. | 2011 | staged | verified | supported (1) |
| cameron-miller-2015 | Cameron, A. C., Miller, D. L. | 2015 | staged | verified | supported (1) |
| cronbach-1951 | Cronbach, L. J. | 1951 | no PDF found | n/a | can't-verify (1) |
| cortina-1993 | Cortina, J. M. | 1993 | staged | **quarantined** (scanned, no text layer) | can't-verify (1) |
| brennan-2001 | Brennan, R. L. | 2001 | no PDF found | n/a | can't-verify (1) |
| lim-2011 | Lim, G. S. | 2011 | no PDF found | n/a | can't-verify (1) |
| hallgren-2012 | Hallgren, K. A. | 2012 | staged | verified | supported (1), can't-verify (1) |
| mcneish-2018 | McNeish, D. | 2018 | no PDF found | n/a | can't-verify (1) |
| wang-luo-2019 | Wang, J., Luo, K. | 2019 | staged | verified | supported (2) |
| li-2022 | Li, W. | 2022 | no PDF found | n/a | can't-verify (3) |
| zheng-etal-2023 | Zheng, L., Chiang, W.-L., Sheng, Y., et al. | 2023 | staged | verified | supported (1), can't-verify (1) |
| norman-etal-2026 | Norman, J. D., Rivera, M. U., Hughes, D. A. | 2026 | staged | verified | supported (2) |
| wang-etal-2023-not-fair-evaluators | Wang, P., et al. | 2023 | staged | verified | supported (1) |
| bavaresco-etal-2024 | Bavaresco, A., et al. (20 authors) | 2024 | staged | verified | supported (1) |
| bachmann-etal-2024-flirting | Bachmann, D., et al. | 2024 | staged | verified | supported (1) |
| xu-etal-2025-fairness-irt | Xu, Z., Kandanaarachchi, S., Ong, C. S., Ntoutsi, E. | 2025 | staged | verified | supported (1) |
| choi-etal-2026-irt-judge | Choi, J., et al. | 2026 | staged | verified | supported (2) |
| fabbri-etal-2020-summeval | Fabbri, A. R., et al. | 2020 | staged | verified | supported (1) |
| autorubric-software | AutoRubric | — | software, not a paper | n/a (existence via URL) | can't-verify (1) |

## Acquisition-failed citations (no local full text)

Nine citations have no accessible full text; every one is documented, none is a
silent gap:

- **ebel-1951** — paywalled, no OA copy found.
- **nakagawa-schielzeth-2010** — paywalled, no OA copy found.
- **clauser-etal-1993** — paywalled; an interlibrary-loan request is outstanding
  (submitted by the user during acquisition).
- **cronbach-1951** — paywalled, no OA copy found.
- **cortina-1993** — PDF was staged, but it is a scanned image with no text layer;
  citepipe does not OCR, so it quarantined rather than fabricate extracted text.
- **brennan-2001** — paywalled; Zotero's auto find-pdf attached the wrong document
  (an NCME instructional module by the same author, mislabeled with the correct
  chapter title) during acquisition — deleted; no legitimate PDF exists locally.
- **lim-2011** — paywalled, no OA copy found.
- **mcneish-2018** — paywalled, no OA copy found.
- **li-2022** — paywalled, no OA copy found.
- **krippendorff-2004** — a Sage book; no OA PDF possible by design (`fetchability:
  book-no-oa-pdf-expected` in the manifest).

Two wrong-paper substitutions were caught and rejected during acquisition, worth
recording since they show the acquisition process actively guarded against
mis-grounding: (1) a `Slocum_Gelin_Zumbo.pdf` that shared an author but was not
the actual Zumbo (1999) report (fixed with the real handbook); (2) Zotero's
auto-find-pdf mis-attaching an NCME instructional module to Brennan (2001) (see
above). A third candidate (Fidler 2013, "Mantel-Haenszel Procedure Revisited"),
proposed by the user as a possible substitute for Clauser et al. (1993), was
correctly rejected as a different paper backing a different claim.

## Claim-fidelity findings

**Zero `not-supported` verdicts.** Every claim checked against extracted paper text
matched what the paper actually says, including the claims carrying the most
methodological weight:

- The cluster-bootstrap ADR's core empirical claim — French & Finch (2010) showing
  single-level logistic-regression DIF fails to hold the nominal .05 Type-I rate
  under between-cluster grouping while a hierarchical model restores it — is quoted
  verbatim in the source.
- SPEC.md's SC2 prior-art re-check claim — that Choi et al. (2026) explicitly defers
  DIF-style stratum comparisons to future work — matches the paper's own limitations
  section word for word.
- The Jodoin-Gierl (2001) 0.035/0.070 Nagelkerke ΔR² A/B/C bands, French & Maller
  (2007)'s finding that purification doesn't substantially improve DIF power/Type-I,
  and Crane et al. (2006)'s use of a Brant test before interpreting DIF interactions
  all check out exactly against source text.

The 20 `can't-verify` verdicts break down as: 15 claims whose citation has no local
extracted text (the 9 acquisition-failed citations above), plus 5 claims where the
citation's evidence text exists but the specific detail checked (e.g., a Harrell
2015 textbook passage, lordif's exact default flag value) fell outside the portion
of a large source retrieved for comparison. None of these represent a contradiction
— they represent verification limits, not fidelity failures.

## Documentation-hygiene findings (defect resolutions)

The prior manual review (`docs/reviews/2026-07-02-multi-persona-review.md`) flagged
four items, all now resolved, plus this audit surfaced one additional defect of the
same class:

- **A5** (`docs/DECISIONS.md` indexed only 4 of 7 ADRs) — **fixed** directly
  (confirmed: all 7 ADRs now indexed).
- **A6** (phantom ADR cross-reference in the nested-strata-confound ADR to a
  nonexistent file) — **fixed** directly; the ADR now points to the real section
  (`2026-06-22-e07-dif-ordinal-logistic-regression.md`, lines 32-66) — confirmed by
  reading the current file.
- **A7** (orphan "Finch et al. (2025)" citation in the cluster-bootstrap ADR,
  absent from `REFERENCES.md`) — **fixed**; the current text cites only French &
  Finch (2010), which is fully corpus-verified and its claim confirmed supported.
  The orphan reference was dropped rather than fabricated, per the project's own
  stated resolution.
- **R2** (re-verify fl-IRT-ing/Bachmann 2024, Fair-IRT/Xu 2025, Choi 2026, Wang
  2023, Bavaresco 2024 claims) — **resolved** as a byproduct of this audit: all five
  citations' claims in `SPEC.md` and `docs/REFERENCES.md` are `supported`.
- **New finding (this audit): phantom bibliography entry.** The nested-strata-confound
  ADR's own References section listed "Shealy, R., and Stout, W. (1993)" — never
  cited in the ADR's body text, and absent from the master `docs/REFERENCES.md`
  bibliography (grep-confirmed: zero hits across all 48 entries). Same defect class
  as A7. **Fixed**: dropped from the ADR's References section (not fabricated into
  `REFERENCES.md`), matching the project's own precedent for exactly this situation.

## Reproducing this audit

```bash
# In the citepipe repo (data/sitesucker)
uv run python scripts/stage_zotero_corpus.py \
  --collection BS995K9X --all-in-collection \
  --corpus data/judge-audit-corpus
uv run citepipe run --corpus data/judge-audit-corpus \
  --output data/processed-judge-audit --workers 12
```

`data/judge-audit-corpus.manifest.json` is the acquisition ledger;
`data/judge-audit-corpus/verdicts.jsonl` is the full claim-level audit trail (66
rows: `citation_key`, `claim_text`, `source_file`, `source_line`, `verdict`, `note`).
Both pipelines are resumable — re-running skips completed files/claims.
