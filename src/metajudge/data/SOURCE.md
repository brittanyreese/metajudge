# SummEval demo data

Source: Fabbri, A., Kryscinski, W., McCann, B., Xiong, C., Socher, R., and Radev, D. (2020).
SummEval: Re-evaluating summarization evaluation. arXiv:2007.12626.
Repository: https://github.com/Yale-LILY/SummEval
License: MIT.

## Subset

Dimension: coherence. Annotators: 3 expert raters. Items: 1600 (100 documents x 16 systems).
Scores: integers 1-5. Stratum: system family (extractive or abstractive) per the Fabbri et al.
(2020) model taxonomy. Prepared by scripts/prep_demo.py (not in wheel).

## Annotator identity

SummEval's expert_annotations field contains a list of 3 annotation dicts per summary.
The released JSONL contains no annotator ID fields. Slot ordering (index 0, 1, 2 mapped to
expert_0, expert_1, expert_2) is assumed consistent across all 1600 items. prep_demo.py
checks for annotator ID keys at prep time and logs the result.

Consequences:
- Krippendorff alpha: unaffected; it measures within-item agreement regardless of slot identity.
- ICC(2,1) and ICC(2,k): valid under the slot-stability assumption. Treat ICC estimates as
  conditional on this assumption.
- DIF (Mantel-Haenszel): uses per-item mean score; rater identity is not used.
