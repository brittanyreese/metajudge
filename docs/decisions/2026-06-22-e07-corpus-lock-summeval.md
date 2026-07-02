# E07 corpus lock: SummEval

Status: Accepted for E07. Date: 2026-06-22.
Locks the demo corpus to SummEval, replacing the earlier HealthBench working assumption.

## Decision

Lock the E07 demo to SummEval (Fabbri et al., 2020, arXiv:2007.12626). HealthBench is demoted to an optional second demo for a healthcare-prestige framing.

Source correction (verified 2026-06-22): the per-rater matrix is in the original SummEval release, model_annotations.aligned.jsonl, hosted at the Salesforce research bucket (storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl) and in the Yale-LILY/SummEval repo. It is NOT in the mteb/summeval HuggingFace parquet, which ships only aggregated expert means per dimension (100 rows, no crowd scores, no per-rater arrays). load_demo() must read the original jsonl, not the mteb mirror.

## Why

SummEval ships a ready crossed matrix, so it avoids the data-sourcing work that HealthBench would have required (generating responses, then running a judge panel with repeats). It exercises both pillars of the report card directly:

- Reliability. 100 source articles summarized by 16 systems, each summary scored by 8 raters (3 experts, 5 crowdworkers) on 4 graded dimensions, with per-rater scores retained. That is a crossed subject-by-rater-by-dimension matrix, so Krippendorff alpha and ICC compute without generating anything.
- DIF. Scores are graded 1 to 5, and the rater pool splits into expert and crowd, which is a ready differential-rater-functioning group. System identity gives a second grouping variable.

License is MIT and the dataset is tutorial-sized, which fits the showcase goal.

## What it changes

1. Method moves off binary-only. SummEval is graded on a 5-point scale, which clears the roughly 5-category threshold for ordinal logistic regression power (Allahyari et al., 2016). So the ordinal-logistic path, deferred in the method decision because HealthBench was binary, is now in scope and becomes the primary per-dimension DIF method (Zumbo, 1999), with the polytomous Mantel or standardized mean difference as the ETS-canonical secondary. Binary Mantel-Haenszel no longer applies to the primary demo, since there is no binary item; the shipped binary MH in `dif.py` is retained for the optional HealthBench path and as the dichotomous reference implementation.
2. The matching decision holds and is reinforced. SummEval has 4 distinct quality dimensions, so the rubric total is multidimensional and the unidimensionality gate applies. Match within dimension or test by bundle when the gate fails.
3. Reliability pillar is now first-class on real data, not a generated matrix.

## Data-model mapping (SummEval to Ratings)

- subject: the summary, identified by (article id, system id). About 1,600 summaries.
- rater: the annotator id, with a rater_type field (expert or crowd).
- item or dimension: the 4 graded dimensions (coherence, consistency, fluency, relevance).
- score: the ordinal 1 to 5 grade. Keep it ordinal and general, as the prior reshape required.
- stratum: rater_type (expert versus crowd) and system id are the DIF grouping variables.
- matching total: the per-summary quality total, gated by the unidimensionality check.

## Verified on download (2026-06-22)

The data-inspection gate is closed. From model_annotations.aligned.jsonl (1,600 records parsed):

- Structure: 1,600 summaries (100 documents by 16 systems), each with exactly 3 expert_annotations and 5 turker_annotations. Keys: decoded, expert_annotations, turker_annotations, id, model_id, references, filepath.
- Granularity: all four dimensions (coherence, consistency, fluency, relevance) are integers from 1 to 5, all five levels present. This clears the roughly 5-category threshold for ordinal logistic regression power, so the OLR path is sound.
- Per-cell N: 4,800 expert and 8,000 crowd ratings per dimension. Sliced by system and rater type, it is 300 expert and 500 crowd per dimension per system, all above the ETS 200 focal floor. Thick matching is advisable but the floor is not at risk.
- Dimensionality, empirical: the four expert-mean dimensions correlate only 0.32 to 0.66 (coherence-relevance highest at 0.66, most pairs 0.3 to 0.5). The matching total is multidimensional in fact, not just in principle, so the unidimensionality gate fails and within-dimension or bundle matching is required. This confirms the matching decision on real data.

## Open items

- Wire `load_demo() to Ratings` (E07 plan Task 8) from the original model_annotations.aligned.jsonl, not the mteb parquet. Map each (id, model_id) to a subject, expand the 3 expert plus 5 crowd annotations into rater rows with a rater_type field, and emit one ordinal score per dimension. Record provenance and the SummEval MIT license in a SOURCE file.

## References

Allahyari, E., Jafari, P., and Bagheri, Z. (2016). A simulation study to assess the effect of the number of response categories on the power of ordinal logistic regression for differential item functioning analysis in rating scales. Computational and Mathematical Methods in Medicine, 2016, 5080826. doi:10.1155/2016/5080826

Fabbri, A. R., Kryscinski, W., McCann, B., Xiong, C., Socher, R., and Radev, D. (2020). SummEval: Re-evaluating Summarization Evaluation. arXiv:2007.12626.

Zumbo, B. D. (1999). A Handbook on the Theory and Methods of Differential Item Functioning (DIF): Logistic Regression Modeling as a Unitary Framework for Binary and Likert-Type (Ordinal) Item Scores. Ottawa: Directorate of Human Resources Research and Evaluation, Department of National Defence.
