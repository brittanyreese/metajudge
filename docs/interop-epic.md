# Interop: auditing an Epic evaluation-instruments judge

Epic's [`evaluation-instruments`](https://github.com/epic-open-source/evaluation-instruments) is a judge runner: it takes clinical text and a rubric (PDSQI-9, the 5 C's, and others) and returns an LLM's scores per sample per criterion through `post.frame_from_evals`. metajudge is the auditor: it takes those scores and reports whether the judge instrument is reliable and whether it functions differently across a stratum. The two compose directly.

## The seam

`Ratings.from_eval_instruments` maps one `frame_from_evals` output per judge into the long-format `Ratings` that the audit pillars consume. The measurement frame is rater = judge, item = evaluated sample, score = one rubric criterion. Rubric criteria are a separate facet, audited one at a time, never treated as raters; the cited reasoning is in the [interop ADR](decisions/2026-06-23-e07-eval-instruments-interop.md). metajudge consumes only the DataFrame, so it never imports Epic and adds no dependency.

## A runnable example

This builds a small fixture in Epic's real `frame_from_evals` schema (the `(criterion, {class, score, notes})` MultiIndex), so it runs with no PHI and no model call. Three judges score twelve clinical summaries on one rubric criterion, split across two output strata.

```python
import pandas as pd
from metajudge import Ratings, audit

samples = [f"sum{i:02d}" for i in range(12)]

def judge_frame(scores):
    # Epic frame_from_evals "detailed" shape for one judge.
    data = {("organization", "score"): scores, ("organization", "class"): ["-"] * len(scores)}
    return pd.DataFrame(data, index=samples, columns=pd.MultiIndex.from_tuples(data.keys()))

stratum = {s: ("abstractive" if i < 6 else "extractive") for i, s in enumerate(samples)}
frames = {
    "judgeA": judge_frame([4, 5, 4, 3, 5, 4, 3, 2, 3, 4, 2, 3]),
    "judgeB": judge_frame([5, 4, 5, 4, 4, 5, 2, 3, 2, 3, 3, 2]),
    "judgeC": judge_frame([4, 4, 5, 3, 5, 4, 3, 2, 2, 3, 2, 3]),
}

ratings = Ratings.from_eval_instruments(frames, criterion="organization", stratum=stratum)
print(audit(ratings, focal="abstractive", reference="extractive").to_markdown())
```

Output:

```
# metajudge report card

## Reliability
- Krippendorff's alpha (ordinal): 0.723 [95% CI 0.418, 0.739]
- ICC(2,1): 0.711; ICC(2,k): 0.880 (12 targets x 3 raters)

## DIF (panel-relative, rest-score conditioner)
> Note: the rest-score conditioner cannot see bias shared across the entire rater panel, so this is panel-relative DIF, not an instrument-level fairness clearance. Pass a valid independent external quality conditioner; instrument-level interpretation requires it and holds only when the conditioner is valid, independent, and appropriate to the evaluation context.

- abstractive vs extractive (conditioner: rest_score, n=36)
- Uniform DIF: chi2(1)=8.16, p=0.0043
- Nonuniform DIF: chi2(1)=0.27, p=0.6060
- Effect size (Nagelkerke R2 delta): 0.111 (Jodoin-Gierl class C)
```

These numbers come from a synthetic fixture and illustrate the seam and the report-card format, not a finding about any clinical instrument.

## On real Epic data

Epic ships example clinical inputs, not saved judge outputs, and the full instruments are PHI-bearing and access-controlled. To audit a real instrument, run Epic's judge to produce a `frame_from_evals` DataFrame per judge or per run on your own governed data, then pass those frames to `Ratings.from_eval_instruments` exactly as above. Nothing leaves your environment: the adapter is a local DataFrame transform. Pass a valid independent external quality conditioner to the DIF step for instrument-level interpretation; see the DIF ADR for caveats.
