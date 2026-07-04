# E07 real-data demo: ELLIPSE human-rater panel, race/ethnicity DIF

Date: 2026-07-04 Status: accepted (worked example; not a claim about ELLIPSE rater fairness in general) Script: `examples/audit_ellipse.py` Data: ELLIPSE Corpus (Crossley et al., 2023; github.com/scrosseye/ELLIPSE-Corpus), CC BY-NC-SA 4.0, not vendored in this repo

## Why this run

Yamashita (2025, Language Testing) scored the ELLIPSE essays with GPT-4o and found significant race/ethnicity bias in a many-facet Rasch model: GPT-4o over-scored Asian/Pacific Islander writers and under-scored Hispanic/Latino writers, with no comparable bias by gender or socioeconomic status. That is an LLM-judge finding. This run asks the same fairness question of ELLIPSE's HUMAN rater panel instead, as a real-data positive-DIF demonstration for metajudge and as the baseline the deferred LLM-judge follow-up (below) will be compared against. No LLM API calls were made in this run.

## Design, pre-registered before the data was scored

Decided and written down before `logistic_dif` was ever called on ELLIPSE data:

- **Item**: essay (`text_id_kaggle`).
- **Rater**: one of two anonymized rating slots per essay, `rater_1` / `rater_2`. The raw file (`ellipsis_raw_rater_scores_anon_all_essay.csv`) draws from a pool of 27 human raters but does not expose individual rater identity, only a per-essay slot label. This is the same convention already used for SummEval's `expert_0`/`expert_1`/`expert_2` in `scripts/prep_demo.py`: a rating slot, not a stable individual across items.
- **Score (primary, pre-registered trait)**: Vocabulary. Chosen a priori because word choice and lexical register are the analytic trait most exposed, in the L2-writing-assessment literature, to rater judgments correlated with a writer's linguistic and cultural background. Yamashita's GPT-4o finding is reported at the holistic-proficiency level, not broken out by analytic trait, so this choice does not borrow foreknowledge of which trait would show an effect.
- **Stratum**: race/ethnicity, collapsed to the pair Yamashita's study flagged: focal = Asian/Pacific Islander, reference = Hispanic/Latino.
- **Conditioner**: external, independent of the studied trait. The leave-one-trait-out mean of the other five analytic traits (Cohesion, Syntax, Phraseology, Grammar, Conventions), pooled across both raters. Holistic Overall is excluded from every analytic trait's conditioner.

A second pass, run only after seeing the primary result, sweeps all 7 score dimensions (Overall plus the 6 analytic traits) with Holm correction. That pass is explicitly labeled exploratory below: its trait choice was not pre-registered, and any hit it finds is hypothesis-generating, not confirmatory.

## Data

Source files (`data/raw/ellipse/`, gitignored, fetched via `git clone` of the corpus repo, both password-protected zips extracted with the passwords documented in the corpus README):

- `ellipsis_raw_rater_scores_anon_all_essay.csv`: 8,890 essays, per-rater raw scores (0-5 integer, not the 0.5-step resolved scale) on Overall plus the 6 analytic traits. This is the raw per-rater file the task called for; the resolved (final, 0.5-step) scores were not needed since the raw file was public.
- `ELLIPSE_Final_github_train.csv` + `ELLIPSE_Final_github_test.csv` (password-protected): demographics (race/ethnicity, gender, SES, grade) for the 6,482 essays in the final reliable corpus.

Cleaning: inner-joined raw ratings to demographics on `text_id_kaggle` (6,468 of 6,482 matched; 14 final-corpus essays have no raw-rater row and 2,422 raw rows are outside the final corpus, most with a malformed or missing id). Dropped 6 essays carrying a literal `0` in any of the 14 raw score columns (a data-quality artifact, not a real point on the documented 1-5 scale; all 6 happen to be Hispanic/Latino). Final N: **6,462 essays**.

Per-stratum N after cleaning:

| Stratum                        | N     |
| ------------------------------ | ----- |
| Hispanic/Latino (reference)    | 4,619 |
| Asian/Pacific Islander (focal) | 791   |
| Black/African American         | 514   |
| White                          | 469   |
| Two or more races/Other        | 46    |
| American Indian/Alaskan Native | 23    |

Both the focal and reference cells clear the >=500 floor the Jodoin-Gierl A/B/C classification was calibrated on (`docs/decisions/2026-06-22-e07-dif-ordinal-logistic-regression.md`); the other four race/ethnicity levels are present in the data but not used in this focal/reference comparison.

## Results: primary, pre-registered analysis (Vocabulary)

Reliability, computed over all 6,462 essays (all six race/ethnicity levels), 2 raters:

- Krippendorff's alpha (ordinal): 0.464 [95% CI 0.445, 0.484]
- ICC(2,1): 0.484 [95% CI 0.465, 0.502]; ICC(2,k): 0.652 [95% CI 0.634, 0.668]

DIF, Asian/Pacific Islander vs Hispanic/Latino, external (leave-one-trait-out) conditioner, n_obs = 10,820 (2 raters x 5,410 items):

- Nagelkerke R2 delta: 0.0000, Jodoin-Gierl class **A** (negligible)
- Total chi2(2): p = 0.882; uniform chi2(1): p = 0.952; nonuniform chi2(1): p = 0.618
- Conditioner-group correlation: 0.084 (well inside the calibrated safe band, |corr| < 0.2; `conditioner_overlap_weak` = False)
- Cluster-robust check (item-cluster bootstrap, n_boot=200, seed=0): R2 delta 95% percentile CI [0.000, 0.079], 189/200 resamples effective, `ci_reliable` = True. Verdict: no robust DIF (the CI reaches the negligible band).

**The pre-registered analysis found no consequential DIF.** On this trait, this stratum pair, and this conditioner, the human-rater panel does not show the bias pattern Yamashita found in GPT-4o. That is itself informative: it is evidence the audit does not manufacture a positive finding by default, which is the honest baseline the LLM-judge follow-up should be read against.

## Results: exploratory sweep (NOT pre-registered)

Same focal/reference pair and conditioner design, all 7 score dimensions, Holm-corrected across the family of 7 total-chi2 tests:

| Trait | R2 delta | Class | p_total (Holm) | conditioner corr | overlap_weak |
| --- | --- | --- | --- | --- | --- |
| Overall | 0.0001 | A | 0.716 | 0.083 | False |
| Cohesion | 0.0000 | A | 1.000 | 0.083 | False |
| **Syntax** | **0.1840** | **C** | **~0** | 0.076 | False |
| Vocabulary | 0.0000 | A | 1.000 | 0.084 | False |
| Phraseology | 0.0001 | A | 1.000 | 0.085 | False |
| Grammar | 0.0010 | A | 5.2e-4 | 0.091 | False |
| Conventions | 0.0015 | A | 7.6e-6 | 0.077 | False |

Grammar and Conventions are statistically significant after Holm correction but land in the Jodoin-Gierl negligible band (R2 delta 0.001 to 0.0015): at n_obs = 10,820 the analytic test has power to detect an effect size with no practical consequence. This is exactly the effect-size-over-p-value framing the report card already leads with (`docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md`).

Syntax is the exception: R2 delta = 0.184, Jodoin-Gierl class **C** (large), Holm-adjusted p approximately 0, conditioner-group correlation 0.076 (safe band, not confounded). This is a substantively large effect by the same classification the pre-registered analysis used, on the same focal/reference pair.

Follow-up robust check (item-cluster bootstrap on Syntax, n_boot=200, seed=0): R2 delta 95% percentile CI [0.000, 0.205], 197/200 resamples effective, `ci_reliable` = True. Because that lower bound sits right at the boundary, a second run with more resamples and a different seed (n_boot=400, seed=1, not part of the committed script but reproducible from `build_ratings`/`build_conditioner` in `examples/audit_ellipse.py`) gives CI [0.0006, 0.213], 397/400 effective: consistent with the first run, and still touching the Jodoin-Gierl 0.035 negligible boundary at the lower end.

**The cluster-robust check does not confirm the point estimate.** Both bootstrap runs put the R2-delta point estimate at a stable 0.184 (class C), but the 95% CI lower bound sits at essentially 0 in both, so under the report card's own robust-flag logic (`docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md`, `docs/decisions/2026-06-23-e07-dif-cluster-bootstrap.md`) this is **not** classified as robustly nonnegligible DIF: the item-cluster resampling shows the analytic point estimate is not stable against which essays happen to be resampled.

## Whether the audit caught consequential DIF

Partially, and the honest answer is more interesting than a clean yes. The pre-registered analysis (Vocabulary) returned an honest class A null, with a cluster-robust CI that also reaches the negligible band; the audit did not manufacture a finding where none was registered. The exploratory Holm-corrected sweep found a class C (large), Holm-significant, non-confounded point estimate on Syntax, which at first read looks like the "positive demo" this task set out to build. But running the audit's own stronger check, the item-cluster bootstrap, on that same trait shows the 95% CI reaching down to the Jodoin-Gierl negligible boundary in two independent bootstrap runs (200 and 400 resamples, two seeds). By the report card's own robust-flag criterion, Syntax is **not** a robustly confirmed instance of DIF from this run.

Read this as the audit's clustering-robust layer doing exactly the job it was built for (`docs/decisions/2026-06-23-e07-dif-cluster-bootstrap.md`): a large, Holm-significant, non-confounded analytic point estimate is exactly the kind of result that invites overclaiming, and the robust check is what stops that here. The Syntax finding is hypothesis-generating, not confirmatory, on two independent grounds now: it was found by looking across 7 traits (not pre-registered), and its own robustness check does not clear the bar. The right next step is a dedicated, pre-registered confirmatory run with a larger bootstrap budget (n_boot >= 1000) and, ideally, a held-out split of the corpus or the deferred LLM-judge panel below, before Syntax is cited as a settled claim about ELLIPSE rater fairness.

## Caveats

1. **Human raters, not an LLM judge.** This audits the corpus's original two-human-rater panel. It is not a test of GPT-4o or any other model; that is the deferred follow-up below.
2. **Rating slots, not stable rater identities.** `rater_1`/`rater_2` are per-essay slot labels drawn from a pool of 27 raters; the raw file does not expose which individual scored which essay. Reliability and DIF here read as "two trained raters per essay," not "the same two people rating everything."
3. **Instrument-level, not panel-relative.** The external leave-one-trait-out conditioner makes this a claim about the trait/stratum pair conditional on the OTHER traits, not merely relative to the two-rater panel's own rest score (which a 2-rater panel cannot support in the first place: `logistic_dif` warns and effectively refuses when N_raters = 2 and no external conditioner is given).
4. **Single stratum pair.** Only Asian/Pacific Islander vs Hispanic/Latino was tested; Black/African American, White, and the two small-N groups (Two or more races/Other, n=46; American Indian/Alaskan Native, n=23) were not screened, and the small-N groups are below the Jodoin-Gierl calibration floor regardless.
5. **Exploratory-vs-confirmatory split is the load-bearing caveat of this report.** The Syntax finding did not exist as a hypothesis before the sweep ran. Holm correction controls the familywise error rate across the 7-trait sweep, and the conditioner overlap is safe, but it is still a post-hoc finding and should be labeled that way in any downstream citation.
6. **Data cleaning.** 6 of 6,468 merged essays were dropped for a literal 0 in a raw score column; this is under 0.1% of the sample and unlikely to move any result, but it is a judgment call (treating 0 as a data artifact rather than a valid floor score) made without ground truth on what produced it.
7. **Wide cluster-bootstrap CIs.** Both DIF checks in this run have an R2-delta CI reaching down to (or within a hair of) 0, even at 791 focal / 4,619 reference items, well above the bootstrap's own minimum cluster-size floor (5). A 200-to-400-resample item-cluster bootstrap on a two-rater panel is evidently not narrow enough to separate "large point estimate" from "negligible" at this stratum size; a confirmatory run should budget for more resamples (n_boot >= 1000, matching the package default) before treating either CI bound as final.

## LLM-judge pilot (run): a self-contained open-model judge

The follow-up above is now partly executed. A self-contained, dependency-free LLM-judge scoring path was added (`examples/_ellipse_judge.py` + `examples/audit_ellipse_llm.py`) and a pilot was run. It replicates the Yamashita AES setup, one LLM judge scoring the essays on the ELLIPSE analytic rubric, with the same DIF question asked of the judge. No private infrastructure: the scorer speaks the OpenAI `/v1/chat/completions` schema against any endpoint (a local `mlx_lm.server`, Ollama's `/v1`, or a hosted API), and the model, prompt, seed, and decoding are all pinned in committed code.

### Pilot design, pre-registered before scoring

- **Judge model**: the pilot was specified as qwen2.5:32b on local Ollama. On the pilot hardware the 32b ran at 25-46 s/essay under memory pressure (several 70B-class models resident), infeasible for 300 essays in-session, so the committed pilot uses its small Qwen sibling, qwen2.5:7b, a one-flag change in the same model family (about 6 s/essay).
- **Prompt**: keyword-style analytic rubric. Each trait is reduced to its salient scoring cues distilled from the official ELLIPSE rubric (`ELL_Rubrics.docx`), rather than the full paragraph descriptors, following the AES finding that keyword prompts beat long-rubric prompts. Temperature 0, fixed seed, structured JSON output (one integer 1-5 per trait), robust parse with a bounded retry.
- **Sample**: a balanced 300-essay pilot, 150 focal (Asian/Pacific Islander) + 150 reference (Hispanic/Latino), the Yamashita-flagged pair, deterministically sampled (seed 0).
- **Traits**: Vocabulary and Syntax (to line up with the human demo's two headline traits) plus holistic Overall, pre-registered here before the audit ran.
- **Conditioner**: external leave-one-trait-out mean of the judge's OWN other analytic-trait scores, keyed by essay, so the match is "quality as the judge sees it," independent of the studied trait. This makes it an instrument-level DIF claim about the single judge, not a panel-relative one (a single judge has no rest-score conditioner and no reliability pillar; alpha/ICC are not reported for one rater).

### Pilot result

Parse-failure rate: **0/300** (every essay returned valid JSON on the first attempt). DIF, Asian/Pacific Islander vs Hispanic/Latino, external conditioner, n_obs = 300 (single judge):

| Trait      | R2 delta | Class | Cluster-robust 95% BCa CI | overlap_weak |
| ---------- | -------- | ----- | ------------------------- | ------------ |
| Vocabulary | 0.0042   | A     | [0.0001, 0.0167]          | False        |
| Syntax     | 0.0009   | A     | [0.0000, 0.0046]          | False        |
| Overall    | 0.0078   | A     | [0.0005, 0.0208]          | False        |

All three land in the Jodoin-Gierl negligible band (class A), with cluster-robust CIs (400 resamples, all effective) contained inside it, and conditioner-group correlations near 0.07, well inside the calibrated safe band, so none is a residual-impurity artifact. n_obs = 300 sits in the "indicative" band (200-499), below the >=500 the A/B/C thresholds were calibrated on, so the class is treated as indicative; the effect sizes are far enough below the 0.035 boundary that this caveat does not change the read.

### Did the LLM judge catch class B/C DIF where the humans did not?

No, not in this pilot. On the same flagged stratum pair, the qwen2.5:7b judge shows negligible (class A) DIF on Vocabulary, Syntax, and Overall alike. Set against the human demo: Vocabulary is a null for both the humans and this LLM judge; Syntax was a large but not-robustly-confirmed class C point estimate for the humans and is a clean class A null for this LLM judge. So the pilot does not reproduce Yamashita's GPT-4o bias pattern. Two honest reasons it should not be over-read as "LLM judges are fair here": the pilot model is a 7B open model, not GPT-4o or the reported 70B; and the single-judge n_obs of 300 is far smaller than the human panel's 10,820, so it has less power to resolve a small effect. The pilot's job is to prove the pipeline end to end and give a local, reproducible baseline, which it does.

### Reproducibility

The pilot scores are committed at `examples/data/ellipse_llm_pilot_qwen2.5-7b.csv` (300 rows, one per essay), so `uv run python examples/audit_ellipse_llm.py` reproduces the DIF audit with no GPU, model, or network. `tests/test_ellipse_llm_pilot.py` pins the three effect sizes and classes off that CSV. Regenerating the scores is the optional GPU path (`--regenerate`), and it is resumable (rerun to continue an interrupted run).

## Reported run and spot-check (parameterized, not yet run)

The committed pilot is a proof of pipeline. The reported judge is `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit` served via `mlx_lm.server` (the keyword-prompt AES literature finds Llama-3.1-70B the strongest open ELLIPSE config). Switching to it is a base_url + model change only:

```
mlx_lm.server --model mlx-community/Meta-Llama-3.1-70B-Instruct-4bit --port 8080
uv run python examples/audit_ellipse_llm.py --regenerate \
    --base-url http://127.0.0.1:8080/v1 \
    --model mlx-community/Meta-Llama-3.1-70B-Instruct-4bit \
    --out examples/data/ellipse_llm_llama31-70b.csv
```

GPT-4o via OpenRouter is available as an optional hosted spot-check to replicate Yamashita's exact model (`--base-url https://openrouter.ai/api/v1 --model openai/gpt-4o`, API key in `ELLIPSE_JUDGE_API_KEY`). A full-corpus GPT-4o run (6,462 essays, one structured multi-trait call each, roughly 400-800 input + 150 output tokens per call) is a low-hundreds-of-dollars job; get budget sign-off first, and consider scaling the reported run to the full corpus rather than the 300-essay pilot for the power to resolve a Yamashita-scale effect.

## References

- Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., and Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus. International Journal of Learner Corpus Research, 9(2), 248-269.
- Yamashita, T. (2025). Exploring potential biases in GPT-4o's ratings of English language learners' essays. Language Testing, 42(3), 344-358. https://doi.org/10.1177/02655322251329435
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349.
- `docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md` (conditioner-overlap diagnostic this run's `conditioner_group_corr` is read against).
- `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md` (the calibrated 0.2 safe-band threshold cited above).
