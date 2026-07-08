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

The pilot was scored with the default prompt (per-trait calls, shared holistic scale anchors, no reasoning step). qwen2.5:7b showed the score-collapse pathology described under "Residual collapse" below only rarely, so its pilot numbers are usable as a pipeline proof; a GPT-4o-class judge needs the collapse-mitigation prompt (the `--reasoning` / `--trait-scoped-anchors` flags) before its scores are a clean measurement.

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
    --reasoning --trait-scoped-anchors \
    --out examples/data/ellipse_llm_llama31-70b.csv
```

GPT-4o via OpenRouter is available as an optional hosted spot-check to replicate Yamashita's exact model (`--base-url https://openrouter.ai/api/v1 --model openai/gpt-4o`, API key in `ELLIPSE_JUDGE_API_KEY`). Any reported or spot-check run on a GPT-4o-class judge must pass `--reasoning --trait-scoped-anchors`: without them GPT-4o collapses its scores (see "Residual collapse" below). Leave the flags off only to reproduce the committed qwen2.5:7b pilot CSV byte-for-byte.

## Batching halo bug found before any full-corpus spend, and the per-trait fix

Before running GPT-4o on the full corpus, an 80-essay ad hoc GPT-4o spot-check and a 179-essay llama-3.3-70b spot-check were run against the batched scorer described above (one chat-completion call per essay, asking for all 7 rubric traits in a single JSON response). Neither trial CSV was committed; both are discarded by this finding.

**Finding.** 51 of the 80 GPT-4o rows had an identical score repeated across all 7 traits (mostly all-2s); the llama-3.3-70b trial showed the same pattern on only 1 of 179 rows. A direct comparison on one essay, same essay and decoding params throughout, confirmed the cause: the batched call (all 7 traits in one JSON request) returned Overall=2 and every other trait=2; a call asking for Overall alone returned Overall=3; a call asking for Grammar alone returned Grammar=2. Response `finish_reason` was `stop` in all cases (46 completion tokens on the batched call), ruling out truncation. Asking GPT-4o to score 7 dimensions in one response induces a halo effect: it forms one holistic impression and repeats it, rather than differentiating per trait. This is not a GPT-4o identity problem; the response metadata confirms genuine `openai/gpt-4o` (`provider: OpenAI` on most calls, one call routed to Azure by OpenRouter's load balancing, with three different `system_fingerprint`s observed across the three calls: `fp_683410201e`, `fp_f38e8ce955`, `fp_85f0f4d955`).

**Fix (necessary, not sufficient).** `examples/_ellipse_judge.py` now makes one chat-completion call per trait (7 isolated calls per essay) instead of one call for all 7. Each call's prompt carries only that trait's keyword rubric, never the other 6, so there is nothing left for the model to anchor across within a single response. `JudgeResult` keeps its existing all-or-nothing contract: `scores` is the full 7-trait dict or `None` (a trait that exhausts its retries stops the essay there, never a partially-filled result). `tests/test_ellipse_judge_scorer.py` pins this: one isolated call per trait, no leakage of other traits' rubric text into a single-trait prompt, and fail-closed (never partial) aggregation. Per-trait isolation removes the halo _within_ a single response, but a per-trait rerun still collapsed on GPT-4o: a second halo channel remained in the prompt itself. The root cause and full fix are in "Residual collapse" below.

**Cost note.** Isolating traits means ~7x the calls per essay, and the essay text is repeated in every call rather than sent once, so a full-corpus GPT-4o run costs roughly 7x the token volume of the original low-hundreds-of-dollars estimate; get budget sign-off against the revised estimate, not the batched-design one, before running the full 6,462-essay corpus.

## Reproducibility protocol per backend

Determinism knobs exist on every backend this scorer talks to, but none of them alone guarantees a byte-identical rerun; each has a distinct failure mode worth pinning against before treating a score CSV as reproducible from source.

- **OpenRouter.** `openai/gpt-4o` on OpenRouter load-balances across providers (OpenAI direct and Azure-hosted OpenAI were both observed serving the same model slug in the spot-check above, each returning a different `system_fingerprint`). Pin the provider with the request's `provider` object: `order: ["openai"]` plus `allow_fallbacks: false` restricts routing to the OpenAI-direct endpoint only (`only: ["openai"]` is the equivalent allow-list form). Add `require_parameters: true` so the request fails closed if a routed provider would silently drop an unsupported parameter (notably `seed`) rather than accept it and ignore it. `seed` is honored on a best-effort basis per-provider on OpenRouter, same caveat as the direct OpenAI API below.
- **OpenAI (direct, or the OpenAI-only route above).** `seed` plus identical other parameters gives "mostly deterministic" output, per OpenAI's own documentation; the `system_fingerprint` response field is the tripwire; log it alongside every scored row and treat a run where it changes mid-run as non-comparable rather than averaging it in silently.
- **Ollama.** The default context window is 4096 tokens (`OLLAMA_CONTEXT_LENGTH` env var, or `options.num_ctx` per-request against the native `/api/generate` and `/api/chat` endpoints), independent of `temperature`/`seed`. A long essay plus the per-trait prompt overhead can approach that ceiling and silently truncate the input rather than error.

## Pins implemented

The three pins above are in `examples/_ellipse_judge.py`, with one correction found while implementing the Ollama pin against the actual API docs rather than assuming the plan would just work.

**OpenRouter provider pin.** `JudgeConfig.provider` defaults to `None`; when `base_url` is OpenRouter and no override was given, `_build_payload` auto-attaches `DEFAULT_OPENROUTER_PROVIDER` (`order: ["openai"]`, `allow_fallbacks: false`, `require_parameters: true`) to the request. A caller can still pass an explicit `provider=` to override or disable it.

**Fingerprint logging.** Every response's `system_fingerprint` is captured on `JudgeResult`. `fingerprint_changed` flags an essay whose 7 per-trait calls did not all report the same fingerprint. `examples/audit_ellipse_llm.py`'s `regenerate()` also tracks the set of fingerprints seen across the whole run and prints a warning the first time a new one shows up mid-run, and persists `system_fingerprint` as a CSV column so it survives past the run that produced it.

**Ollama num_ctx: raised server-side, not per-request.** `options.num_ctx` is a parameter of Ollama's _native_ `/api/generate` and `/api/chat` endpoints. The OpenAI-compatible `/v1/chat/completions` endpoint this scorer uses does not accept it (verified against Ollama's own OpenAI-compatibility docs: the supported request fields are `model`, `messages`, `frequency_penalty`, `presence_penalty`, `response_format`, `seed`, `stop`, `stream`, `stream_options`, `temperature`, `top_p`, `max_tokens`, `tools`, `reasoning_effort`, `reasoning`; there is no `options` field). So the context window has to be raised server-side: the `OLLAMA_CONTEXT_LENGTH` env var on `ollama serve`, or a `PARAMETER num_ctx <n>` line in a custom Modelfile. `JudgeConfig` does not get a `num_ctx` field, because a field that cannot reach the endpoint it configures would look like a fix without being one. What the client side does instead: `/v1/chat/completions` returns `usage.prompt_tokens` (Ollama's OpenAI-compat equivalent of `prompt_eval_count`), and `JudgeResult.truncated` flags a value implausibly small for the essay's own length (`_CHARS_PER_TOKEN_ESTIMATE = 4`, a heuristic, not a real tokenizer) as a truncation tripwire.

`tests/test_ellipse_judge_scorer.py` pins all three: the provider auto-pin (present on an OpenRouter `base_url`, absent otherwise, overridable), `fingerprint_changed` set when per-trait calls disagree and left `False` when they agree, and `truncated` set when `prompt_tokens` is implausibly small for the essay and left `False` when it is plausible.

**Known limits of these pins, from the GPT-4o spot-check.** On an 80-essay per-trait GPT-4o run, `fingerprint_changed` fired on all 80 essays: `openai/gpt-4o` on the OpenAI-only route rotates through a fixed set of replica fingerprints keyed by trait position (a benign same-provider load-balance, not the Azure-vs-direct drift the provider pin targets). So the disqualifying signal is an Azure-origin call, not fingerprint variance in general, and the OpenAI-compatible response has no field naming the serving provider directly, only the fingerprint, so the two cannot be told apart from the client side alone. The truncation tripwire likewise false-positives on a large-context backend: on GPT-4o's 128k context the `chars/4` floor sits near the reported `prompt_tokens` for a normal essay, and real single-essay truncation is implausible. The heuristic is calibrated for Ollama's 4096-token ceiling and should not be read as a truncation signal on a large-context judge. Both flags stay (cheap to log, harmless), but neither is a hard gate on GPT-4o.

## Residual collapse after the per-trait fix: root cause and fix (2026-07-05)

The per-trait fix above removed the within-response halo but not the collapse. A per-trait GPT-4o rerun still assigned an identical score across all 7 traits on 43 of 80 essays (54%), fully isolated, one call per trait, with all three pins active and no Azure-routed call in the run. Isolating the calls dropped collapse from the batched design's 51/80 but did not eliminate it. This section settles the root cause and ships the fix.

**The collapse is scorer-side, not genuine essay quality (human-rater control, no API spend).** If a "2 across all 7 traits" collapse reflected genuinely uniform essay quality, a human rater scoring the identical essay on the identical rubric should collapse at a comparable rate. Using the human scores already on disk: `rater_1` collapsed on 11/80 essays (14%), `rater_2` on 10/80 (12%), against GPT-4o's 43/80 (54%), roughly 4x either human rater on the exact same essays. The collapse also does not enrich the way the "genuine quality" reading predicts: of the 43 GPT-4o-collapsed essays only 9 (21%) also had a human rater collapse, barely above the human baseline, and 86% of GPT-4o's collapses land on exactly `2`, not spread across the 1-5 scale the way independent per-trait judgments coinciding by essay quality would. Essay length (collapsed mean 2159 vs 2573 characters, point-biserial -0.20) and a human quality proxy (3.03 vs 3.45) both lean the expected way but far too weakly to explain a 4x gap. The read: a scorer-side default near the low-middle of the scale, not the essays deserving a uniform score.

**Root cause: a second halo channel in the prompt.** Reading the per-trait prompt builder settles which scorer-side mechanism is at work, before any further spend. Each of the 7 per-trait prompts shares 9 of its 12 lines: only the trait name, the one-line keyword rubric, and the JSON key differ. The shared block includes the scale anchors, and those anchors are holistic (360 characters describing whole-essay proficiency: "native-like facility," "errors that impede communication"), sent identically on every trait's call and longer than the trait-specific keyword line they sit next to. So even one trait at a time, the model reads the essay, forms one holistic impression, and the shared holistic anchors re-prime that same impression on all 7 calls. The batching fix closed the within-response halo; the shared holistic anchor is a second, cross-response halo channel. Two design choices carry it: the shared holistic anchors, and a bare-integer JSON response that gives the model no room to reach a trait-specific judgment before committing a score.

**The fix, confirmed by a paired ablation.** Two opt-in prompt changes address both channels: `trait_scoped_anchors` replaces the shared holistic anchors with per-trait anchors that name the trait at every scale level, and `reasoning` requires a one-sentence justification of the trait before the integer score. A paired ablation (2026-07-05) scored the same 40 essays (20 focal + 20 reference, seed 7) twice in one run, once under the default prompt and once under both flags, against `openai/gpt-4o` on OpenRouter with the provider pin active. Collapse dropped from 24/40 (60%) under the default prompt to 3/40 (8%) under the flags, below the human-rater baseline of 12-14% and materially below the default rate on the same essays. Because it is the same essays under both prompts, essay difficulty is controlled, so the gap is the prompt. The residual 8% (three essays, all scored `2`) is in-band with the human raters' own collapse rate and reads as genuinely uniform-weak essays, not a scorer artifact. The collapse was a prompt-design gap on this repo's end, diagnosable by inspecting the scorer's own prompt, not a limit of GPT-4o's ability to discriminate the rubric.

**Shipped.** The two knobs live on `build_messages` and `JudgeConfig` (`reasoning`, `trait_scoped_anchors`), both defaulting off so the committed qwen2.5:7b pilot CSV reproduces byte-for-byte, and are exposed on the regenerate CLI as `--reasoning` / `--trait-scoped-anchors`. `tests/test_ellipse_judge_scorer.py` pins the default path unchanged, the trait-scoped anchors naming the trait and differing across traits, the reasoning field preceding the score, and the parser ignoring the extra key; `tests/test_audit_ellipse_llm_cli.py` pins that the CLI flags reach the config.

**One question left open, deliberately.** The ablation changed both knobs at once, so it confirms the combined fix drops collapse to human-in-band levels but does not attribute how much each knob contributes. A one-knob-at-a-time ablation would isolate that; it is cheap but not needed to ship the fix, and is not blocking. With the collapse resolved, the reported-run and full-corpus GPT-4o spend (using the flags) is unblocked on this question; it remains gated only on the ~7x-token budget sign-off noted above. (Answered 2026-07-08; see the next section.)

## Knob attribution: requiring a justification before the score carries the fix (2026-07-08, live)

The question above is answered, with one caveat named below. A single-knob ablation ran each knob alone on the same 40 essays (20 focal + 20 reference, seed 7) and the same `openai/gpt-4o` route with the provider pin (`examples/ablation_knob_attribution.py`; raw scores in `examples/data/ablation_reasoning_only.csv` and `ablation_trait_scoped_anchors_only.csv`).

| Arm | Collapse rate | Wilson 95% CI |
| --- | --- | --- |
| Default, neither knob (2026-07-05 paired run) | 24/40 (60%) | [45, 74]% |
| Trait-scoped anchors only | 16/40 (40%) | [26, 55]% |
| Reasoning only | 2/40 (5%) | [1, 17]% |
| Both knobs (2026-07-05 paired run) | 3/40 (8%) | [3, 20]% |
| Human-rater baseline (2026-07-05 paired run) | 12-14% | n/a |

The reasoning knob does almost all the work. Requiring a one-sentence justification before the integer, with nothing else changed, drops collapse from 60% to 5% (2/40); the gap is large and significant against both the 60% default (Fisher exact p = 1.3e-7) and the 40% anchors-only arm (p = 3e-4). Trait-scoped anchors alone reach 40% (16/40), which is not distinguishable from the 60% default at this n (Fisher p = 0.12), so any anchor-only contribution is within noise here.

Two limits bound how far to read this.

- The reasoning arm confounds content with output order. It requires free text before the integer, so it shows that emitting text before the score breaks the collapse, but it cannot separate genuine per-trait reasoning from any buffer that defeats immediate anchoring. A placeholder-text control would isolate that and is not yet run. The honest mechanism is "commit the integer last, not first," not "the model reasons."
- The rates are single-run point estimates at n = 40 with wide intervals. Reasoning-only [1, 17]% overlaps both-knobs [3, 20]% and the 12-14% human band, so reasoning-only is indistinguishable from both-knobs and from the human raters (2 vs 3 collapses, Fisher p = 1.0), not below them. The 60% and 8% endpoints are the 2026-07-05 paired run, not re-run beside the single-knob arms, so the comparison assumes the route behaved the same across the two dates. `system_fingerprint` rotated on all seven per-trait calls of every essay in both arms; OpenRouter hides the serving provider, so a benign replica rotation cannot be told from a backend switch client-side. It is balanced across the two arms, so it does not explain the reasoning-versus-anchors gap, but it is an open caveat on the absolute rates.

This refines the root cause named above. The two candidates were the shared holistic anchors and the bare integer that commits a score with no room to reason first. The data puts the second first: the collapse tracks the model committing an integer before it differentiates the trait, and the anchor change alone does not measurably move it. The earlier reading named the shared anchors as the second halo channel; on this evidence the output-order channel leads and the anchor channel is not separable from noise.

Both knobs are recommended on for a GPT-4o-class judge, and both still default off in code so the committed qwen2.5:7b pilot reproduces byte for byte. The one-line rule for a practitioner: make the judge emit its per-trait justification before the integer, not after.

## References

- Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., and Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus. International Journal of Learner Corpus Research, 9(2), 248-269.
- Yamashita, T. (2025). Exploring potential biases in GPT-4o's ratings of English language learners' essays. Language Testing, 42(3), 344-358. https://doi.org/10.1177/02655322251329435
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349.
- `docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md` (conditioner-overlap diagnostic this run's `conditioner_group_corr` is read against).
- `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md` (the calibrated 0.2 safe-band threshold cited above).
