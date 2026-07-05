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

GPT-4o via OpenRouter is available as an optional hosted spot-check to replicate Yamashita's exact model (`--base-url https://openrouter.ai/api/v1 --model openai/gpt-4o`, API key in `ELLIPSE_JUDGE_API_KEY`).

## Halo/anchoring bug found before any full-corpus spend, and the fix

Before running GPT-4o on the full corpus, an 80-essay ad hoc GPT-4o spot-check and a 179-essay llama-3.3-70b spot-check were run against the batched scorer described above (one chat-completion call per essay, asking for all 7 rubric traits in a single JSON response). Neither trial CSV was committed; both are discarded by this finding.

**Finding.** 51 of the 80 GPT-4o rows had an identical score repeated across all 7 traits (mostly all-2s); the llama-3.3-70b trial showed the same pattern on only 1 of 179 rows. A direct comparison on one essay, same essay and decoding params throughout, confirmed the cause: the batched call (all 7 traits in one JSON request) returned Overall=2 and every other trait=2; a call asking for Overall alone returned Overall=3; a call asking for Grammar alone returned Grammar=2. Response `finish_reason` was `stop` in all cases (46 completion tokens on the batched call), ruling out truncation. Asking GPT-4o to score 7 dimensions in one response induces a halo effect: it forms one holistic impression and repeats it, rather than differentiating per trait. This is not a GPT-4o identity problem; the response metadata confirms genuine `openai/gpt-4o` (`provider: OpenAI` on most calls, one call routed to Azure by OpenRouter's load balancing, with three different `system_fingerprint`s observed across the three calls: `fp_683410201e`, `fp_f38e8ce955`, `fp_85f0f4d955`). It is a scorer-prompt design flaw, and because qwen2.5:7b and llama-3.3-70b were scored with the same batched design, the effect size on Vocabulary/Syntax/Overall reported above under "Pilot result" is a scoring artifact risk, not yet a clean measurement, even though qwen2.5:7b haloed far less often than GPT-4o did in this spot-check.

**Fix.** `examples/_ellipse_judge.py` now makes one chat-completion call per trait (7 isolated calls per essay) instead of one call for all 7. Each call's prompt carries only that trait's keyword rubric, never the other 6, so there is nothing left for the model to anchor across within a single response. `JudgeResult` keeps its existing all-or-nothing contract: `scores` is the full 7-trait dict or `None` (a trait that exhausts its retries stops the essay there, never a partially-filled result). `tests/test_ellipse_judge_scorer.py` pins this: one isolated call per trait, no leakage of other traits' rubric text into a single-trait prompt, and fail-closed (never partial) aggregation.

**Cost note.** Isolating traits means ~7x the calls per essay, and the essay text is repeated in every call rather than sent once, so a full-corpus GPT-4o run costs roughly 7x the token volume of the original low-hundreds-of-dollars estimate; get budget sign-off against the revised estimate, not the batched-design one, before running the full 6,462-essay corpus. Re-running the 80-essay spot-check with the per-trait scorer (to confirm the collapse is gone across the whole sample, not just the one live comparison) is inexpensive by comparison.

## Reproducibility protocol per backend

Determinism knobs exist on every backend this scorer talks to, but none of them alone guarantees a byte-identical rerun; each has a distinct failure mode worth pinning against before treating a score CSV as reproducible from source.

- **OpenRouter.** `openai/gpt-4o` on OpenRouter load-balances across providers (OpenAI direct and Azure-hosted OpenAI were both observed serving the same model slug in the spot-check above, each returning a different `system_fingerprint`). Pin the provider with the request's `provider` object: `order: ["openai"]` plus `allow_fallbacks: false` restricts routing to the OpenAI-direct endpoint only (`only: ["openai"]` is the equivalent allow-list form). Add `require_parameters: true` so the request fails closed if a routed provider would silently drop an unsupported parameter (notably `seed`) rather than accept it and ignore it. `seed` is honored on a best-effort basis per-provider on OpenRouter, same caveat as the direct OpenAI API below.
- **OpenAI (direct, or the OpenAI-only route above).** `seed` plus identical other parameters gives "mostly deterministic" output, per OpenAI's own documentation; the `system_fingerprint` response field is the tripwire; log it alongside every scored row and treat a run where it changes mid-run as non-comparable rather than averaging it in silently.
- **Ollama.** The default context window is 4096 tokens (`OLLAMA_CONTEXT_LENGTH` env var, or `options.num_ctx` per-request against the native `/api/generate` and `/api/chat` endpoints), independent of `temperature`/`seed`. A long essay plus the per-trait prompt overhead can approach that ceiling and silently truncate the input rather than error.

## Pins implemented

The three pins above are now in `examples/_ellipse_judge.py`, with one correction found while implementing the Ollama pin against the actual API docs rather than assuming the plan above would just work.

**OpenRouter provider pin.** `JudgeConfig.provider` defaults to `None`; when `base_url` is OpenRouter and no override was given, `_build_payload` auto-attaches `DEFAULT_OPENROUTER_PROVIDER` (`order: ["openai"]`, `allow_fallbacks: false`, `require_parameters: true`) to the request, matching the plan above. A caller can still pass an explicit `provider=` to override or disable it.

**Fingerprint logging.** Every response's `system_fingerprint` is captured on `JudgeResult`. `fingerprint_changed` flags an essay whose 7 per-trait calls did not all report the same fingerprint (the exact pattern the spot-check above caught: three different fingerprints across three calls on one essay). `examples/audit_ellipse_llm.py`'s `regenerate()` also tracks the set of fingerprints seen across the whole run and prints a warning the first time a new one shows up mid-run, and persists `system_fingerprint` as a CSV column so it survives past the run that produced it.

**Ollama num_ctx: the plan above was wrong, corrected here.** `options.num_ctx` is a parameter of Ollama's _native_ `/api/generate` and `/api/chat` endpoints. Checked against Ollama's own OpenAI-compatibility docs (`docs/api/openai-compatibility.mdx`): the supported request fields for `/v1/chat/completions` are `model`, `messages`, `frequency_penalty`, `presence_penalty`, `response_format`, `seed`, `stop`, `stream`, `stream_options`, `temperature`, `top_p`, `max_tokens`, `tools`, `reasoning_effort`, `reasoning`. There is no `options` field, so a `num_ctx` sent in this scorer's request body would just be dropped: not the silent truncation the plan above worried about, but not a fix either. Since `_ellipse_judge.py` only ever talks to the OpenAI-compatible endpoint (by design, so the same scorer works against any OpenAI-compatible backend), the context window has to be raised server-side instead: the `OLLAMA_CONTEXT_LENGTH` env var on `ollama serve`, or a `PARAMETER num_ctx <n>` line in a custom Modelfile for the model in use. `JudgeConfig` does not get a `num_ctx` field, because a field that cannot reach the endpoint it configures would be worse than no field: it would look like a fix without being one.

What the client side CAN do, and now does: `/v1/chat/completions` does return `usage.prompt_tokens` (confirmed in the same docs), Ollama's OpenAI-compatible equivalent of `prompt_eval_count`. `JudgeResult.truncated` compares the last `prompt_tokens` seen for an essay against a crude chars-per-token floor on the essay's own text length (`_CHARS_PER_TOKEN_ESTIMATE = 4`, a heuristic, not a real tokenizer) and flags it if the reported token count is implausibly small for what was sent. `regenerate()` prints a warning and persists `prompt_tokens` and `truncated` as CSV columns. This is a tripwire, not a fix: an Ollama run against a corpus with essays long enough to risk the 4096-token default still needs `OLLAMA_CONTEXT_LENGTH` raised before that run, not after.

`tests/test_ellipse_judge_scorer.py` pins all three: the provider auto-pin (present on an OpenRouter `base_url`, absent otherwise, overridable), `fingerprint_changed` set when per-trait calls disagree and left `False` when they agree, and `truncated` set when `prompt_tokens` is implausibly small for the essay and left `False` when it is plausible.

## Re-trial: 80-essay GPT-4o spot-check with the pins live, and a correction to the "Fix" claim above

An ad hoc, disposable rerun of the 80-essay spot-check (balanced 40 focal + 40 reference, seed 1, not the committed pilot sample), against `openai/gpt-4o` on OpenRouter with the per-trait scorer and all three pins active. Not committed: same pattern as the original discarded 80-essay trial above.

**Parsing and routing held up cleanly.** Zero parse failures across 560 calls (80 essays x 7 traits). The run-wide set of `system_fingerprint` values seen across all 80 essays was a single value, so the OpenAI-only provider pin did what it was built for: no Azure-routed call turned up anywhere in this run.

**The collapse rate did not go away, and the "Fix" section above overclaimed.** 43 of 80 essays (54%) still had all 7 traits land on the exact same score, fully isolated, one call per trait, against the real per-trait scorer, not the manual one-off comparison the original fix was based on. That single-essay comparison (Overall=3 alone vs Overall=2 batched) was real, but it was one essay, and it is not representative of the base rate: isolating the calls reduces collapse from the batched design's 51/80 but does not eliminate it.

**Follow-up diagnostic separated two competing explanations.** Reran 3 of the collapsed essays (one all-2s, one all-3s, one all-1s) as fresh, independent calls in a separate process, printing every trait's raw response. All 3 reproduced their exact prior scores. The `system_fingerprint` sequence across each essay's 7 calls turned out to be a fixed, trait-position-keyed rotation through 5 values (`Overall`/`Conventions` -> `fp_8772b5f549`, `Cohesion`/`Grammar` -> `fp_a77e540149`, `Syntax` -> `fp_683410201e`, `Vocabulary` -> `fp_ccaab42819`, `Phraseology` -> `fp_4fa7959453`), identical across all 3 essays and across the two separate runs. That is ordinary replica load-balancing inside the OpenAI-only route, not the Azure-vs-direct drift the provider pin targets, and not a source of leakage between calls (each call is a fresh, stateless request with only its own trait's rubric in the prompt; `tests/test_ellipse_judge_scorer.py::test_build_messages_isolates_the_single_trait` already pins that no other trait's text reaches the prompt).

Since the isolated per-trait design rules out cross-call leakage architecturally, and the collapsed scores reproduce identically across independently issued requests, this is not the batching-anchoring bug recurring. **The tentative reading offered here in the first pass of this section, that the collapse reflects genuine correlated essay quality, does not survive an adversarial check and is retracted below.**

**The `fingerprint_changed` flag is too blunt to use as written.** It fired on 80/80 essays in the spot-check, driven by the same-provider replica rotation just described, not by provider drift. Treating every one of those as "non-comparable" per the reproducibility-protocol wording above would discard the whole dataset. The flag stays (it is cheap, harmless to log, and would still catch a real Azure leak if `allow_fallbacks`/`require_parameters` ever failed), but the decision rule needs to be: an Azure-origin call is the disqualifying signal, not fingerprint variance in general, and there is no field in the OpenAI-compatible response that names the serving provider directly, only the fingerprint, so telling the two apart from the client side alone is not solved here.

**The truncation tripwire false-positived on all 3 flagged essays.** Their raw lengths (5105, 4152, 5358 characters) put the crude `chars/4` floor near or above the actual `prompt_tokens` GPT-4o reported, and GPT-4o's 128k context makes real truncation on a single essay implausible. The heuristic was written against Ollama's 4096-token ceiling; it is not calibrated for a large-context backend and should not be trusted as a truncation signal there. It is left as-is rather than backend-conditioned, because a backend-aware threshold is speculative complexity for a check whose only real job, so far, is Ollama.

## Adversarial check: is the collapse a scorer artifact, or genuine essay quality? The human-rater control says artifact

The "genuine correlated quality" reading above was a hypothesis from 3 essays, not a finding. Testing it against the full 80-essay sample with no new API spend, using data already on disk, kills it.

**Human raters scoring the SAME 80 essays on the SAME 7 traits do not collapse at anywhere near GPT-4o's rate.** If a "2 across all 7 traits" collapse reflected genuine uniform essay quality, a human rater scoring the identical essay on the identical rubric should collapse at a comparable rate. Instead: `rater_1` collapsed (identical score on all 7 traits) on 11/80 essays (14%), `rater_2` on 10/80 (12%), against GPT-4o's 43/80 (54%). GPT-4o collapses roughly 4x more often than either human rater on the exact same essays.

**No enrichment in the direction the "genuine quality" hypothesis predicts.** Of the 43 GPT-4o-collapsed essays, only 9 (21%) also had a human rater collapse, barely above the human baseline rate, and of the 37 GPT-4o-non-collapsed essays, 11 (30%) had a human rater collapse, if anything the opposite direction. If GPT-4o's collapses were essays that are just genuinely uniform, they should correlate with human collapses far more than this (n is small, 43 vs 37, so treat this as suggestive rather than a powered test, but the direction is the wrong one for the retracted hypothesis, not a null result that merely failed to reach significance).

**The collapsed score itself is suspicious.** 37 of the 43 collapses (86%) land on exactly `2`, not spread across the 1-5 scale the way independent, genuine per-trait judgments landing on the same value by essay-quality coincidence would be. Essay length (collapsed mean 2159 vs non-collapsed 2573 characters, point-biserial correlation -0.20) and a human-rated quality proxy (collapsed mean 3.03 vs non-collapsed 3.45, on the 14-column all-trait, both-rater mean) both point the same direction, weakly, but neither gap is large enough to explain a 4x difference in collapse rate against the human baseline on the same essays.

**Reading:** this looks like a scorer-side default or heuristic shortcut on GPT-4o's part, concentrated near the low-middle of the scale, not the essays independently deserving a uniform score. It does not look like an OpenRouter-routing artifact: the per-essay diagnostic already showed stateless, leakage-free, reproducible isolated calls, and the fingerprint analysis above shows the provider pin holding (no Azure). The two live candidates left are (a) a genuine limitation of GPT-4o's zero-shot per-trait discrimination on this specific keyword-rubric prompt, most visible on essays that read as uniformly weak, or (b) a prompt-design gap on this repo's end: asking for a bare JSON integer with no reasoning step gives the model no room to work out a trait-specific judgment before answering, so it falls back to a fast holistic guess. These have different fixes (accept the limitation and report it vs. add a short chain-of-thought or few-shot step before the score) and are not yet distinguished by anything run so far.

**What would distinguish them, not yet run (real API spend, needs sign-off before running):**

1. A same-essay ablation: rerun a subsample with a one-sentence justification step required before the JSON score (still isolated per trait, still fails closed on unparseable output). If collapse drops sharply, it is a prompt-design gap on this repo's end. If it does not move, GPT-4o plausibly cannot discriminate this rubric well on essays like these regardless of prompting.
2. A same-essay OpenRouter-vs-direct-OpenAI control: score the same subsample against the direct OpenAI API (not OpenRouter) to rule out any residual OpenRouter-layer effect the fingerprint/routing checks above did not already cover. Lower priority than (1) given the fingerprint evidence already collected, but cheap to run alongside it.

**Before any full-corpus GPT-4o run:** neither ablation above has been run. Until one of them narrows this to "prompt gap" or "model limitation," the pilot's Vocabulary/Syntax/Overall DIF numbers should be read as provisional, not a clean measurement, and the 6,462-essay corpus spend should wait on this, not just on the collapse rate being "acceptable."

## References

- Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., and Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus. International Journal of Learner Corpus Research, 9(2), 248-269.
- Yamashita, T. (2025). Exploring potential biases in GPT-4o's ratings of English language learners' essays. Language Testing, 42(3), 344-358. https://doi.org/10.1177/02655322251329435
- Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349.
- `docs/decisions/2026-07-01-e07-dif-nested-strata-confound.md` (conditioner-overlap diagnostic this run's `conditioner_group_corr` is read against).
- `docs/decisions/2026-07-02-e07-overlap-threshold-calibration.md` (the calibrated 0.2 safe-band threshold cited above).
