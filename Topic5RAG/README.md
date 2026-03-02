# Topic 5: RAG - Retrieval Augmented Generation

This folder contains my portfolio work for Topic 5.

## Table of contents

- `requirements.txt` - dependencies for Topic 5 scripts
- `download_topic5_resources.sh` - downloads notebook + corpora from course site
- `rag_core.py` - reusable RAG core (loading, chunking, retrieval, LLM calls)
- `run_topic5_exercises.py` - CLI runner for Exercises 1, 2, and 4-11
- `task3_frontier_manual_comparison.md` - manual template for Exercise 3
- `outputs/` - saved JSON/TXT outputs by exercise

## Setup

```bash
conda create -n topic5 -y python=3.12
conda activate topic5
pip install -r Topic5RAG/requirements.txt
```

Download resources (optional but recommended):

```bash
bash Topic5RAG/download_topic5_resources.sh
```

Notes:
- `Corpora.zip` is large (~229 MB).
- The scripts use text files (`.txt/.md/.text`) for corpus loading.

## Corpus path examples

Use a folder that directly contains text files (or subfolders with text files). Example:

```bash
--corpus-dir Topic5RAG/resources/Corpora/NewModelT
```

Congressional Record corpus example:

```bash
--corpus-dir Topic5RAG/resources/Corpora/Congressional_Record_Jan_2026/txt
```

## Ollama model example (Qwen 2.5 1.5B)

```bash
ollama pull qwen2.5:1.5b
```

Use `--open-provider ollama --open-model qwen2.5:1.5b` in exercise commands.

## Exercise commands

### Exercise 1: Open Model RAG vs No-RAG

Model T corpus:

```bash
python -u Topic5RAG/run_topic5_exercises.py ex1 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model qwen2.5:1.5b \
  --rag-k 30 \
  --max-context-chars 24000 \
  --output-dir Topic5RAG/outputs/ex1 \
  2>&1 | tee Topic5RAG/outputs/ex1/ex1_ModelT_terminal.txt
```

Congressional Record corpus:

```bash
python -u Topic5RAG/run_topic5_exercises.py ex1 \
  --corpus congress \
  --corpus-dir Topic5RAG/resources/Corpora/Congressional_Record_Jan_2026/txt \
  --open-provider ollama \
  --open-model qwen2.5:1.5b \
  --rag-k 30 \
  --max-context-chars 24000 \
  --output-dir Topic5RAG/outputs/ex1 \
  2>&1 | tee Topic5RAG/outputs/ex1/ex1_congress_terminal.txt
```

### Exercise 2: Open Model + RAG vs GPT-4o Mini

Model T corpus:

```bash
python -u Topic5RAG/run_topic5_exercises.py ex2 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model qwen2.5:1.5b \
  --large-provider openai \
  --large-model gpt-4o-mini \
  --rag-k 30 \
  --max-context-chars 24000 \
  --output-dir Topic5RAG/outputs/ex2 \
  2>&1 | tee Topic5RAG/outputs/ex2/ex2_ModelT_terminal.txt
```

Congressional Record corpus:

```bash
python -u Topic5RAG/run_topic5_exercises.py ex2 \
  --corpus congress \
  --corpus-dir Topic5RAG/resources/Corpora/Congressional_Record_Jan_2026/txt \
  --open-provider ollama \
  --open-model qwen2.5:1.5b \
  --large-provider openai \
  --large-model gpt-4o-mini \
  --rag-k 30 \
  --max-context-chars 24000 \
  --output-dir Topic5RAG/outputs/ex2 \
  2>&1 | tee Topic5RAG/outputs/ex2/ex2_congress_terminal.txt
```

### Exercise 3: Frontier model manual comparison

- Use `task3_frontier_manual_comparison.md` to record web UI comparisons.

### Exercise 4: Top-K retrieval effect

```bash
python -u Topic5RAG/run_topic5_exercises.py ex4 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model qwen2.5:1.5b \
  --k-values 1,3,5,10,20 \
  --output-dir Topic5RAG/outputs/ex4 \
  2>&1 | tee Topic5RAG/outputs/ex4/ex4_terminal.txt
```

### Exercise 5: Unanswerable questions

```bash
python -u Topic5RAG/run_topic5_exercises.py ex5 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --output-dir Topic5RAG/outputs/ex5 \
  2>&1 | tee Topic5RAG/outputs/ex5/ex5_terminal.txt
```

### Exercise 6: Query phrasing sensitivity

```bash
python -u Topic5RAG/run_topic5_exercises.py ex6 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --output-dir Topic5RAG/outputs/ex6 \
  2>&1 | tee Topic5RAG/outputs/ex6/ex6_terminal.txt
```

### Exercise 7: Chunk overlap experiment

```bash
python -u Topic5RAG/run_topic5_exercises.py ex7 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --overlap-values 0,64,128,256 \
  --output-dir Topic5RAG/outputs/ex7 \
  2>&1 | tee Topic5RAG/outputs/ex7/ex7_terminal.txt
```

### Exercise 8: Chunk size experiment

```bash
python -u Topic5RAG/run_topic5_exercises.py ex8 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --chunk-sizes 128,512,2048 \
  --output-dir Topic5RAG/outputs/ex8 \
  2>&1 | tee Topic5RAG/outputs/ex8/ex8_terminal.txt
```

### Exercise 9: Retrieval score analysis

```bash
python -u Topic5RAG/run_topic5_exercises.py ex9 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --output-dir Topic5RAG/outputs/ex9 \
  2>&1 | tee Topic5RAG/outputs/ex9/ex9_terminal.txt
```

### Exercise 10: Prompt template variations

```bash
python -u Topic5RAG/run_topic5_exercises.py ex10 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --output-dir Topic5RAG/outputs/ex10 \
  2>&1 | tee Topic5RAG/outputs/ex10/ex10_terminal.txt
```

### Exercise 11: Cross-document synthesis

```bash
python -u Topic5RAG/run_topic5_exercises.py ex11 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --k-values 3,5,10 \
  --output-dir Topic5RAG/outputs/ex11 \
  2>&1 | tee Topic5RAG/outputs/ex11/ex11_terminal.txt
```

## Discussion summary template

After running exercises, summarize:

1. Where no-RAG hallucinated and where RAG grounded answers.
2. Whether GPT-4o Mini no-RAG outperformed local small model + RAG on each query type.
3. Which `k` gave best tradeoff between completeness and noise.
4. Whether strict grounding prompt reduced hallucinations on unanswerable questions.
5. Which query phrasings retrieved the best chunks.
6. Best chunk size/overlap settings for this corpus.
7. Suggested similarity score threshold from Exercise 9.
8. Which prompt template was most accurate/useful.
9. How well cross-document synthesis worked at `k=3/5/10`.

## Exercise 1 direct answers

Based on latest runs:
- `outputs/ex1/ex1_results_20260301_223727.json` (Model T corpus)
- `outputs/ex1/ex1_results_20260301_224533.json` (Congressional Record corpus)

### Does the model hallucinate specific values without RAG?

Yes. In these results, no-RAG either refused to answer or generated unsupported details.
Examples:
- Model T spark plug gap: no-RAG gave a specific numeric range that was not grounded in retrieved corpus text.
- Congressional query about Elise Stefanik: no-RAG answered with an unrelated TV-show narrative.

### Does RAG ground the answers in the actual manual?

Partially. RAG was generally more on-topic and connected to retrieved chunks, but it still produced incorrect specifics on some questions.
Examples:
- Better grounding: Model T oil question returned "light high grade engine oil," which matches corpus wording.
- Remaining errors: Congressional answers sometimes added incorrect names/details not supported by top retrieved text.

### Are there questions where the model's general knowledge is actually correct?

Not clearly in this specific Exercise 1 run. Most no-RAG responses were refusals or low-confidence/hallucinated content, so we did not observe a strong "general knowledge wins" case here.

## Exercise 2 direct answers

Based on latest runs:
- `outputs/ex2/ex2_results_20260301_230009.json` (Model T corpus)
- `outputs/ex2/ex2_results_20260301_230841.json` (Congressional Record corpus)

### Does GPT-4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?

Mixed, but on time-sensitive Congressional Record questions GPT-4o Mini avoided hallucinations better.
- For January 2026 event questions, GPT-4o Mini explicitly refused due its October 2023 cutoff instead of inventing details.
- Qwen 2.5 1.5B + RAG answered those same questions but included several incorrect specifics (for example, wrong details for the Elise Stefanik question).
- However, GPT-4o Mini still hallucinated on some non-date-specific questions (for example, Main Street Parity Act explanation and pregnancy-center speaker names).

### Which questions does GPT-4o Mini answer correctly?

From these runs, none of the GPT-4o Mini answers can be confidently marked as fully correct and corpus-grounded.
- Correct handling (abstention, not answering): the two January 2026 Congressional questions were declined because they are beyond its stated training cutoff.
- The remaining responses were either generic/unsupported (Model T) or likely incorrect for this specific corpus context (Congress policy/speaker questions).

### Compare GPT-4o Mini pretraining cut-off with corpus age

In this run, GPT-4o Mini stated a knowledge cutoff of **October 2023**.
- Congressional Record corpus: January 2026 documents, about **2 years and 3 months newer** than cutoff.
- Model T corpus: historical material from the early 1900s (Model T era), roughly **a century older** than cutoff.

Implication: age alone does not guarantee correctness. Very old topics (Model T) can still produce hallucinations without retrieval, while post-cutoff events (Congress 2026) require retrieval/current sources to answer accurately.

## Exercise 4 direct answers

Based on latest run:
- `outputs/ex4/ex4_results_20260301_235004.json`

### At what point does adding more context stop helping?

In this run, gains mostly stopped after about `k=3` to `k=5`.
- `k=1 -> k=3` sometimes helped (for example, carburetor query moved from abstention to a relevant excerpt).
- Beyond that, quality did not improve consistently, and often degraded.
- Latency increased sharply after `k=5` (mean response time): `k=1 ~11.11s`, `k=3 ~20.70s`, `k=5 ~21.68s`, `k=10 ~69.74s`, `k=20 ~87.98s`.

### When does too much context hurt (irrelevant information, confusion)?

In this run, harm was clear by `k=10` and worse at `k=20`.
- Spark plug gap answers drifted into incorrect ring-gap values and inconsistent numbers.
- Oil query changed from abstention at low `k` to incorrect/irrelevant answers at higher `k`.
- This indicates extra retrieved text introduced noise and confused grounding rather than helping.

### How does k interact with chunk size?

For this run, chunk size was fixed at `512` (overlap `128`), so interaction is inferred:
- Higher `k` linearly increased context volume (`k=5` around ~2.9k chars, `k=10` around ~5.8k, `k=20` around ~11.6k), increasing noise risk.
- With larger chunks, this problem usually appears earlier (each chunk carries more mixed/irrelevant content).
- With smaller chunks, you can often use higher `k` before confusion, but may need more chunks to preserve full procedural context.
- Use Exercise 8 to measure this directly for your corpus.

## Exercise 5 direct answers

Based on latest run:
- `outputs/ex5/ex5_results_20260301_235839.json`

### Does the model admit it doesn't know?

Yes, under `strict_grounding` it consistently admits uncertainty:
- All three unanswerable queries returned: `I cannot answer this from the available documents.`

Under `permissive`, it is inconsistent:
- For `capital of France`, it first notes missing context, then answers from general knowledge (`Paris`).
- For the other two questions, it does not reliably admit uncertainty.

### Does it hallucinate plausible-sounding but wrong answers?

Yes, especially with the `permissive` prompt variant.
- `horsepower of a 1925 Model T`: gave an irrelevant answer about Model N/service chapter text instead of horsepower.
- `synthetic oil` question: generated a detailed modern-style explanation even though the manual context did not support it.

### Does retrieved context help or hurt? (Does irrelevant context encourage hallucination?)

Both, depending on prompt behavior:
- Helps with `strict_grounding`: retrieved context is used as a guardrail, leading to abstention instead of fabricated answers.
- Can hurt with `permissive`: irrelevant retrieved chunks appear to encourage plausible but unsupported explanations and false confidence.

## Exercise 6 direct answers

Based on latest run:
- `outputs/ex6/ex6_results_20260302_000153.json`

### Which phrasings retrieve the best chunks?

In this run, no phrasing consistently retrieved clearly "best" maintenance-schedule chunks; retrieval was unstable across paraphrases.
- By top-1 similarity score, ranking was:
  - `How often should I service the engine?` (`0.1559`)
  - `When do I need to check the engine?` (`0.1525`)
  - `What is the recommended maintenance schedule for the engine?` (`0.1343`)
  - `Preventive maintenance requirements` (`0.1313`)
  - `engine maintenance intervals` (`0.1239`)
- However, high score did not always mean better relevance. Many top chunks were still off-target (procedures, parts, or service-admin text rather than a clean maintenance schedule).

### Do keyword-style queries work better or worse than natural questions?

Worse in this run.
- The keyword-style query `engine maintenance intervals` had the lowest top-1 score and did not improve chunk quality.
- Natural-language queries generally scored higher, but still suffered relevance drift.

### What does this tell you about potential query rewriting strategies?

This corpus likely needs intent-aware rewriting, not just shorter keywords.
- Add domain anchors from corpus language (for example: `inspection intervals`, `time study`, `service schedule`, `check every`, `lubrication`).
- Use multi-query expansion (2-3 rewrites) and fuse results instead of relying on one phrasing.
- Prefer constrained rewrites that include both object and action (for example: `engine + periodic inspection + interval`) to reduce semantic drift.
- Add a lightweight re-ranker or section filter so top chunks favor maintenance-schedule content over generic procedural noise.

## Exercise 7 direct answers

Based on latest run:
- `outputs/ex7/ex7_results_20260302_000535.json`

### Does higher overlap improve retrieval of complete information?

Yes, up to a point.
- `overlap=0` top chunk starts mid-context and is less complete.
- `overlap=64` and `overlap=128` retrieve cleaner "step-by-step procedure" language around section boundaries.
- `overlap=256` does not add much new completeness versus `128`; it mostly duplicates neighboring context.

### What's the cost? (Index size, redundant information in context)

Cost rises quickly as overlap increases.
- Chunk count growth vs `overlap=0`:
  - `64`: `1020` chunks (`+14.2%`)
  - `128`: `1190` chunks (`+33.3%`)
  - `256`: `1785` chunks (`+99.9%`, almost 2x index size)
- Redundancy in top retrieved chunks increased strongly at high overlap (top1/top2 text similarity):
  - `0`: `0.021`
  - `64`: `0.043`
  - `128`: `0.250`
  - `256`: `0.500`

### Is there a point of diminishing returns?

Yes. In this run, diminishing returns appear after about `overlap=128`.
- Retrieval quality signal (`sum(top5 scores)`) improved from `0 -> 64 -> 128`, but only marginally from `128 -> 256`.
- The `256` setting roughly doubled index size versus `0` while adding mostly redundant context.
- Practical choice from this run: `overlap=64` or `128` is a better tradeoff than `256`.

## Exercise 8 direct answers

Based on latest run:
- `outputs/ex8/ex8_results_20260302_001056.json`

### How does chunk size affect retrieval precision (relevant vs. irrelevant content)?

Smaller chunks were more precise in this run.
- Mean top-1 similarity score by chunk size:
  - `128`: `0.2661`
  - `512`: `0.1934`
  - `2048`: `0.1244`
- A simple noise check on top-1 chunks (`CHAPTER/PAGE/table-of-contents-like text`) also worsened as size increased:
  - `128`: `0/4` noisy top-1 chunks
  - `512`: `2/4`
  - `2048`: `3/4`

### How does it affect answer completeness?

There is a precision-completeness tradeoff:
- `128` gives focused snippets but can be fragmented (good local evidence, less full procedure context per chunk).
- `2048` carries more context per chunk but often mixes in unrelated material, reducing usable completeness.
- `512` is a middle ground, but in this corpus/run it still picked noisy top chunks frequently.

### Is there a sweet spot for your corpus?

For this run, `128` looked best for retrieval quality.
- It had the highest top-1 scores and the cleanest top results.
- If later generation quality appears too fragmented, test a nearby range (for example `192` or `256`) rather than jumping to `2048`.

### Does optimal size depend on the type of question?

Yes.
- Fact lookups (for example, a specific setting/value) tend to benefit from smaller chunks (`128`) because precision dominates.
- Multi-step procedural questions may need slightly larger chunks or more retrieved chunks (`k`) to preserve continuity.
- Practical strategy: choose chunk size by query class, or use hybrid retrieval (small + medium chunks with reranking).

## Exercise 9 direct answers

Based on latest run:
- `outputs/ex9/ex9_results_20260302_001412.json`

### When is there a clear "winner" (large gap between #1 and #2)?

A clear winner appeared when `winner_gap_1_2` was relatively large (around `>= 0.03` in this run).
- Clear case: `What is the correct spark plug gap for a Model T Ford?` with gap `0.0809`.
- Moderate case: `How do I fix a slipping transmission band?` with gap `0.0355`.

Most other queries had small gaps (`~0.0017` to `~0.0187`), indicating no dominant chunk.

### When are scores tightly clustered (ambiguous)?

Scores were tightly clustered for most maintenance/oil/carburetor phrasings.
- Examples:
  - `What is the recommended maintenance schedule for the engine?` gap `0.0017`
  - `What oil should I use in a Model T engine?` gap `0.0032`
  - `Preventive maintenance requirements` gap `0.0036`

These distributions suggest ambiguity: multiple chunks look similarly relevant to the retriever.

### What score threshold would you use to filter out irrelevant results?

From this run, a practical starting threshold is about `0.11`.
- Exercise output suggested: `0.1133` (mean top-5 score).
- Suggested policy:
  - Keep chunks `>= 0.11` by default.
  - If too few chunks remain, back off slightly (for example to `0.10`) to preserve recall.

### How does score distribution correlate with answer quality?

Correlation is partial, not absolute.
- Better pattern: larger top gap and stronger top scores tended to align with more focused retrieval (for example transmission query looked more coherent than maintenance-schedule queries).
- Ambiguous pattern: tightly clustered scores tended to align with noisier/less focused retrieval (seen in maintenance-schedule style queries).
- Important caveat: high score/gap does **not** guarantee correctness. In this corpus, spark-plug retrieval had a clear winner but still led to wrong/confused downstream answers due to imperfect source text and nearby distractor content.

## Exercise 10 direct answers

Based on latest run:
- `outputs/ex10/ex10_results_20260302_003400.json`

### Which prompt produces the most accurate answers?

For this run, `strict_grounding` was the most reliable for avoiding wrong claims, but it was over-conservative.
- It abstained in all 4/4 questions (`I cannot answer...` or equivalent), which reduced hallucinations.
- Other prompts (`permissive`, `structured`, and sometimes `citation`) produced more detailed but often incorrect or weakly grounded answers.

So, if "accurate" means "do not make unsupported claims," `strict_grounding` performed best.

### Which produces the most useful answers?

`citation` was the most useful compromise in this run.
- It sometimes provided context-linked guidance while still admitting missing evidence on some questions.
- `permissive`/`structured` were more verbose, but usefulness was reduced by incorrect details and weak grounding.
- `strict_grounding` was safest but often not useful for completing the task because of frequent abstention.

### Is there a trade-off between strict grounding and helpfulness?

Yes, clearly in this run.
- More strict grounding => fewer hallucinations, more abstentions.
- More permissive prompting => more "helpful-looking" detail, but higher hallucination risk.

Practical takeaway: use strict grounding for high-stakes factual fidelity; use a citation-style prompt when you need a better balance between groundedness and task usefulness.

## Exercise 11 direct answers

Based on latest run:
- `outputs/ex11/ex11_results_20260302_004131.json`

Note: this run used `NewModelT` with `loaded_docs=1`, so this is effectively **cross-chunk synthesis** (not multi-file cross-document synthesis).

### Can the model successfully combine information from multiple chunks?

Partially.
- At higher `k` (especially `k=5` and `k=10`), the model did combine details from multiple retrieved chunks.
- But synthesis quality was inconsistent: combined outputs often mixed relevant and irrelevant details (for example, monthly tasks mixed with service-office workflow/admin text).

### Does it miss information that wasn't retrieved?

Yes.
- Answers tracked whatever was in top retrieved chunks.
- When retrieval did not surface the right sections, the model either missed key content or filled with loosely related material.
- Example pattern in this run: "tune-up tools" stayed tied to retrieved radiator/shop-tool chunks, not a complete tune-up-specific checklist.

### Does contradictory information in different chunks cause problems?

Yes, it can.
- The safety-warning query showed unstable behavior across `k` (`k=5` said no explicit warnings, while `k=10` produced multiple warnings including questionable combinations), suggesting chunk-to-chunk mixing/conflict.
- Higher `k` increased the chance of pulling heterogeneous context, which made synthesis less stable and more error-prone.

Practical takeaway: for synthesis tasks, retrieval quality and chunk curation matter more than simply increasing `k`.
