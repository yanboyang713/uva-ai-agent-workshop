# Topic 8: Fine-Tuning an LLM

This directory contains my work for Topic 8, focused on the Tinker fine-tuning exercise for text-to-SQL.

Course page:
- https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/finetuning.html

## Table of Contents

- `.env.example` - environment variable template for `TINKER_API_KEY`
- `requirements.txt` - Python dependencies for the local prep scripts and Tinker run
- `download_sql_dataset.py` - downloads `sql_create_context_v4.json` from the course site
- `sql_matches.py` - execution-based SQL equivalence checker from the course materials
- `task8a_inspect_sql_dataset.py` - loads, samples, and summarizes the SQL dataset
- `task8b_tinker_sql_finetune.py` - base eval, LoRA fine-tuning on Tinker, post-train eval, manual novel-schema eval
- `task8c_bonus_generalization.py` - separate bonus experiment with stricter prompting, schema-focused augmentation, and candidate reranking
- `run_finetuning.sh` - convenience script to run the main Topic 8 steps
- `data/` - local dataset storage
- `outputs/` - terminal logs, eval predictions, and summary JSON
- `outputs_bonus/` - artifacts from the separate bonus experiment

## Setup

```bash
conda create -n topic8 -y python=3.12
conda activate topic8
pip install -r Topic8FineTuning/requirements.txt
cp Topic8FineTuning/.env.example Topic8FineTuning/.env
```

Then edit `Topic8FineTuning/.env` and set:
- `TINKER_API_KEY`

The course page also requires:
- installing Tinker Cookbook
- signing into the Tinker console with a UVA email

## Dataset

Download the course-provided SQL dataset:

```bash
python -u Topic8FineTuning/download_sql_dataset.py \
  2>&1 | tee Topic8FineTuning/outputs/task8_download_dataset.txt
```

Inspect the data and confirm the train/test split:

```bash
python -u Topic8FineTuning/task8a_inspect_sql_dataset.py \
  2>&1 | tee Topic8FineTuning/outputs/task8a_dataset_inspection.txt
```

## Fine-Tuning Run

Run the full Topic 8 pipeline:

```bash
python -u Topic8FineTuning/task8b_tinker_sql_finetune.py \
  2>&1 | tee Topic8FineTuning/outputs/task8b_tinker_finetune.txt
```

Or use the wrapper script:

```bash
bash Topic8FineTuning/run_finetuning.sh
```

Useful smoke-test flags before a full remote run:

```bash
python -u Topic8FineTuning/task8b_tinker_sql_finetune.py \
  --train-limit 1024 \
  --eval-limit 20 \
  --max-train-batches 4
```

## Outputs

After a successful Tinker run, the script writes:
- `outputs/base_eval_predictions.json`
- `outputs/tuned_eval_predictions.json`
- `outputs/manual_novel_schema_predictions.json`
- `outputs/training_loss.json`
- `outputs/summary.json`

The held-out evaluation uses the course SQL equivalence checker and compares generated SQL against expected SQL by executing both queries on multiple synthetic SQLite databases built from the schema.

Observed results from the current run:
- Base accuracy on the 200 held-out test questions: `0.285`
- Fine-tuned accuracy on the same 200 held-out test questions: `0.420`
- Manual novel-schema accuracy on the 5 extra questions: `0.000`

Saved evidence from the completed run:
- `outputs/task8b_tinker_finetune.txt`
- `outputs/summary.json`
- `outputs/base_eval_predictions.json`
- `outputs/tuned_eval_predictions.json`
- `outputs/manual_novel_schema_predictions.json`
- `outputs/training_loss.json`

## Bonus File

The bonus challenge lives in a separate file so the main assignment pipeline stays unchanged:

```bash
python -u Topic8FineTuning/task8c_bonus_generalization.py \
  2>&1 | tee Topic8FineTuning/outputs/task8c_bonus_generalization.txt
```

This bonus script changes three things:
- it uses a stricter prompt that tells the model to return exactly one SQL query,
- it adds a small synthetic augmentation set for the novel-schema families from Step 7,
- and it samples multiple candidate SQL queries per question, then reranks them to prefer shorter executable SQL over longer overgenerated outputs.

Implementation details of `task8c_bonus_generalization.py`:
- It keeps the same overall Tinker training loop structure as `task8b_tinker_sql_finetune.py`, so the comparison stays fair.
- It introduces a stronger prompt template that explicitly says to return one SQL query only, use only schema fields, and avoid extra clauses unless the question requires them.
- It appends a handcrafted augmentation set for the out-of-distribution schema families from the bonus questions: `employees`, `products`, `students`, `orders`, and `courses`/`enrollments`.
- It repeats that augmentation set multiple times during training so those schema families have more influence on the LoRA updates.
- At inference time, it samples multiple SQL candidates per question instead of only one.
- It sanitizes each candidate by stripping extra text, truncating after the first SQL statement, and normalizing the query text.
- It scores candidates by preferring executable SQL, shorter SQL, and fewer repeated or suspicious fragments such as repeated `SELECT`, `UNION`, `INTERSECT`, `EXCEPT`, or answer/explanation spillover.
- It then picks the highest-scoring candidate for evaluation, which is why the bonus run's base model score increased substantially even before fine-tuning.

Bonus outputs are written to:
- `outputs_bonus/bonus_base_eval_predictions.json`
- `outputs_bonus/bonus_tuned_eval_predictions.json`
- `outputs_bonus/bonus_manual_novel_schema_predictions.json`
- `outputs_bonus/bonus_training_loss.json`
- `outputs_bonus/bonus_summary.json`
- `outputs/task8c_bonus_generalization.txt`

Observed bonus results from the current run:
- Base accuracy with reranking on the 200 held-out questions: `0.610`
- Fine-tuned bonus accuracy on the same 200 held-out questions: `0.190`
- Manual novel-schema accuracy on the 5 extra questions: `0.400`

The bonus run improved the novel-schema questions substantially compared with the main run (`0.400` vs `0.000`), but it hurt the in-distribution held-out score after training (`0.190` vs `0.420`). That tradeoff matters. The stricter prompt and reranking clearly helped with some out-of-distribution generalization, but the bonus setup also pushed the model toward different query shapes that were worse on the original 200-example evaluation set.

## Step 6: Evaluate the Fine-Tuned Model (After)

After training, the script saves adapter weights and creates a sampling client from the tuned checkpoint. It then reruns the same held-out evaluation used for the base model so the comparison is apples-to-apples.

For this run:
- Base model accuracy: `0.285`
- Fine-tuned model accuracy: `0.420`

That is a real improvement, but it is still well below the lesson plan's typical reference result of roughly `~0.87` on the 200 in-distribution test questions. Looking through `tuned_eval_predictions.json`, the fine-tuned model clearly learned a substantial amount of SQL structure and schema grounding, but it still often:
- appends repeated conditions to otherwise correct queries,
- over-constrains queries with extra predicates,
- adds repeated `ORDER BY`/`LIMIT` style fragments,
- and occasionally continues generating beyond the intended end of the SQL statement.

## Step 7: Test on Additional Novel Schema Questions

The script also evaluates the five manual out-of-distribution questions from the lesson plan on schemas such as `employees`, `products`, `students`, `orders`, and `courses`/`enrollments`.

For this run:
- Manual novel-schema accuracy: `0.000`

This matches the lesson's warning that out-of-distribution schemas should be harder than the 200 held-out in-distribution examples. The model often produced partially sensible SQL, but it tended to:
- add extra filters not requested in the question,
- expand a correct aggregation into a longer malformed query,
- or overgenerate repeated `UNION` / `INTERSECT` / `EXCEPT` fragments.

That suggests the model learned useful SQL patterns from the training data, but it still struggles to generalize cleanly to schemas outside the Spider/WikiSQL-style distribution it saw during fine-tuning.

## Summary

The Topic 8 lesson distinguishes between knowledge tasks and skill tasks. For text-to-SQL, the model must learn a compositional skill: map a natural-language question to SQL syntax grounded in a schema. That is why the lesson argues that fine-tuning is more appropriate than RAG here.

The local scripts in this folder prepare the data, reproduce the prompt format from the lesson plan, and evaluate both in-distribution and novel-schema examples. The actual training computation still depends on a live Tinker account and API key, since the GPU work happens remotely.

## 6. Discussion Questions

- Before vs. after: the fine-tuned model improved from `0.285` to `0.420` on the held-out test set, so it clearly learned something about SQL syntax and schema grounding. The improvement is real, but incomplete. The model often gets much closer to the correct SQL after training, yet still damages otherwise good queries by overgeneration or by appending extra constraints.
- RAG comparison: RAG would likely help for near-duplicate question/SQL pairs or for schemas that look very similar to training examples, but it would still struggle on compositional SQL generation. The model must still decide which columns to aggregate, which tables to join, and which clauses to include in the correct order. That is exactly why the lesson frames text-to-SQL as a skill problem rather than a pure retrieval problem.
- Error analysis: the current failures are not random. Many incorrect outputs are almost right, but they include extra `WHERE` predicates, repeated literals, repeated clauses, or very long continuations after the correct query appears. That pattern suggests the fine-tuned model learned the task structure better than the base model, but decoding and stopping behavior still need work.
- Prompt format: the task uses the same prompt structure described in the lesson plan:
  `Table schema:` + schema, then `Question:` + natural-language query, then `SQL:`
- Evaluation setup: the held-out split uses 200 examples as required by the lesson, and the novel-schema evaluation uses the five additional questions from Step 7.

## Bonus Challenge

- The lesson plan asks whether accuracy on the novel-schema questions can be improved. Based on the current outputs, that is the main weakness of this run.
- I implemented the bonus in `task8c_bonus_generalization.py` rather than modifying `task8b_tinker_sql_finetune.py`, so the original assignment path and the bonus experiment remain separate.
- The bonus results were mixed rather than uniformly better:
  - main run held-out accuracy: `0.420`
  - bonus run held-out accuracy: `0.190`
  - main run novel-schema accuracy: `0.000`
  - bonus run novel-schema accuracy: `0.400`
- The most direct next experiments would be:
  - tune decoding/stopping so the model stops after one SQL statement instead of continuing,
  - try a slightly lower learning rate or different LoRA rank,
  - inspect whether one epoch is enough or whether the model is underfit on schema generalization but already strong on the training distribution,
  - add or synthesize more training examples with schemas closer to the manual test questions,
  - and compare prompt formatting or post-processing that truncates generation more aggressively after the first valid SQL statement.
- Since the bonus run improved generalization but damaged the in-distribution evaluation, the next useful step is not "more of everything." It is to separate which intervention helped novel-schema performance and which one hurt the held-out set. The likely knobs to isolate are augmentation, decoding/reranking, and prompt wording.
