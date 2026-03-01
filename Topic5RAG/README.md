# Topic 5: RAG - Retrieval Augmented Generation

This folder contains my portfolio work for Topic 5.

## Table of contents

- `requirements.txt` - dependencies for Topic 5 scripts
- `download_topic5_resources.sh` - downloads notebook + corpora from course site
- `rag_core.py` - reusable RAG core (loading, chunking, retrieval, LLM calls)
- `run_topic5_exercises.py` - CLI runner for Exercises 1, 2, and 4-11
- `task3_frontier_manual_comparison.md` - manual template for Exercise 3
- `outputs/` - saved JSON/TXT outputs by exercise

## Team

- Team members: add names here.

## Setup

```bash
conda activate topic3
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

## Exercise commands

### Exercise 1: Open Model RAG vs No-RAG

```bash
python -u Topic5RAG/run_topic5_exercises.py ex1 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --rag-k 5 \
  --output-dir Topic5RAG/outputs/ex1 \
  2>&1 | tee Topic5RAG/outputs/ex1/ex1_terminal.txt
```

### Exercise 2: Open Model + RAG vs GPT-4o Mini

```bash
python -u Topic5RAG/run_topic5_exercises.py ex2 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
  --large-provider openai \
  --large-model gpt-4o-mini \
  --rag-k 5 \
  --output-dir Topic5RAG/outputs/ex2 \
  2>&1 | tee Topic5RAG/outputs/ex2/ex2_terminal.txt
```

### Exercise 3: Frontier model manual comparison

- Use `task3_frontier_manual_comparison.md` to record web UI comparisons.

### Exercise 4: Top-K retrieval effect

```bash
python -u Topic5RAG/run_topic5_exercises.py ex4 \
  --corpus modelt \
  --corpus-dir Topic5RAG/resources/Corpora/NewModelT \
  --open-provider ollama \
  --open-model llama3.2:1b \
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
