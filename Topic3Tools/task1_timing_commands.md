# Task 1 Timing Guide (Sequential vs Parallel)

## 0) Ollama model setup (Llama 3.2 1B)

Pull the model:

```bash
ollama pull llama3.2:1b
```

Verify it is installed:

```bash
ollama list
```

Quick sanity check:

```bash
ollama run llama3.2:1b
```

## 1) Baseline with local HF models (Topic 1 style)

Use these two scripts (single subject each):

- `task1_program1_topicA.py`
- `task1_program2_topicB.py`

Time sequential:

```bash
time { python task1_program1_topicA.py ; python task1_program2_topicB.py ; }
```
- python task1_program1_topicA.py  744.04s user 9.98s system 374% cpu 3:21.34 total
- python task1_program2_topicB.py  497.33s user 7.24s system 374% cpu 2:14.60 total

Time parallel:

```bash
time { python task1_program1_topicA.py & python task1_program2_topicB.py & wait; }
```
======================================================================
EVALUATION SUMMARY
======================================================================
Model: meta-llama/Llama-3.2-1B-Instruct
None (full precision)
Total Subjects: 1
Total Questions: 152
Total Correct: 75
Overall Accuracy: 49.34%
Duration: 8.9 minutes
Real time: 531.09 s
CPU time: 1142.04 s (user 1108.04 s, system 34.00 s)
======================================================================

EVALUATION SUMMARY
======================================================================
Model: meta-llama/Llama-3.2-1B-Instruct
None (full precision)
Total Subjects: 1
Total Questions: 100
Total Correct: 45
Overall Accuracy: 45.00%
Duration: 7.8 minutes
Real time: 468.23 s
CPU time: 897.43 s (user 865.57 s, system 31.87 s)
======================================================================
## 2) Repeat using Ollama-backed versions

Use these Ollama variants:

- `task1_program1_topicA_ollama.py`
- `task1_program2_topicB_ollama.py`

Quick smoke test:

```bash
python task1_program1_topicA_ollama.py --limit 5
python task1_program2_topicB_ollama.py --limit 5
```

Then time full sequential and parallel runs:

```bash
time { python task1_program1_topicA_ollama.py ; python task1_program2_topicB_ollama.py ; }
time { python task1_program1_topicA_ollama.py & python task1_program2_topicB_ollama.py & wait; }
```

## 3) Save logs for portfolio

```bash
time { python task1_program1_topicA_ollama.py ; python task1_program2_topicB_ollama.py ; } \
  2>&1 | tee outputs/task1/task1_sequential_ollama.txt

time { python task1_program1_topicA_ollama.py & python task1_program2_topicB_ollama.py & wait; } \
  2>&1 | tee outputs/task1/task1_parallel_ollama.txt
```
