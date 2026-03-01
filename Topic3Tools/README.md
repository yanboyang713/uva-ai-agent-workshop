# Topic 3: Agent Tool Use

This directory contains my portfolio work for Topic 3.

## Table of contents

- `requirements.txt` - Python dependencies for Topic 3 scripts
- `task1_timing_commands.md` - commands and workflow for sequential vs parallel timing (local model vs Ollama)
- `task2_gpt4omini_test.py` - OpenAI API setup test with `gpt-4o-mini`
- `task3_manual_tool_handling_custom.py` - manual tool loop with custom calculator (geometry included)
- `task4_langgraph_tool_handling_custom.py` - LangChain/LangGraph-style tool loop with 4 tools
- `task5_langgraph_conversation_checkpoint.py` - single long conversation via LangGraph nodes + SQLite checkpointing/recovery
- `outputs/` - terminal traces, logs, and checkpoint DB files
- `outputs/task4/task4_trace_excerpts.md` - selected Task 4 trace excerpts used in discussion

## Environment setup

From repo root:

```bash
conda create -n topic3 -y python=3.12 jupyterlab -c conda-forge
conda activate topic3
pip install -r Topic3Tools/requirements.txt
```

Set your OpenAI key in your shell profile (do not commit secrets):

```bash
export OPENAI_API_KEY="your-actual-key"
```

## Task 1 observations (Ollama timing)

Using the logs:
- `outputs/task1/task1_sequential_ollama.txt`
- `outputs/task1/task1_parallel_ollama.txt`

Measured durations from the scripts:

| Run mode | Topic A: astronomy | Topic B: business_ethics | Combined wall-clock for both topics |
|---|---:|---:|---:|
| Sequential | 246.35 s | 171.61 s | 417.96 s |
| Parallel | 444.14 s | 361.02 s | ~444.14 s (the slower of the two) |

What I observed:
- Parallel was slower overall than sequential on this setup: about `26.18 s` slower (`~6.3%` slower wall-clock).
- Each individual job became much slower in parallel.
- `astronomy`: `246.35 s -> 444.14 s` (`~80%` slower).
- `business_ethics`: `171.61 s -> 361.02 s` (`~110%` slower).
- Accuracy stayed roughly similar:
- `astronomy` stayed at `25%` in both runs.
- `business_ethics` changed slightly (`31%` sequential vs `30%` parallel).

Interpretation:
- Running two clients against one local Ollama server/model on the same machine caused strong resource contention (CPU/RAM and model serving queueing).
- In this configuration, parallel launch did not create true speedup and instead increased latency per request.

## Task 2 notes

`client = OpenAI()`
- Constructs the OpenAI API client object and reads credentials/config from environment variables (especially `OPENAI_API_KEY`) unless explicitly provided.

`response = client.chat.completions.create(...)`
- Sends a chat completion request to the selected model with the provided messages and generation settings, then returns the model output (plus metadata like usage tokens).


```bash
python -u Topic3Tools/task2_gpt4omini_test.py 2>&1 | tee Topic3Tools/outputs/task2/task2_gpt4omini_test.txt

```

## Task 3 
```bash
python -u Topic3Tools/task3_manual_tool_handling_custom.py 2>&1 | tee Topic3Tools/outputs/task3/task3_manual.txt
```

## Task 4 
```bash
python -u Topic3Tools/task4_langgraph_tool_handling_custom.py 2>&1 | tee Topic3Tools/outputs/task4/task4_langgraph_tools.txt
```

### Task 4 discussion from traces

Trace file used:
- `Topic3Tools/outputs/task4/task4_langgraph_tools.txt`
- `Topic3Tools/outputs/task4/task4_trace_excerpts.md`

Observed behavior:
- Multiple tool calls in one turn worked.
- In Test 2, Iteration 1 invoked `count_letter` twice (`i` and `s`) before producing the final answer in Iteration 2.
- Sequential chaining across outer-loop iterations worked.
- In Test 3, Iteration 1 invoked `count_letter` twice, Iteration 2 invoked `calculator`, and Iteration 3 produced the final response.
- All required tools were exercised in this run set:
- `count_letter` (Tests 1-3)
- `calculator` (Test 3)
- `get_weather` and `text_stats` (Test 4)

Assignment-specific answers:
- A single query using all tools was not observed in the current trace. The tools were all used across multiple tests.
- The 5-turn outer-loop limit was not reached in the current trace. The longest chain observed was 3 iterations.

Trace-backed evidence:
- Test 2 shows two `count_letter` calls in the same iteration (multi-tool in one turn).
- Test 3 shows cross-iteration chaining: `count_letter` tools first, then `calculator` in next iteration, then final answer.
- Test 4 shows `get_weather` and `text_stats` together in one iteration.

Queries used for the assignment goals:
1. Single query intended to use all tools:
`For "Mississippi riverboats" and "Tokyo", do all of the following using tools: count the number of i's, count the number of s's, compute sin(i-s), get Tokyo weather, and report the number of unique letters in "Tokyo". Then summarize all results.`
2. Query intended to push sequential chaining to the 5-iteration cap:
`Use exactly one tool call per iteration and do not combine tool calls in one response. Steps: (1) count i in "Mississippi riverboats", (2) count s in "Mississippi riverboats", (3) calculate i-s, (4) calculate sin(i-s), (5) get weather in Tokyo, (6) compute unique letters in "Tokyo". Only after step 6 give the final answer.`

Command to run and capture these traces:

```bash
python -u Topic3Tools/task4_langgraph_tool_handling_custom.py \
  --max-iterations 5 \
  --query "For \"Mississippi riverboats\" and \"Tokyo\", do all of the following using tools: count the number of i's, count the number of s's, compute sin(i-s), get Tokyo weather, and report the number of unique letters in \"Tokyo\". Then summarize all results." \
  --query "Use exactly one tool call per iteration and do not combine tool calls in one response. Steps: (1) count i in \"Mississippi riverboats\", (2) count s in \"Mississippi riverboats\", (3) calculate i-s, (4) calculate sin(i-s), (5) get weather in Tokyo, (6) compute unique letters in \"Tokyo\". Only after step 6 give the final answer." \
  2>&1 | tee Topic3Tools/outputs/task4/task4_langgraph_tools_extra.txt
```

## Task 5 
```bash
python -u Topic3Tools/task5_langgraph_conversation_checkpoint.py \
  --thread-id topic3-demo \
  --checkpoint-db Topic3Tools/outputs/task5/task5_checkpoints.db \
  2>&1 | tee Topic3Tools/outputs/task5/task5_chat.txt
```

Recovery test (start, stop, resume):

```bash
python -u Topic3Tools/task5_langgraph_conversation_checkpoint.py \
  --thread-id topic3-demo \
  --checkpoint-db Topic3Tools/outputs/task5/task5_checkpoints.db
```

Use the same `--thread-id` and `--checkpoint-db` on restart to recover state.

## Task 5 Mermaid diagram

```mermaid
flowchart TD
    A[Human Input] --> B[Agent Node: LLM with tools bound]
    B -->|Tool calls present| C[Tools Node: execute each tool]
    C --> B
    B -->|No tool calls| D[Assistant response returned]
    D --> A
```

## Task 6 question: missed parallelization opportunity
When the assistant emits multiple independent tool calls in one turn (for example two `count_letter` calls), the current tools node executes them sequentially.
A clear optimization is to execute independent tool calls concurrently (thread pool or async gather), then feed all tool results back together.
