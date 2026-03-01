# Topic 4: Exploring Tools

This directory contains my portfolio work for Topic 4.

## Table of contents

- `requirements.txt` - dependencies for Topic 4 scripts
- `task3_toolnode_vs_react_answers.md` - answers to ToolNode vs create_react_agent questions
- `task5_option1_wikipedia_duckduckgo_research_assistant.py` - Option 1 project implementation
- `outputs/task3_analysis/` - notes or captures for Task 3 analysis
- `outputs/task5_option1/` - traces and terminal logs for Option 1

## Setup

```bash
conda create -n topic4 -y python=3.12
conda activate topic4
pip install -r Topic4Exploring/requirements.txt
```

## Task 3 (starter code analysis)

Question answers are in:
- `Topic4Exploring/task3_toolnode_vs_react_answers.md`

## Task 5: 2-Hour Agent Project (Option 1)

Project chosen:
- **Option 1: Wikipedia + DuckDuckGo**
- "Research Assistant - Compare Sources"

### How to run (Ollama, no API key)

```bash
ollama pull llama3.2:1b
python -u Topic4Exploring/task5_option1_wikipedia_duckduckgo_research_assistant.py \
  --provider ollama \
  --model llama3.2:1b \
  --query "Compare key claims about CRISPR gene editing: what do Wikipedia and recent web results emphasize?" \
  2>&1 | tee Topic4Exploring/outputs/task5_option1/task5_option1_run1.txt
```

Second sample query:

```bash
python -u Topic4Exploring/task5_option1_wikipedia_duckduckgo_research_assistant.py \
  --provider ollama \
  --model llama3.2:1b \
  --query "Compare Wikipedia and web coverage of fusion energy progress in the last few years." \
  2>&1 | tee Topic4Exploring/outputs/task5_option1/task3_option1_run2.txt
```

The script also writes a timestamped JSON trace in:
- `Topic4Exploring/outputs/task5_option1/`
  - Note: older runs before this fix wrote JSON traces under `outputs/task3_option1/`.

Generated run files in this portfolio:
- `Topic4Exploring/outputs/task5_option1/task5_option1_run1.txt`
- `Topic4Exploring/outputs/task5_option1/task3_option1_run1.txt`
- `Topic4Exploring/outputs/task5_option1/task3_option1_run2.txt`
- `Topic4Exploring/outputs/task5_option1/option1_trace_20260301_141506.json`
- `Topic4Exploring/outputs/task5_option1/option1_trace_20260301_141623.json`

### Discussion

What worked:
- In `task3_option1_run1.txt`, the agent invoked both tools in one step (`wikipedia_lookup` + `duckduckgo_search`).
- The JSON traces `option1_trace_20260301_141506.json` and `option1_trace_20260301_141623.json` confirm two-tool usage and captured tool arguments/results.
- The report format (topic + compare/contrast + brief bullets) was produced consistently.

Interpretation:
- Small local model behavior (`llama3.2:1b`) is less reliable for strict tool-policy adherence.
- DuckDuckGo Instant Answer API often returns sparse content for some queries, which can reduce useful retrieval.
