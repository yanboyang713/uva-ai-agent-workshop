# Topic 2: Agent Orchestration Frameworks (LangGraph)

This folder contains my portfolio work for **Topic 2: Agent Orchestration Frameworks**.

## Table of contents

- `requirements.txt` — dependencies used in the provided starter code
- `langgraph_simple_llama_agent.py` — downloaded starter program (unmodified)
- `langgraph_crash_recovery.html` — downloaded reading on checkpointing / recovery
- Task 0: Read the LangChain Graph API
- Task 1: tracing toggle
  - `task1_tracing_verbose_quiet.py`
  - outputs from my terminal sessions on /outputs/task1/task1.txt
- Task 2: Please, check `task2_empty_input_observations.md` first get more details.
  - Modify the code
  - `task2_no_empty_input_branch.py`
- Task 3: run Llama + Qwen in parallel, then join and print both
  - `task3_parallel_llama_qwen.py`
  - outputs from my terminal sessions on /outputs/task3/task3.txt
- Task 4: run only one model (route by `"Hey Qwen"`)
  - `task4_router_hey_qwen.py`
  - outputs from my terminal sessions on /outputs/task4/task4.txt
- Task 5: chat history using a messages list (Llama only)
  - `task5_chat_history_messages_llama_only.py`
  - outputs from my terminal sessions on /outputs/task5/task5.txt
- Task 6: chat history + switching between Llama and Qwen (3 participants)
  - `task6_chat_history_switch_llama_qwen.py`
  - `script -q -f ./outputs/task6/task6.txt -c "python -u task6_chat_history_switch_llama_qwen.py"`
  - outputs from my terminal sessions on /outputs/task6/task6.txt
- Task 7: checkpointing + crash recovery (resume after kill/restart)
  - `task7_checkpoint_recovery_chat.py`
  - `script -q -f ./outputs/task7/task7.txt -c "python -u task7_checkpoint_recovery_chat.py"`
  - outputs/task7/task7.txt - first run
  - outputs/task7/task7-next-run.txt - second run
- Run logs
  - `outputs/` — put captured terminal outputs here (use `tee`)

## How to run

From the repo root:

```bash
conda create -n topic2 -y python=3.12 jupyterlab -c conda-forge
conda activate topic2
pip install -r requirements.txt
python -u task1_tracing_verbose_quiet.py 2>&1 | tee task1.txt
conda deactivate
```

To capture terminal output into a file:

```bash
python -u task1_tracing_verbose_quiet.py 2>&1 | tee task1.txt
```

Notes:
- The default Llama model is `meta-llama/Llama-3.2-1B-Instruct`.
- Override model IDs with `--llama-model ...` / `--qwen-model ...` or env vars `LLAMA_MODEL_ID` / `QWEN_MODEL_ID`.
- For Tasks 1–2, you can change the model with `--model-id ...` and optionally sanity-check the graph with `--dry-run` first.
