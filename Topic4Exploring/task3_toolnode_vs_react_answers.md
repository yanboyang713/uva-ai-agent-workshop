# Task 3: ToolNode vs create_react_agent

Based on the provided starter files:
- `toolnode_example.py`
- `react_agent_example.py`

## How to run these two programs (Task 3)

### 1) Environment and dependencies

From repo root:

```bash
conda activate topic3
pip install -r Topic4Exploring/requirements.txt
```

Set OpenAI key in shell:

```bash
export OPENAI_API_KEY="your-key"
```

### 2) Run `toolnode_example.py` (interactive)

Run from the Topic 4 folder so graph PNG files are saved there:

```bash
cd Topic4Exploring
python -u toolnode_example.py
```

Test inputs at the prompt:
- `What is the weather in Tokyo and population of New York?`
- `verbose`
- `2 + 2 * 10`
- `exit`

### 3) Run `react_agent_example.py` (interactive)

```bash
cd Topic4Exploring
python -u react_agent_example.py
```

Suggested test inputs at the prompt:
- `What is the weather in London and population of Paris?`
- `quiet`
- `What is 144 / 12?`
- `exit`

### 4) Capture terminal output for portfolio files

Run non-interactively by piping inputs:

```bash
cd Topic4Exploring
printf "What is the weather in Tokyo and population of New York?\nverbose\n2 + 2 * 10\nexit\n" \
  | python -u toolnode_example.py 2>&1 | tee ./outputs/task3_analysis/task3_toolnode_run.txt

printf "What is the weather in London and population of Paris?\nquiet\nWhat is 144 / 12?\nexit\n" \
  | python -u react_agent_example.py 2>&1 | tee ./outputs/task3_analysis/task3_react_run.txt
```

The programs also generate graph images in `Topic4Exploring/`:
- ToolNode version: `langchain_manual_tool_graph.png`
- ReAct version: `langchain_react_agent.png` and `langchain_conversation_graph.png`

## 1) What Python features does ToolNode use to dispatch tools in parallel?
ToolNode relies on asynchronous/concurrent execution patterns under the hood (async tasks and gathering results), plus function metadata from tool objects for dynamic dispatch.
Tools that benefit most are I/O-bound tools with independent calls:
- web/API lookups
- database queries
- file/network retrieval

CPU-bound tools usually benefit less unless they release the GIL or run in separate workers.

## 2) How do the two programs handle special inputs such as "verbose" and "exit"?
Both implementations parse user input in an input node and set a `command` field in graph state:
- `"exit"` routes to `END`
- `"verbose"` / `"quiet"` toggle the state flag and route back to input
- normal text clears command and continues to the model node

So special command handling is explicit graph routing in both programs.

## 3) Compare the graph diagrams of the two programs. How do they differ?
They are very similar at the conversation-wrapper level:
- input -> model/agent -> output -> trim_history -> input

Main difference:
- ToolNode version has explicit `tools` node and conditional routing from model to tools, then back to model.
- ReAct version hides tool loop inside `create_react_agent`, so wrapper graph is simpler while internal agent graph handles the action/observation loop.

## 4) When is create_react_agent too restrictive vs ToolNode?
Use ToolNode/manual orchestration when you need custom control flow that is not standard ReAct:
- strict business rules on tool order
- branch-specific retry/timeout behavior per tool
- custom safety/approval gates between reasoning and tool execution
- deterministic multi-tool pipelines with mixed parallel + sequential stages
- richer state transitions than one generic ReAct loop
