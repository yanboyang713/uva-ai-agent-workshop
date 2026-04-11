# Topic 7: MCP and A2A

This folder contains my work for Topic 7, focused on:
- `2. MCP lesson plan and tasks`
- `3. A2A lesson plan and tasks`

Course page:
- https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic7MCP/mcp.html

## Table of Contents

- `requirements.txt` - dependencies for MCP and A2A tasks
- `.env.example` - environment variable template
- `mcp_common.py` - shared MCP client/utilities (tools/list, tools/call, parsing)
- `task2a_mcp_tools_discovery.py` - Exercise A: discover Asta tools
- `task2b_mcp_direct_drills.py` - Exercise B: direct calls for search/citations/references
- `task2c_mcp_research_chatbot.py` - Exercise C: dynamic tool-calling research chatbot
- `task2d_mcp_citation_network_explorer.py` - Exercise D: autonomous citation-neighborhood report
- `a2a_agent_template.py` - A2A student agent template
- `a2a_registry.py` - A2A registry server
- `a2a_test.py` - local end-to-end A2A pipeline test
- `a2a_trivia.py` - trivia tournament orchestrator
- `semantic_similarity.py` - optional embedding-based similarity helper
- `test_ngrok.sh` - quick local ngrok sanity check helper
- `outputs/` - suggested location for saved terminal logs and generated outputs

## Setup

```bash
conda create -n topic7 -y python=3.12
conda activate topic7
pip install -r Topic7MCP/requirements.txt
cp Topic7MCP/.env.example Topic7MCP/.env
```

Then edit `Topic7MCP/.env` with real values:
- `OPENAI_API_KEY`
- `ASTA_API_KEY`
- `REGISTRY_URL` (for A2A class run)

## MCP Tasks (Section 2)

### Exercise A - discover tools

```bash
python -u Topic7MCP/task2a_mcp_tools_discovery.py \
  2>&1 | tee Topic7MCP/outputs/mcp/task2a_tools_discovery.txt
```

### Exercise B - three direct drills

```bash
python -u Topic7MCP/task2b_mcp_direct_drills.py \
  2>&1 | tee Topic7MCP/outputs/mcp/task2b_direct_drills.txt
```

Discussion after drills:

- The live Asta endpoint on April 11, 2026 does not exactly match the older lesson text. It exposes `search_papers_by_relevance` instead of `search_papers`, and it does not expose `get_references`, so the drill script falls back to `get_paper(fields=references)` plus `get_paper_batch` for metadata enrichment.
- `search_papers_by_relevance` returns multiple paper hits and is best treated as a list of paper records.
- `get_citations` also returns a list, but the records are "papers that cite the seed paper" rather than direct paper search hits, so the semantic meaning of each row is different even though the parsed shape is still list-like.
- The fallback references flow is structurally different: `get_paper(fields=references)` returns one seed paper object that contains a nested `references` list, and those nested reference objects are often partial records with only `paperId` and `title`. To sort references by year, I had to fetch extra metadata with `get_paper_batch`.
- The MCP server often places human-readable JSON snippets inside `result.content[*].text`, but for the current Asta server the cleaner machine-readable payload is usually in `result.structuredContent`. I updated `mcp_common.py` to prefer `structuredContent` first and only flatten `content[*].text` when structured data is missing.
- When the only available data is JSON embedded in `content[0]["text"]`, I treat that text as a JSON string and parse it with the shared helpers in `mcp_common.py` (`extract_mcp_text`, `parse_json_maybe`, and `normalize_rows_from_tool_text`) rather than string-slicing it by hand.

### Exercise C - dynamic MCP chatbot

Single question mode:

```bash
python -u Topic7MCP/task2c_mcp_research_chatbot.py \
  --once "Find recent papers about large language model agents" \
  2>&1 | tee Topic7MCP/outputs/mcp/task2c_chatbot_once.txt
```

Interactive mode:

```bash
python -u Topic7MCP/task2c_mcp_research_chatbot.py
```

Output: Topic7MCP/outputs/mcp/task2c.txt

Discussion:

- Compared to Exercise B, Exercise C does not hard-code tool names, argument shapes, or per-tool calling logic into the main chatbot workflow.
- The chatbot first loads the current MCP tool inventory from `tools/list`, then converts each tool's server-provided schema into an OpenAI tool definition.
- Because the schema comes from the server, I wrote almost no tool-specific code. The model decides which tool to call and what arguments to send using the live tool definitions.
- In Exercise B, I had to manually choose tools, manually supply the right argument names, and manually work around mismatches between the older lesson text and the live Asta server.
- In Exercise C, the structure is generic. If Asta added new tools tomorrow, the chatbot would pick them up from `tools/list` and work the same way without rewriting the orchestration layer.
- That is the core value of MCP in this assignment: the client stays general, while the server remains the source of truth for tool capabilities and schemas.

### Exercise D - citation neighborhood explorer

```bash
python -u Topic7MCP/task2d_mcp_citation_network_explorer.py ARXIV:2210.03629 \
  > Topic7MCP/outputs/mcp/task2d_react_report.md
```

### Closing Discussion

- Writing tool schemas by hand gives you direct control, but MCP automation removes a large amount of duplicated client code. The main benefit is adaptability: the client can discover tools and argument schemas at runtime instead of hard-coding every capability. The cost is extra protocol complexity and new failure modes, such as transport negotiation problems, schema drift, incomplete server metadata, and the need to robustly parse structured tool outputs.
- The Asta tools return much more JSON than the model actually needs for a useful answer, so I had to filter aggressively. In practice, I kept compact fields that change the reasoning outcome, such as title, year, abstract, authors, citation counts, and a small number of top references or recent citations. I discarded bulky or repetitive fields that would mostly consume tokens without improving the answer. When I passed too much raw output, the context became noisy and the responses were less focused. When I summarized and selected only the relevant fields, the responses were shorter and more coherent.
- In Exercise D, I fixed the tool-calling order myself: get the seed paper, then references, then citations, then author follow-ups, then summarize. Letting the LLM decide the order would require exposing those tools dynamically, defining stopping conditions, tracking intermediate state, and giving the model a planner role rather than a single summarizer role. That would be more flexible, but it could also go wrong in predictable ways: redundant tool calls, missing essential steps, shallow exploration, overlong context windows, or expensive loops where the model keeps asking for more data without improving the report.
- MCP is still young, and a mature ecosystem should offer more than just a transport for tools. I would want stronger interoperability guarantees across servers, better schema conventions, richer capability discovery, versioning support, standard auth/session patterns, better observability for debugging tool calls, and more reliable structured outputs so clients do not need so many defensive fallbacks. Stronger ecosystem support for caching, pagination, incremental retrieval, and quality metadata would also make MCP systems easier to build and much more reliable.

## A2A Tasks (Section 3)

### Local no-ngrok verification

```bash
python -u Topic7MCP/a2a_test.py \
  2>&1 | tee Topic7MCP/outputs/a2a/a2a_local_test.txt
```

### Real class flow with ngrok

Install ngrok on Arch Linux:

```bash
yay -S ngrok
```

Configure ngrok:

1. Sign up for a verified ngrok account at `https://dashboard.ngrok.com/signup`
2. Copy your authtoken from `https://dashboard.ngrok.com/get-started/your-authtoken`
3. Add it locally:

```bash
ngrok config add-authtoken <your_token>
ngrok config check
```

Terminal 1:

```bash
ngrok http 8000
```

Terminal 2:

```bash
python -u Topic7MCP/a2a_agent_template.py
```

Optional dry-run (prompt tuning without ngrok/registry):

```bash
python -u Topic7MCP/a2a_agent_template.py --dryrun
```

If you run registry yourself:

```bash
python -u Topic7MCP/a2a_registry.py
```
