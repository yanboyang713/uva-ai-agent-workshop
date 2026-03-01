"""
Topic 4, Option 1: Wikipedia + DuckDuckGo
Project: Research Assistant - Compare Sources

This version avoids extra third-party tool packages:
- Wikipedia tool uses MediaWiki HTTP API
- DuckDuckGo tool uses DDG Instant Answer HTTP API
- Agent uses LangGraph create_react_agent
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


USER_AGENT = "topic4-exploring-tools/1.0"


def _http_get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@tool
def wikipedia_lookup(query: str) -> str:
    """Search Wikipedia and return concise summaries for top matches."""
    search_params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        }
    )
    search_url = f"https://en.wikipedia.org/w/api.php?{search_params}"
    search_data = _http_get_json(search_url)
    hits = search_data.get("query", {}).get("search", [])

    if not hits:
        return "No Wikipedia matches found."

    lines: list[str] = []
    for hit in hits:
        title = hit.get("title", "")
        snippet = hit.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        lines.append(f"- {title}: {snippet} (URL: {page_url})")

    return "Wikipedia matches:\n" + "\n".join(lines)


@tool
def duckduckgo_search(query: str) -> str:
    """Search DuckDuckGo (instant answer endpoint) and return summary + related topics."""
    ddg_params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
    )
    ddg_url = f"https://api.duckduckgo.com/?{ddg_params}"
    data = _http_get_json(ddg_url)

    lines: list[str] = []
    abstract = data.get("AbstractText", "")
    abstract_url = data.get("AbstractURL", "")
    heading = data.get("Heading", "")
    if heading or abstract:
        lines.append(f"- Instant Answer: {heading} | {abstract} (URL: {abstract_url})")

    related = data.get("RelatedTopics", [])[:5]
    for item in related:
        if "Topics" in item:
            for sub in item.get("Topics", [])[:3]:
                text = sub.get("Text", "")
                first_url = sub.get("FirstURL", "")
                if text:
                    lines.append(f"- Related: {text} (URL: {first_url})")
        else:
            text = item.get("Text", "")
            first_url = item.get("FirstURL", "")
            if text:
                lines.append(f"- Related: {text} (URL: {first_url})")

    if not lines:
        return "DuckDuckGo did not return useful instant-answer content for this query."
    return "DuckDuckGo findings:\n" + "\n".join(lines[:8])


def build_model(provider: str, model_name: str) -> ChatOpenAI:
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url=base_url,
            api_key="ollama",
        )

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for provider=openai.")
        return ChatOpenAI(model=model_name, temperature=0)

    raise ValueError(f"Unsupported provider: {provider}")


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _message_to_trace_row(index: int, msg: BaseMessage) -> dict[str, Any]:
    row: dict[str, Any] = {"index": index, "type": type(msg).__name__}
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            row["tool_calls"] = msg.tool_calls
        if msg.content:
            row["content"] = msg.content
    elif isinstance(msg, ToolMessage):
        row["name"] = getattr(msg, "name", None)
        row["tool_call_id"] = msg.tool_call_id
        row["content"] = msg.content
    else:
        row["content"] = getattr(msg, "content", None)
    return row


def run_research(
    *,
    query: str,
    provider: str,
    model_name: str,
    recursion_limit: int,
) -> tuple[str, list[BaseMessage], list[dict[str, Any]]]:
    model = build_model(provider, model_name)
    tools = [wikipedia_lookup, duckduckgo_search]

    system_prompt = (
        "You are a research assistant. You MUST call both tools at least once:\n"
        "1) wikipedia_lookup\n"
        "2) duckduckgo_search\n"
        "After using both, produce a concise report with sections:\n"
        "- Topic\n"
        "- Wikipedia Findings\n"
        "- DuckDuckGo/Web Findings\n"
        "- Compare/Contrast\n"
        "- Brief Report (3-6 bullets)\n"
        "Cite concrete points from each source in plain text."
    )

    app = create_react_agent(model=model, tools=tools, prompt=system_prompt)
    result = app.invoke(
        {"messages": [("user", query)]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    final_answer = _extract_final_answer(messages)
    trace_rows = [_message_to_trace_row(i, m) for i, m in enumerate(messages, start=1)]
    return final_answer, messages, trace_rows


def save_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def print_trace(messages: list[BaseMessage]) -> None:
    print("\n=== Trace (tool activity) ===")
    for idx, msg in enumerate(messages, start=1):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"[{idx}] AI requested {len(msg.tool_calls)} tool call(s)")
            for tc in msg.tool_calls:
                print(f"      - {tc['name']} args={tc['args']}")
        elif isinstance(msg, ToolMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(text) > 240:
                text = text[:240] + "..."
            print(f"[{idx}] Tool result ({getattr(msg, 'name', 'unknown')}): {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Option 1: Wikipedia + DuckDuckGo research assistant")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Model name (for ollama use local tag, for openai use model id).",
    )
    parser.add_argument(
        "--query",
        default="Compare key claims about CRISPR gene editing: what do Wikipedia and web results emphasize?",
        help="Research query.",
    )
    parser.add_argument("--recursion-limit", type=int, default=20)
    parser.add_argument(
        "--trace-json",
        default=None,
        help="Optional trace JSON path. Defaults to outputs/task5_option1 timestamp file.",
    )
    args = parser.parse_args()

    final_answer, messages, trace_rows = run_research(
        query=args.query,
        provider=args.provider,
        model_name=args.model,
        recursion_limit=args.recursion_limit,
    )

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print_trace(messages)
    print("\n=== Final Report ===\n")
    print(final_answer)

    if args.trace_json:
        trace_path = Path(args.trace_json)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_path = Path(__file__).resolve().parent / "outputs" / "task5_option1" / f"option1_trace_{ts}.json"

    payload = {
        "provider": args.provider,
        "model": args.model,
        "query": args.query,
        "trace": trace_rows,
        "final_answer": final_answer,
    }
    save_trace(trace_path, payload)
    print(f"\nSaved trace JSON: {trace_path}")


if __name__ == "__main__":
    main()
