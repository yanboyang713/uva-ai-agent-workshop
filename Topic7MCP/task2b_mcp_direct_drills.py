"""Exercise B (MCP): three direct tool-call drills against Asta."""

from __future__ import annotations

from typing import Any

from mcp_common import (
    MCPClient,
    normalize_rows_from_tool_text,
    tool_name_or_none,
    unwrap_embedded_paper,
)


def _year_of(paper: dict[str, Any]) -> int:
    year = paper.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    return -1


def _title_of(paper: dict[str, Any]) -> str:
    return str(paper.get("title") or paper.get("paperTitle") or "(untitled)")


def drill_1_search_papers(client: MCPClient, tool_name: str) -> None:
    print("=" * 80)
    print("DRILL 1: search_papers - recent LLM agent papers")
    print("=" * 80)

    arguments = {
        "fields": "title,abstract,year,authors",
        "limit": 5,
    }
    if tool_name == "search_papers_by_relevance":
        arguments["keyword"] = "large language model agents"
    else:
        arguments["query"] = "large language model agents"

    text = client.call_tool_text(tool_name, arguments, request_id=2)

    rows = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(text)]
    if not rows:
        print("No paper rows parsed from response. Raw response:\n")
        print(text[:2000])
        return

    for i, paper in enumerate(rows[:5], 1):
        print(f"{i}. {_title_of(paper)} ({_year_of(paper) if _year_of(paper) > 0 else 'unknown year'})")


def drill_2_get_citations(client: MCPClient, tool_name: str) -> None:
    print("\n" + "=" * 80)
    print("DRILL 2: get_citations - BERT citations since 2023")
    print("=" * 80)

    text = client.call_tool_text(
        tool_name,
        {
            "paper_id": "ARXIV:1810.04805",
            "fields": "title,year,authors,citationCount",
            "limit": 10,
            "publication_date_range": "2023-01-01:",
        },
        request_id=3,
    )

    rows = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(text)]
    print(f"Total parsed citing papers: {len(rows)}")

    for i, paper in enumerate(rows[:5], 1):
        print(f"{i}. {_title_of(paper)} ({_year_of(paper) if _year_of(paper) > 0 else 'unknown year'})")


def _merge_reference_metadata(
    raw_references: list[dict[str, Any]],
    enriched_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(row.get("paperId")): row
        for row in enriched_rows
        if row.get("paperId") is not None
    }

    merged: list[dict[str, Any]] = []
    for ref in raw_references:
        paper_id = ref.get("paperId")
        if paper_id is None:
            merged.append(ref)
            continue
        paper_id_str = str(paper_id)
        enriched = by_id.get(paper_id_str, {})
        combined = dict(ref)
        combined.update(enriched)
        merged.append(combined)

    return merged


def drill_3_get_references(
    client: MCPClient,
    references_tool_name: str,
    batch_tool_name: str | None,
) -> None:
    print("\n" + "=" * 80)
    print("DRILL 3: get_references - ReAct references sorted by year")
    print("=" * 80)

    if references_tool_name == "get_references":
        text = client.call_tool_text(
            references_tool_name,
            {
                "paper_id": "ARXIV:2210.03629",
                "fields": "title,year,authors,citationCount",
                "limit": 50,
            },
            request_id=4,
        )
        rows = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(text)]
    else:
        print("Using get_paper(fields=references) fallback because get_references is not exposed.")
        text = client.call_tool_text(
            references_tool_name,
            {
                "paper_id": "ARXIV:2210.03629",
                "fields": "title,year,references",
            },
            request_id=4,
        )
        raw_references = [
            ref
            for ref in normalize_rows_from_tool_text(text)
            if isinstance(ref, dict) and ref.get("paperId") and ref.get("title")
        ]
        reference_ids = [
            str(ref.get("paperId"))
            for ref in raw_references
        ]

        enriched_rows: list[dict[str, Any]] = []
        if batch_tool_name and reference_ids:
            batch_text = client.call_tool_text(
                batch_tool_name,
                {
                    "ids": reference_ids[:100],
                    "fields": "title,year,authors,citationCount",
                },
                request_id=5,
            )
            enriched_rows = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(batch_text)]

        rows = _merge_reference_metadata(raw_references, enriched_rows)

    rows.sort(key=lambda p: (_year_of(p) <= 0, _year_of(p), _title_of(p).lower()))

    if not rows:
        print("No references parsed from response. Raw response:\n")
        print(text[:2000])
        return

    for paper in rows:
        year = _year_of(paper)
        year_text = str(year) if year > 0 else "unknown year"
        print(f"- {year_text}: {_title_of(paper)}")


def main() -> None:
    client = MCPClient()
    tools = client.list_tools()

    search_tool = tool_name_or_none(tools, ["search_papers", "search_papers_by_relevance"])
    citations_tool = tool_name_or_none(tools, ["get_citations"])
    references_tool = tool_name_or_none(tools, ["get_references", "get_paper"])
    batch_tool = tool_name_or_none(tools, ["get_paper_batch"])

    if not search_tool:
        raise RuntimeError("Could not find a search tool (search_papers or equivalent) from tools/list")
    if not citations_tool:
        raise RuntimeError("Could not find get_citations from tools/list")
    if not references_tool:
        raise RuntimeError("Could not find a references-capable tool (get_references or get_paper) from tools/list")

    print(
        f"Using tools: search={search_tool}, citations={citations_tool}, "
        f"references={references_tool}, batch={batch_tool or 'none'}\n"
    )

    drill_1_search_papers(client, search_tool)
    drill_2_get_citations(client, citations_tool)
    drill_3_get_references(client, references_tool, batch_tool)


if __name__ == "__main__":
    main()
