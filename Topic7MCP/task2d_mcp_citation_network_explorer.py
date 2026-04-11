"""Exercise D (MCP): autonomous citation network explorer agent."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mcp_common import (
    MCPClient,
    normalize_rows_from_tool_text,
    parse_json_maybe,
    tool_name_or_none,
    unwrap_embedded_paper,
)

load_dotenv()


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _paper_title(p: dict[str, Any]) -> str:
    return str(p.get("title") or p.get("paperTitle") or "(untitled)")


def _paper_year(p: dict[str, Any]) -> int:
    return _to_int(p.get("year"), default=0)


def _paper_citations(p: dict[str, Any]) -> int:
    return _to_int(p.get("citationCount"), default=0)


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
        combined = dict(ref)
        combined.update(by_id.get(str(paper_id), {}))
        merged.append(combined)

    return merged


def _extract_seed_paper(tool_text: str) -> dict[str, Any]:
    parsed = parse_json_maybe(tool_text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("data"), dict):
            return parsed["data"]
        if isinstance(parsed.get("paper"), dict):
            return parsed["paper"]
        return parsed

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]

    return {}


def _author_name(author: dict[str, Any]) -> str:
    if not isinstance(author, dict):
        return "Unknown Author"
    return str(author.get("name") or author.get("authorName") or "Unknown Author")


def _author_id(author: dict[str, Any]) -> str | None:
    if not isinstance(author, dict):
        return None
    for key in ("authorId", "author_id", "id"):
        value = author.get(key)
        if value is not None:
            return str(value)
    nested = author.get("author")
    if isinstance(nested, dict):
        for key in ("authorId", "author_id", "id"):
            value = nested.get(key)
            if value is not None:
                return str(value)
    return None


def _call_first_valid(
    client: MCPClient,
    tool_name: str,
    argument_options: list[dict[str, Any]],
    request_id: int,
) -> str:
    last_exc: Exception | None = None
    for args in argument_options:
        try:
            return client.call_tool_text(tool_name, args, request_id=request_id)
        except Exception as exc:  # nosec
            last_exc = exc
            continue
    raise RuntimeError(f"All argument variants failed for {tool_name}: {last_exc}")


def _build_summary_with_llm(model: str, seed: dict[str, Any], references: list[dict], citations: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        abstract = seed.get("abstract") or "No abstract available."
        return str(abstract)[:1000]

    client = OpenAI(api_key=api_key)

    prompt = {
        "seed_title": _paper_title(seed),
        "seed_abstract": seed.get("abstract", ""),
        "seed_year": seed.get("year"),
        "top_references": [
            {
                "title": _paper_title(r),
                "year": _paper_year(r),
                "citation_count": _paper_citations(r),
                "abstract": r.get("abstract", ""),
            }
            for r in references[:5]
        ],
        "recent_citations": [
            {
                "title": _paper_title(c),
                "year": _paper_year(c),
                "citation_count": _paper_citations(c),
                "abstract": c.get("abstract", ""),
            }
            for c in citations[:5]
        ],
    }

    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Write exactly one compact paragraph summarizing the seed paper's "
                    "core idea and its citation neighborhood trends."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=True),
            },
        ],
    )
    return completion.choices[0].message.content or "Summary unavailable."


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous citation network explorer")
    parser.add_argument("paper_id", nargs="?", default="ARXIV:2210.03629")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    client = MCPClient()
    tools = client.list_tools()

    get_paper_tool = tool_name_or_none(tools, ["get_paper"])
    get_refs_tool = tool_name_or_none(tools, ["get_references", "get_paper"])
    get_paper_batch_tool = tool_name_or_none(tools, ["get_paper_batch"])
    get_citations_tool = tool_name_or_none(tools, ["get_citations"])
    get_author_papers_tool = tool_name_or_none(tools, ["get_author_papers"])

    if not get_paper_tool:
        raise RuntimeError("get_paper tool not found in tools/list")
    if not get_refs_tool:
        raise RuntimeError("Could not find a references-capable tool (get_references or get_paper)")
    if not get_citations_tool:
        raise RuntimeError("get_citations tool not found in tools/list")

    seed_text = _call_first_valid(
        client,
        get_paper_tool,
        [
            {
                "paper_id": args.paper_id,
                "fields": "title,abstract,year,authors,fieldsOfStudy,citationCount,referenceCount,externalIds",
            },
            {"paper_id": args.paper_id},
            {"id": args.paper_id},
        ],
        request_id=10,
    )
    seed = _extract_seed_paper(seed_text)

    if get_refs_tool == "get_references":
        refs_text = _call_first_valid(
            client,
            get_refs_tool,
            [
                {
                    "paper_id": args.paper_id,
                    "fields": "title,abstract,year,authors,citationCount,paperId,externalIds",
                    "limit": 80,
                },
                {"paper_id": args.paper_id, "limit": 80},
                {"id": args.paper_id, "limit": 80},
            ],
            request_id=11,
        )
        refs = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(refs_text)]
    else:
        refs_text = _call_first_valid(
            client,
            get_paper_tool,
            [
                {
                    "paper_id": args.paper_id,
                    "fields": "title,year,references",
                },
                {"paper_id": args.paper_id},
                {"id": args.paper_id},
            ],
            request_id=11,
        )
        raw_references = [
            ref
            for ref in normalize_rows_from_tool_text(refs_text)
            if isinstance(ref, dict) and ref.get("paperId") and ref.get("title")
        ]
        ref_ids = [str(ref["paperId"]) for ref in raw_references[:100]]
        enriched_rows: list[dict[str, Any]] = []

        if get_paper_batch_tool and ref_ids:
            enriched_text = _call_first_valid(
                client,
                get_paper_batch_tool,
                [
                    {
                        "ids": ref_ids,
                        "fields": "title,abstract,year,authors,citationCount,paperId,externalIds",
                    }
                ],
                request_id=13,
            )
            enriched_rows = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(enriched_text)]

        refs = _merge_reference_metadata(raw_references, enriched_rows)

    refs.sort(key=lambda p: _paper_citations(p), reverse=True)
    top_refs = refs[:5]

    current_year = dt.date.today().year
    min_year = current_year - 2
    citations_text = _call_first_valid(
        client,
        get_citations_tool,
        [
            {
                "paper_id": args.paper_id,
                "fields": "title,abstract,year,authors,citationCount,paperId,externalIds",
                "limit": 60,
                "publication_date_range": f"{min_year}-01-01:",
            },
            {"paper_id": args.paper_id, "limit": 60},
            {"id": args.paper_id, "limit": 60},
        ],
        request_id=12,
    )
    citations = [unwrap_embedded_paper(r) for r in normalize_rows_from_tool_text(citations_text)]
    citations.sort(key=lambda p: (_paper_year(p), _paper_citations(p)), reverse=True)
    recent_citations = citations[:5]

    author_profiles: list[dict[str, str]] = []
    seed_authors = seed.get("authors") or []
    seed_title = _paper_title(seed)

    if get_author_papers_tool and isinstance(seed_authors, list):
        for idx, author in enumerate(seed_authors, start=1):
            name = _author_name(author)
            aid = _author_id(author)
            if not aid:
                author_profiles.append(
                    {
                        "author": name,
                        "note": "No author id in seed metadata; skipped automated lookup.",
                    }
                )
                continue

            try:
                author_text = _call_first_valid(
                    client,
                    get_author_papers_tool,
                    [
                        {
                            "author_id": aid,
                            "fields": "title,year,citationCount,paperId,externalIds",
                            "limit": 25,
                        },
                        {"authorId": aid, "limit": 25},
                        {"id": aid, "limit": 25},
                    ],
                    request_id=100 + idx,
                )
                author_papers = [
                    unwrap_embedded_paper(r)
                    for r in normalize_rows_from_tool_text(author_text)
                    if _paper_title(unwrap_embedded_paper(r)).lower() != seed_title.lower()
                ]
                author_papers.sort(key=lambda p: _paper_citations(p), reverse=True)

                if author_papers:
                    best = author_papers[0]
                    author_profiles.append(
                        {
                            "author": name,
                            "paper": _paper_title(best),
                            "year": str(_paper_year(best) or "?"),
                            "citation_count": str(_paper_citations(best)),
                        }
                    )
                else:
                    author_profiles.append(
                        {
                            "author": name,
                            "note": "No additional papers found.",
                        }
                    )
            except Exception as exc:  # nosec
                author_profiles.append(
                    {
                        "author": name,
                        "note": f"Author lookup failed: {exc}",
                    }
                )

    summary = _build_summary_with_llm(args.model, seed, top_refs, recent_citations)

    print(f"# Citation Neighborhood Report: {seed_title}")
    print()
    print(f"- Seed ID: `{args.paper_id}`")
    print(f"- Year: {_paper_year(seed) or 'unknown'}")
    print(f"- Fields of Study: {', '.join(seed.get('fieldsOfStudy', []) or []) or 'unknown'}")
    print(f"- Authors: {', '.join(_author_name(a) for a in seed_authors) or 'unknown'}")
    print()
    print("## One-Paragraph Summary")
    print(summary.strip())
    print()

    print("## Foundational Works")
    if top_refs:
        for ref in top_refs:
            print(
                f"- **{_paper_title(ref)}** ({_paper_year(ref) or 'unknown'}) "
                f"- citations: {_paper_citations(ref)}"
            )
    else:
        print("- No references returned by MCP.")
    print()

    print("## Recent Developments")
    if recent_citations:
        for citation in recent_citations:
            print(
                f"- **{_paper_title(citation)}** ({_paper_year(citation) or 'unknown'}) "
                f"- citations: {_paper_citations(citation)}"
            )
    else:
        print("- No recent citations returned by MCP.")
    print()

    print("## Author Profiles")
    if author_profiles:
        for profile in author_profiles:
            if "paper" in profile:
                print(
                    f"- **{profile['author']}**: {profile['paper']} "
                    f"({profile['year']}), citations={profile['citation_count']}"
                )
            else:
                print(f"- **{profile['author']}**: {profile.get('note', 'No data')}" )
    else:
        print("- No author profile data available.")


if __name__ == "__main__":
    main()
