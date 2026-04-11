"""
Exercise A (MCP): Discover Asta tools via tools/list.

Answers required by the lesson plan:
- For "transformer attention mechanisms", use `search_papers` (or `search_papers_by_relevance` if that is the
  server's equivalent name).
- To find who else publishes in the same area as a specific author, combine
  `search_authors_by_name` + `get_author_papers`.
"""

from __future__ import annotations

import argparse

from mcp_common import MCPClient


def summarize_schema(tool: dict) -> tuple[list[str], list[str]]:
    schema = tool.get("inputSchema") or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    required_params: list[str] = []
    optional_params: list[str] = []

    for name in sorted(properties.keys()):
        field = properties.get(name) or {}
        field_type = field.get("type", "any")
        entry = f"{name} ({field_type})"
        if name in required:
            required_params.append(entry)
        else:
            optional_params.append(entry)

    return required_params, optional_params


def main() -> None:
    parser = argparse.ArgumentParser(description="List MCP tools from Asta and summarize required params.")
    parser.add_argument(
        "--endpoint",
        default="https://asta-tools.allen.ai/mcp/v1",
        help="MCP endpoint URL (default: Asta)",
    )
    args = parser.parse_args()

    client = MCPClient(endpoint=args.endpoint)
    tools = client.list_tools()

    if not tools:
        print("No tools were returned by tools/list.")
        return

    print(f"Discovered {len(tools)} tools from {args.endpoint}\n")

    for tool in tools:
        name = tool.get("name", "<unknown>")
        description = (tool.get("description") or "").strip().splitlines()[0]
        required, optional = summarize_schema(tool)

        print(f"Tool: {name}")
        print(f"  Description: {description or '(no description)'}")
        print(f"  Required: {', '.join(required) if required else 'None'}")
        print(f"  Optional: {', '.join(optional) if optional else 'None'}")
        print()


if __name__ == "__main__":
    main()
