#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from aiops_workflow.config import ConnectorConfig
from aiops_workflow.connectors.mcp_stdio import StdioMCPClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Org-roam MCP by searching nodes and reading one node."
    )
    parser.add_argument(
        "--query",
        default="retrieval augmented generation",
        help="Search query used when --node-id is not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of search results to show.",
    )
    parser.add_argument(
        "--node-id",
        help="Read a specific node ID directly instead of searching first.",
    )
    parser.add_argument(
        "--print-content",
        action="store_true",
        help="Print the full node content instead of truncating it.",
    )
    return parser


def truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def build_client() -> StdioMCPClient:
    config = ConnectorConfig.from_env()
    print("Using config:")
    print(json.dumps(
        {
            "org_roam_mcp_command": config.org_roam_mcp_command,
            "org_roam_mcp_args": config.org_roam_mcp_args or [],
            "org_roam_db_path": config.org_roam_db_path,
            "org_roam_dir": config.org_roam_dir,
        },
        indent=2,
    ))
    return StdioMCPClient(
        command=config.org_roam_mcp_command,
        args=config.org_roam_mcp_args or [],
        env=config.mcp_env(),
    )


def search_nodes(client: StdioMCPClient, query: str, limit: int) -> dict[str, Any]:
    return client.call_tool("search_nodes", {"query": query, "limit": limit})


def get_node(client: StdioMCPClient, node_id: str) -> dict[str, Any]:
    return client.call_tool("get_node", {"node_id": node_id})


def main() -> int:
    args = build_parser().parse_args()
    client = build_client()

    try:
        if args.node_id:
            node_id = args.node_id
            print(f"\nReading node: {node_id}")
        else:
            search_result = search_nodes(client, args.query, args.limit)
            print(f"\nSearch result for query: {args.query!r}")
            print(json.dumps(search_result, indent=2))

            results = search_result.get("results", [])
            if not results:
                print("\nNo nodes matched the query.")
                return 1

            node_id = results[0]["id"]
            print(f"\nReading first result node_id: {node_id}")

        node = get_node(client, node_id)
        node_to_print = dict(node)
        content = node_to_print.get("content", "")
        if isinstance(content, str) and not args.print_content:
            node_to_print["content"] = truncate(content)

        print(json.dumps(node_to_print, indent=2))
        return 0
    except Exception as exc:
        print(f"\nOrg-roam MCP test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
