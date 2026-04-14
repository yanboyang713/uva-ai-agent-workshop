from __future__ import annotations

from typing import Any

from ..config import ConnectorConfig
from ..models import OrgRoamNodeContext
from ..state import WorkflowState
from .mcp_stdio import StdioMCPClient


class OrgRoamMCPBrowser:
    def __init__(self, config: ConnectorConfig):
        self.client = StdioMCPClient(
            command=config.org_roam_mcp_command,
            args=config.org_roam_mcp_args or [],
            env=config.mcp_env(),
        )

    def _search_candidates(self, state: WorkflowState) -> list[dict[str, Any]]:
        queries = [
            f"{state.get('namespace', '')} {state.get('workload', '')}".strip(),
            state.get("workload", ""),
            state.get("trigger", ""),
            state.get("namespace", ""),
        ]
        seen: set[str] = set()
        aggregated: list[dict[str, Any]] = []
        for query in queries:
            if not query:
                continue
            payload = self.client.call_tool("search_nodes", {"query": query, "limit": 3})
            for result in payload.get("results", []):
                node_id = result.get("id")
                if isinstance(node_id, str) and node_id not in seen:
                    seen.add(node_id)
                    aggregated.append(result)
        return aggregated[:3]

    def browse(self, state: WorkflowState) -> list[OrgRoamNodeContext]:
        contexts: list[OrgRoamNodeContext] = []
        for candidate in self._search_candidates(state):
            node_id = candidate.get("id")
            if not isinstance(node_id, str):
                continue
            detail = self.client.call_tool("get_node", {"node_id": node_id})
            backlinks_payload = self.client.call_tool("get_backlinks", {"node_id": node_id})
            backlinks = [
                item.get("source_title", item.get("source_id", "unknown"))
                for item in backlinks_payload.get("backlinks", [])
                if isinstance(item, dict)
            ]
            contexts.append(
                OrgRoamNodeContext(
                    node_id=node_id,
                    title=detail.get("title", candidate.get("title", node_id)),
                    content=detail.get("content", ""),
                    backlinks=[str(link) for link in backlinks],
                    tags=[str(tag) for tag in detail.get("tags", candidate.get("tags", []))],
                )
            )
        return contexts
