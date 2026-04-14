from __future__ import annotations

from typing import Any

from ..config import ConnectorConfig
from ..models import K8sGPTFinding
from ..state import WorkflowState
from .mcp_stdio import StdioMCPClient


class K8sGPTMCPAnalyzer:
    def __init__(self, config: ConnectorConfig):
        args = list(config.k8sgpt_mcp_args or ["serve", "--mcp"])
        if not any(arg in {"--backend", "-b"} for arg in args):
            args.extend(["--backend", config.k8sgpt_backend])
        self.client = StdioMCPClient(
            command=config.k8sgpt_mcp_command,
            args=args,
            env=config.mcp_env(),
        )

    def _normalize_findings(self, payload: Any) -> list[K8sGPTFinding]:
        if isinstance(payload, dict):
            candidates = payload.get("results") or payload.get("items") or payload.get("analysis") or payload
            if isinstance(candidates, list):
                findings: list[K8sGPTFinding] = []
                for item in candidates:
                    if not isinstance(item, dict):
                        continue
                    findings.append(
                        K8sGPTFinding(
                            resource=str(
                                item.get("resource")
                                or item.get("name")
                                or item.get("kind")
                                or "unknown-resource"
                            ),
                            severity=str(item.get("severity", item.get("status", "info"))),
                            description=str(
                                item.get("description")
                                or item.get("text")
                                or item.get("details")
                                or item
                            ),
                            recommendation=str(
                                item.get("recommendation")
                                or item.get("solution")
                                or item.get("advice")
                                or ""
                            ),
                        )
                    )
                if findings:
                    return findings
            if "text" in payload:
                return [
                    K8sGPTFinding(
                        resource="cluster",
                        severity="info",
                        description=str(payload["text"]),
                        recommendation="Review the analyzer output and continue triage.",
                    )
                ]
        if isinstance(payload, list):
            return [
                K8sGPTFinding(
                    resource="cluster",
                    severity="info",
                    description=str(item),
                    recommendation="Review the analyzer output and continue triage.",
                )
                for item in payload
            ]
        return []

    def analyze(self, state: WorkflowState) -> list[K8sGPTFinding]:
        arguments: dict[str, Any] = {"namespace": state.get("namespace"), "explain": False}
        try:
            payload = self.client.call_tool("analyze", arguments)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            return [
                K8sGPTFinding(
                    resource="k8sgpt",
                    severity="warning",
                    description=f"K8sGPT MCP analysis unavailable: {message}",
                    recommendation=(
                        "Configure K8sGPT authentication or provider settings, then rerun the workflow. "
                        "Cluster triage continued with kubectl, RAG, and Org-roam context only."
                    ),
                )
            ]
        return self._normalize_findings(payload)
