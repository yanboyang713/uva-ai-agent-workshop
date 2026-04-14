from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict


class WorkflowState(TypedDict, total=False):
    incident_id: str
    trigger: str
    namespace: str
    workload: str
    severity: str
    evidence: dict[str, list[str]]
    retrieved_runbook_passages: list[dict[str, Any]]
    retrieved_org_roam_nodes: list[dict[str, Any]]
    k8sgpt_findings: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    plan: list[dict[str, Any]]
    risk_level: str
    approval_required: bool
    approval_mode: str
    approval_status: str
    execution_result: dict[str, Any]
    verification: dict[str, Any]
    final_report: str
    trace: list[str]
    user_goal: NotRequired[str]


def build_initial_state(
    *,
    incident_id: str = "inc-local-001",
    trigger: str = "CrashLoopBackOff",
    namespace: str = "payments",
    workload: str = "api-server",
    severity: str = "warning",
    approval_mode: str = "auto",
) -> WorkflowState:
    return WorkflowState(
        incident_id=incident_id,
        trigger=trigger,
        namespace=namespace,
        workload=workload,
        severity=severity,
        evidence={"events": [], "logs": [], "metrics": [], "manifests": []},
        retrieved_runbook_passages=[],
        retrieved_org_roam_nodes=[],
        k8sgpt_findings=[],
        plan=[],
        risk_level="unknown",
        approval_required=False,
        approval_mode=approval_mode,
        approval_status="not_needed",
        trace=[],
    )
