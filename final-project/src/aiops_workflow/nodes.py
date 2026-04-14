from __future__ import annotations

from typing import Callable, Literal

from .runtime import WorkflowRuntime
from .state import WorkflowState


NodeHandler = Callable[[WorkflowState], WorkflowState]


def _append_trace(state: WorkflowState, message: str) -> list[str]:
    return [*state.get("trace", []), message]


def incident_monitor_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        incident = runtime.incident_monitor.detect(state)
        return {
            **incident,
            "trace": _append_trace(state, "Incident monitor classified the trigger and target workload."),
        }

    return handle


def context_collector_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        evidence = runtime.kubernetes.collect_context(state)
        return {
            "evidence": evidence,
            "trace": _append_trace(state, "Context collector gathered events, logs, metrics, and manifests."),
        }

    return handle


def k8s_rag_retriever_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        passages = [passage.to_dict() for passage in runtime.rag.retrieve(state)]
        return {
            "retrieved_runbook_passages": passages,
            "trace": _append_trace(state, "Kubernetes RAG retriever fetched the most relevant runbook passages."),
        }

    return handle


def org_roam_browser_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        nodes = [node.to_dict() for node in runtime.org_roam.browse(state)]
        return {
            "retrieved_org_roam_nodes": nodes,
            "trace": _append_trace(state, "Org-roam browser explored note context and backlinks."),
        }

    return handle


def k8sgpt_tool_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        findings = [finding.to_dict() for finding in runtime.k8sgpt.analyze(state)]
        return {
            "k8sgpt_findings": findings,
            "trace": _append_trace(state, "K8sGPT MCP analysis added live cluster findings."),
        }

    return handle


def diagnosis_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        diagnosis = runtime.reasoner.diagnose(state).to_dict()
        return {
            "diagnosis": diagnosis,
            "trace": _append_trace(state, "Diagnosis agent combined cluster evidence, RAG, Org-roam, and K8sGPT signals."),
        }

    return handle


def remediation_planner_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        plan = [action.to_dict() for action in runtime.reasoner.plan(state)]
        return {
            "plan": plan,
            "trace": _append_trace(state, "Remediation planner generated ordered actions and rollback steps."),
        }

    return handle


def safety_policy_node(_: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        plan = state.get("plan", [])
        approval_required = any(step["requires_approval"] for step in plan)
        risk_level = "low"
        if any(step["risk"] == "high" for step in plan):
            risk_level = "high"
        elif any(step["risk"] == "medium" for step in plan):
            risk_level = "medium"

        return {
            "approval_required": approval_required,
            "risk_level": risk_level,
            "approval_status": "required" if approval_required else "not_needed",
            "trace": _append_trace(state, "Safety policy evaluated risk and approval requirements."),
        }

    return handle


def human_approval_node(_: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        mode = state.get("approval_mode", "auto")
        if mode == "auto":
            status = "approved"
            trace = "Human approval node auto-approved the plan for demo execution."
        elif mode == "deny":
            status = "denied"
            trace = "Human approval node denied the plan."
        else:
            status = "pending"
            trace = "Human approval node is waiting for a human decision."

        return {
            "approval_status": status,
            "trace": _append_trace(state, trace),
        }

    return handle


def executor_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        execution_result = runtime.kubernetes.execute_plan(state).to_dict()
        return {
            "execution_result": execution_result,
            "trace": _append_trace(state, "Executor ran the approved plan."),
        }

    return handle


def verifier_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        verification = runtime.kubernetes.verify(state).to_dict()
        return {
            "verification": verification,
            "trace": _append_trace(state, "Verifier checked whether the workload recovered."),
        }

    return handle


def reporter_node(runtime: WorkflowRuntime) -> NodeHandler:
    def handle(state: WorkflowState) -> WorkflowState:
        report = runtime.reasoner.summarize(state)
        return {
            "final_report": report,
            "trace": _append_trace(state, "Reporter produced the incident summary."),
        }

    return handle


def route_after_safety(state: WorkflowState) -> Literal["human_approval", "executor"]:
    if state.get("approval_required"):
        return "human_approval"
    return "executor"


def route_after_human_approval(state: WorkflowState) -> Literal["executor", "reporter"]:
    if state.get("approval_status") == "approved":
        return "executor"
    return "reporter"
