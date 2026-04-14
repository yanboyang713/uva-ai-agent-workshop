from __future__ import annotations

import sys
import time
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from .nodes import (
    context_collector_node,
    diagnosis_node,
    executor_node,
    human_approval_node,
    incident_monitor_node,
    k8s_rag_retriever_node,
    k8sgpt_tool_node,
    org_roam_browser_node,
    remediation_planner_node,
    reporter_node,
    route_after_human_approval,
    route_after_safety,
    safety_policy_node,
    verifier_node,
)
from .runtime import WorkflowRuntime
from .state import WorkflowState


def _with_progress(
    name: str,
    handler: Callable[[WorkflowState], WorkflowState],
    *,
    show_progress: bool,
) -> Callable[[WorkflowState], WorkflowState]:
    if not show_progress:
        return handler

    def wrapped(state: WorkflowState) -> WorkflowState:
        start = time.monotonic()
        print(f"[progress] start {name}", file=sys.stderr, flush=True)
        result = handler(state)
        elapsed = time.monotonic() - start
        print(f"[progress] done  {name} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        return result

    return wrapped


def build_workflow(runtime: WorkflowRuntime, *, show_progress: bool = False):
    builder = StateGraph(WorkflowState)

    builder.add_node(
        "incident_monitor",
        _with_progress("incident_monitor", incident_monitor_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "context_collector",
        _with_progress("context_collector", context_collector_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "k8s_rag_retriever",
        _with_progress("k8s_rag_retriever", k8s_rag_retriever_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "org_roam_browser",
        _with_progress("org_roam_browser", org_roam_browser_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "k8sgpt_tool",
        _with_progress("k8sgpt_tool", k8sgpt_tool_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "diagnosis",
        _with_progress("diagnosis", diagnosis_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "remediation_planner",
        _with_progress("remediation_planner", remediation_planner_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "safety_policy",
        _with_progress("safety_policy", safety_policy_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "human_approval",
        _with_progress("human_approval", human_approval_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "executor",
        _with_progress("executor", executor_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "verifier",
        _with_progress("verifier", verifier_node(runtime), show_progress=show_progress),
    )
    builder.add_node(
        "reporter",
        _with_progress("reporter", reporter_node(runtime), show_progress=show_progress),
    )

    builder.add_edge(START, "incident_monitor")
    builder.add_edge("incident_monitor", "context_collector")
    builder.add_edge("context_collector", "k8s_rag_retriever")
    builder.add_edge("k8s_rag_retriever", "org_roam_browser")
    builder.add_edge("org_roam_browser", "k8sgpt_tool")
    builder.add_edge("k8sgpt_tool", "diagnosis")
    builder.add_edge("diagnosis", "remediation_planner")
    builder.add_edge("remediation_planner", "safety_policy")
    builder.add_conditional_edges(
        "safety_policy",
        route_after_safety,
        {
            "human_approval": "human_approval",
            "executor": "executor",
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {
            "executor": "executor",
            "reporter": "reporter",
        },
    )
    builder.add_edge("executor", "verifier")
    builder.add_edge("verifier", "reporter")
    builder.add_edge("reporter", END)

    return builder.compile()


def run_workflow(
    runtime: WorkflowRuntime,
    initial_state: WorkflowState,
    *,
    show_progress: bool = False,
) -> WorkflowState:
    return build_workflow(runtime, show_progress=show_progress).invoke(initial_state)
