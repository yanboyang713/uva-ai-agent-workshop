from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .config import ConnectorConfig
from .models import (
    Diagnosis,
    ExecutionResult,
    K8sGPTFinding,
    OrgRoamNodeContext,
    PlannedAction,
    RetrievedPassage,
    VerificationResult,
)
from .state import WorkflowState

if TYPE_CHECKING:
    from .connectors import (
        HaystackQdrantRetriever,
        K8sGPTMCPAnalyzer,
        KubectlKubernetesClient,
        OllamaClient,
        OllamaReasoner,
        OrgRoamMCPBrowser,
    )


class IncidentMonitor(Protocol):
    def detect(self, state: WorkflowState) -> dict[str, str]: ...


class KubernetesClient(Protocol):
    def collect_context(self, state: WorkflowState) -> dict[str, list[str]]: ...

    def execute_plan(self, state: WorkflowState) -> ExecutionResult: ...

    def verify(self, state: WorkflowState) -> VerificationResult: ...


class K8sRAGRetriever(Protocol):
    def retrieve(self, state: WorkflowState) -> list[RetrievedPassage]: ...


class OrgRoamBrowser(Protocol):
    def browse(self, state: WorkflowState) -> list[OrgRoamNodeContext]: ...


class K8sGPTAnalyzer(Protocol):
    def analyze(self, state: WorkflowState) -> list[K8sGPTFinding]: ...


class Reasoner(Protocol):
    def diagnose(self, state: WorkflowState) -> Diagnosis: ...

    def plan(self, state: WorkflowState) -> list[PlannedAction]: ...

    def summarize(self, state: WorkflowState) -> str: ...


@dataclass
class WorkflowRuntime:
    incident_monitor: IncidentMonitor
    kubernetes: KubernetesClient
    rag: K8sRAGRetriever
    org_roam: OrgRoamBrowser
    k8sgpt: K8sGPTAnalyzer
    reasoner: Reasoner


class DemoIncidentMonitor:
    def detect(self, state: WorkflowState) -> dict[str, str]:
        return {
            "incident_id": state["incident_id"],
            "trigger": state["trigger"],
            "namespace": state["namespace"],
            "workload": state["workload"],
            "severity": state.get("severity", "warning"),
        }


class DemoKubernetesClient:
    def collect_context(self, state: WorkflowState) -> dict[str, list[str]]:
        trigger = state["trigger"]
        namespace = state["namespace"]
        workload = state["workload"]

        if trigger == "CrashLoopBackOff":
            return {
                "events": [
                    f"Pod {workload}-7d9c failed: ConfigMap 'api-config' not found",
                    f"Deployment/{workload} in namespace/{namespace} is not progressing",
                ],
                "logs": [
                    "startup failed: could not load required configuration",
                    "fatal error: config map api-config missing",
                ],
                "metrics": [
                    "restart_count=6",
                    "ready_replicas=0",
                ],
                "manifests": [
                    "envFrom.configMapRef.name=api-config",
                    "readinessProbe.path=/healthz",
                ],
            }

        if trigger == "ImagePullBackOff":
            return {
                "events": [
                    f"Failed to pull image ghcr.io/example/{workload}:bad-tag",
                    "Back-off pulling image",
                ],
                "logs": [],
                "metrics": ["ready_replicas=0"],
                "manifests": [f"image=ghcr.io/example/{workload}:bad-tag"],
            }

        if trigger == "Pending":
            return {
                "events": [
                    "0/1 nodes are available: insufficient memory",
                    f"Pod {workload}-0 is Pending",
                ],
                "logs": [],
                "metrics": ["cluster_memory_pressure=high"],
                "manifests": ["resources.requests.memory=4Gi"],
            }

        return {
            "events": [f"Detected {trigger} for {workload} in {namespace}"],
            "logs": ["No specialized demo logs were configured for this trigger"],
            "metrics": [],
            "manifests": [],
        }

    def execute_plan(self, state: WorkflowState) -> ExecutionResult:
        commands = [item["command"] for item in state.get("plan", [])]
        if not commands:
            return ExecutionResult(
                status="skipped",
                details="No executable actions were generated",
                commands=[],
            )
        return ExecutionResult(
            status="completed",
            details="Demo executor simulated the approved remediation steps",
            commands=commands,
        )

    def verify(self, state: WorkflowState) -> VerificationResult:
        trigger = state["trigger"]
        if trigger == "CrashLoopBackOff":
            return VerificationResult(
                success=True,
                summary="Pods recovered after configuration was restored",
                signals=[
                    "ready_replicas=1",
                    "restart_count stabilized",
                    "rollout status reports success",
                ],
            )
        return VerificationResult(
            success=True,
            summary="Demo verification marked the workload as recovered",
            signals=["no active alerts", "workload available"],
        )


class DemoK8sRAGRetriever:
    def retrieve(self, state: WorkflowState) -> list[RetrievedPassage]:
        trigger = state["trigger"]
        workload = state["workload"]
        namespace = state["namespace"]

        if trigger == "CrashLoopBackOff":
            return [
                RetrievedPassage(
                    source_id="runbook-001",
                    title="Runbook: Missing ConfigMap after deployment",
                    content=(
                        "When a workload enters CrashLoopBackOff immediately after rollout, "
                        "verify envFrom configMapRef names before restarting pods."
                    ),
                    score=0.93,
                    metadata={"namespace": namespace, "workload": workload, "kind": "runbook"},
                )
            ]

        if trigger == "ImagePullBackOff":
            return [
                RetrievedPassage(
                    source_id="runbook-002",
                    title="Runbook: Bad image tag or registry auth",
                    content=(
                        "Check the resolved image tag in the deployment manifest and confirm that "
                        "the container registry can serve the image."
                    ),
                    score=0.91,
                    metadata={"namespace": namespace, "workload": workload, "kind": "runbook"},
                )
            ]

        return [
            RetrievedPassage(
                source_id="runbook-generic",
                title="Generic workload incident triage",
                content="Inspect events, describe the workload, and validate recent rollout changes.",
                score=0.74,
                metadata={"namespace": namespace, "workload": workload, "kind": "runbook"},
            )
        ]


class DemoOrgRoamBrowser:
    def browse(self, state: WorkflowState) -> list[OrgRoamNodeContext]:
        trigger = state["trigger"]
        if trigger == "CrashLoopBackOff":
            return [
                OrgRoamNodeContext(
                    node_id="staging-bootstrap-issues",
                    title="Staging bootstrap issues",
                    content=(
                        "Post-release failures in staging were previously caused by missing "
                        "ConfigMaps and Secrets after Helm hook timing issues."
                    ),
                    backlinks=["payments-api-postmortem", "staging-release-checklist"],
                    tags=["kubernetes", "staging", "runbook"],
                )
            ]
        return [
            OrgRoamNodeContext(
                node_id="generic-k8s-notes",
                title="Generic Kubernetes troubleshooting notes",
                content="Check events first, then compare the failure against known rollout problems.",
                backlinks=["weekly-ops-notes"],
                tags=["kubernetes"],
            )
        ]


class DemoK8sGPTAnalyzer:
    def analyze(self, state: WorkflowState) -> list[K8sGPTFinding]:
        trigger = state["trigger"]
        workload = state["workload"]
        if trigger == "CrashLoopBackOff":
            return [
                K8sGPTFinding(
                    resource=f"deployment/{workload}",
                    severity="warning",
                    description="Referenced ConfigMap does not exist in the target namespace",
                    recommendation="Create or restore the missing ConfigMap before restarting pods",
                )
            ]
        if trigger == "ImagePullBackOff":
            return [
                K8sGPTFinding(
                    resource=f"deployment/{workload}",
                    severity="warning",
                    description="Container image cannot be pulled with the current tag",
                    recommendation="Update the deployment to a valid image tag",
                )
            ]
        return [
            K8sGPTFinding(
                resource=f"deployment/{workload}",
                severity="info",
                description="No specialized demo finding was generated",
                recommendation="Continue with standard workload triage",
            )
        ]


class DemoReasoner:
    def diagnose(self, state: WorkflowState) -> Diagnosis:
        trigger = state["trigger"]
        evidence = state["evidence"]
        first_event = evidence.get("events", ["No event data available"])[0]
        rag_title = state.get("retrieved_runbook_passages", [{}])[0].get("title", "No runbook")
        org_title = state.get("retrieved_org_roam_nodes", [{}])[0].get("title", "No note")

        if trigger == "CrashLoopBackOff":
            return Diagnosis(
                summary="The deployment is failing because required runtime configuration is missing.",
                root_cause="A ConfigMap referenced by the workload is absent in the target namespace.",
                confidence="medium",
                evidence=[first_event, rag_title, org_title],
                next_check="Confirm whether the missing ConfigMap should be created by Helm hooks.",
            )

        if trigger == "ImagePullBackOff":
            return Diagnosis(
                summary="The deployment is blocked by a container image resolution failure.",
                root_cause="The manifest references a bad image tag or inaccessible registry artifact.",
                confidence="medium",
                evidence=[first_event, rag_title],
                next_check="Verify the exact image tag resolved in the deployment spec.",
            )

        return Diagnosis(
            summary="The incident needs standard Kubernetes triage.",
            root_cause=f"Trigger {trigger} requires additional environment-specific diagnostics.",
            confidence="low",
            evidence=[first_event],
            next_check="Collect more logs and recent rollout changes before mutating the cluster.",
        )

    def plan(self, state: WorkflowState) -> list[PlannedAction]:
        trigger = state["trigger"]
        workload = state["workload"]

        if trigger == "CrashLoopBackOff":
            return [
                PlannedAction(
                    description="Verify the missing ConfigMap reference in the deployment.",
                    command=f"kubectl -n {state['namespace']} describe deployment/{workload}",
                    risk="low",
                    requires_approval=False,
                    rollback="No rollback required for read-only validation.",
                ),
                PlannedAction(
                    description="Restore the missing ConfigMap from the approved template.",
                    command=f"kubectl -n {state['namespace']} apply -f manifests/{workload}-configmap.yaml",
                    risk="medium",
                    requires_approval=True,
                    rollback=f"kubectl -n {state['namespace']} delete configmap api-config",
                ),
                PlannedAction(
                    description="Restart the deployment to pick up the restored configuration.",
                    command=f"kubectl -n {state['namespace']} rollout restart deployment/{workload}",
                    risk="medium",
                    requires_approval=True,
                    rollback=f"kubectl -n {state['namespace']} rollout undo deployment/{workload}",
                ),
            ]

        if trigger == "ImagePullBackOff":
            return [
                PlannedAction(
                    description="Inspect the deployment image reference.",
                    command=f"kubectl -n {state['namespace']} get deployment/{workload} -o yaml",
                    risk="low",
                    requires_approval=False,
                    rollback="No rollback required for read-only validation.",
                ),
                PlannedAction(
                    description="Patch the deployment to a known good image tag.",
                    command=(
                        f"kubectl -n {state['namespace']} set image deployment/{workload} "
                        f"{workload}=ghcr.io/example/{workload}:stable"
                    ),
                    risk="medium",
                    requires_approval=True,
                    rollback=f"kubectl -n {state['namespace']} rollout undo deployment/{workload}",
                ),
            ]

        return [
            PlannedAction(
                description="Gather additional context before remediation.",
                command=f"kubectl -n {state['namespace']} describe deployment/{workload}",
                risk="low",
                requires_approval=False,
                rollback="No rollback required for read-only validation.",
            )
        ]

    def summarize(self, state: WorkflowState) -> str:
        diagnosis = state.get("diagnosis", {})
        verification = state.get("verification", {})
        execution_result = state.get("execution_result", {})
        org_titles = [node["title"] for node in state.get("retrieved_org_roam_nodes", [])]
        runbook_titles = [passage["title"] for passage in state.get("retrieved_runbook_passages", [])]

        lines = [
            f"Incident: {state.get('incident_id', 'unknown')}",
            f"Trigger: {state.get('trigger', 'unknown')}",
            f"Namespace / Workload: {state.get('namespace', 'unknown')} / {state.get('workload', 'unknown')}",
            f"Diagnosis: {diagnosis.get('root_cause', 'not available')}",
            f"Confidence: {diagnosis.get('confidence', 'unknown')}",
            f"Runbook evidence: {', '.join(runbook_titles) if runbook_titles else 'none'}",
            f"Org-roam context: {', '.join(org_titles) if org_titles else 'none'}",
            f"Approval status: {state.get('approval_status', 'unknown')}",
            f"Execution status: {execution_result.get('status', 'not executed')}",
            f"Verification: {verification.get('summary', 'not verified')}",
        ]
        return "\n".join(lines)


class DemoWorkflowRuntime(WorkflowRuntime):
    def __init__(self) -> None:
        super().__init__(
            incident_monitor=DemoIncidentMonitor(),
            kubernetes=DemoKubernetesClient(),
            rag=DemoK8sRAGRetriever(),
            org_roam=DemoOrgRoamBrowser(),
            k8sgpt=DemoK8sGPTAnalyzer(),
            reasoner=DemoReasoner(),
        )


class RealWorkflowRuntime(WorkflowRuntime):
    def __init__(self, config: ConnectorConfig):
        from .connectors import (
            HaystackQdrantRetriever,
            K8sGPTMCPAnalyzer,
            KubectlKubernetesClient,
            OllamaClient,
            OllamaReasoner,
            OrgRoamMCPBrowser,
        )

        ollama = OllamaClient(config)
        super().__init__(
            incident_monitor=DemoIncidentMonitor(),
            kubernetes=KubectlKubernetesClient(config),
            rag=HaystackQdrantRetriever(config, ollama),
            org_roam=OrgRoamMCPBrowser(config),
            k8sgpt=K8sGPTMCPAnalyzer(config),
            reasoner=OllamaReasoner(ollama),
        )
