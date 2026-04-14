from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..config import ConnectorConfig
from ..models import Diagnosis, PlannedAction
from ..state import WorkflowState


class ConnectorError(RuntimeError):
    pass


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ConnectorError(f"Model output did not contain a JSON object: {text}")
    return stripped[start : end + 1]


class DiagnosisPayload(BaseModel):
    summary: str
    root_cause: str
    confidence: Literal["low", "medium", "high"]
    evidence: list[str] = Field(default_factory=list)
    next_check: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class PlannedActionPayload(BaseModel):
    description: str
    command: str
    risk: Literal["low", "medium", "high"]
    requires_approval: bool
    rollback: str = ""

    @field_validator("risk", mode="before")
    @classmethod
    def normalize_risk(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("requires_approval", mode="before")
    @classmethod
    def normalize_requires_approval(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


class PlanPayload(BaseModel):
    actions: list[PlannedActionPayload]


class OllamaClient:
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._client = httpx.Client(
            base_url=config.ollama_base_url.rstrip("/"),
            timeout=config.ollama_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, text: str | list[str], *, model: str | None = None) -> list[list[float]]:
        try:
            response = self._client.post(
                "/api/embed",
                json={
                    "model": model or self.config.ollama_embedding_model,
                    "input": text,
                    "truncate": True,
                },
            )
        except httpx.TimeoutException as exc:
            raise ConnectorError(
                f"Ollama embed timed out after {self.config.ollama_timeout_seconds}s using "
                f"model {model or self.config.ollama_embedding_model}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Ollama embed request failed: {exc}") from exc
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise ConnectorError(f"Unexpected Ollama embed response: {payload}")
        return embeddings

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": model or self.config.ollama_chat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json" if json_mode else None,
                    "options": {"temperature": 0.1},
                },
            )
        except httpx.TimeoutException as exc:
            raise ConnectorError(
                f"Ollama chat timed out after {self.config.ollama_timeout_seconds}s using "
                f"model {model or self.config.ollama_chat_model}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Ollama chat request failed: {exc}") from exc
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ConnectorError(f"Unexpected Ollama chat response: {payload}")
        return content.strip()


class OllamaReasoner:
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    def _truncate(self, value: str, limit: int = 320) -> str:
        stripped = value.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 3] + "..."

    def _state_snapshot(self, state: WorkflowState) -> str:
        snapshot = {
            "incident_id": state.get("incident_id"),
            "trigger": state.get("trigger"),
            "namespace": state.get("namespace"),
            "workload": state.get("workload"),
            "severity": state.get("severity"),
            "evidence": {
                key: [self._truncate(item, 240) for item in values[:2]]
                for key, values in state.get("evidence", {}).items()
            },
            "retrieved_runbook_passages": [
                {
                    "title": item.get("title"),
                    "content": self._truncate(str(item.get("content", "")), 220),
                }
                for item in state.get("retrieved_runbook_passages", [])[:2]
            ],
            "retrieved_org_roam_nodes": [
                {
                    "title": item.get("title"),
                    "content": self._truncate(str(item.get("content", "")), 220),
                    "backlinks": item.get("backlinks", [])[:3],
                }
                for item in state.get("retrieved_org_roam_nodes", [])[:2]
            ],
            "k8sgpt_findings": [
                {
                    "resource": item.get("resource"),
                    "severity": item.get("severity"),
                    "description": self._truncate(str(item.get("description", "")), 180),
                }
                for item in state.get("k8sgpt_findings", [])[:3]
            ],
            "plan": state.get("plan", []),
            "risk_level": state.get("risk_level"),
        }
        return json.dumps(snapshot, indent=2)

    def _collect_strings(self, state: WorkflowState) -> list[str]:
        evidence = state.get("evidence", {})
        texts = list(evidence.get("events", [])) + list(evidence.get("logs", [])) + list(evidence.get("manifests", []))
        texts.extend(str(item.get("content", "")) for item in state.get("retrieved_runbook_passages", []))
        texts.extend(str(item.get("description", "")) for item in state.get("k8sgpt_findings", []))
        return [text for text in texts if text]

    def _find_configmap_name(self, state: WorkflowState) -> str:
        for text in self._collect_strings(state):
            for pattern in [
                r"ConfigMap ['\"]?([A-Za-z0-9.-]+)['\"]?",
                r"configMapRef(?:.|\n)*?name:\s*([A-Za-z0-9.-]+)",
                r"config map ([A-Za-z0-9.-]+)",
            ]:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return match.group(1)
        return "api-config"

    def _known_good_image(self, state: WorkflowState) -> str:
        trigger = state.get("trigger", "")
        if trigger == "ImagePullBackOff":
            return "busybox:1.36"
        return "busybox:1.36"

    def _plan_has_unsupported_commands(self, actions: list[PlannedAction]) -> bool:
        unsupported_tokens = ("|", "&&", ";", "||", "$(", "`")
        placeholders = ("<path/", "stable-tag", "latest or a specific stable version")
        for action in actions:
            command = action.command
            if any(token in command for token in unsupported_tokens):
                return True
            if any(token in command for token in placeholders):
                return True
        return False

    def _normalize_actions(self, state: WorkflowState, actions: list[PlannedAction]) -> list[PlannedAction]:
        if self._plan_has_unsupported_commands(actions):
            return self._fallback_plan(state)
        return actions

    def _fallback_diagnosis(self, state: WorkflowState, reason: str) -> Diagnosis:
        trigger = state.get("trigger", "Unknown")
        namespace = state.get("namespace", "default")
        workload = state.get("workload", "workload")
        combined = "\n".join(self._collect_strings(state)).lower()
        configmap_name = self._find_configmap_name(state)
        evidence = [self._truncate(item) for item in self._collect_strings(state)[:3]] + [self._truncate(reason)]

        if "configmap" in combined and ("not found" in combined or "missing" in combined):
            return Diagnosis(
                summary=(
                    f"Fallback diagnosis: {workload} in namespace {namespace} is likely failing because "
                    f"required ConfigMap {configmap_name} is missing."
                ),
                root_cause=f"Missing ConfigMap {configmap_name} referenced by the workload startup configuration.",
                confidence="medium",
                evidence=evidence,
                next_check=f"kubectl -n {namespace} get configmap {configmap_name}",
            )

        if trigger == "ImagePullBackOff" or "failed to pull image" in combined or "imagepullbackoff" in combined:
            return Diagnosis(
                summary=f"Fallback diagnosis: {workload} is likely using an invalid image tag or inaccessible registry.",
                root_cause="Container image could not be pulled successfully.",
                confidence="medium",
                evidence=evidence,
                next_check=f"kubectl -n {namespace} get deployment {workload} -o yaml",
            )

        if trigger == "Pending" or "insufficient memory" in combined or "insufficient cpu" in combined:
            return Diagnosis(
                summary=f"Fallback diagnosis: {workload} is unschedulable because cluster resources are insufficient.",
                root_cause="Pod scheduling is blocked by resource constraints.",
                confidence="medium",
                evidence=evidence,
                next_check=f"kubectl -n {namespace} describe pod -l app={workload}",
            )

        return Diagnosis(
            summary=f"Fallback diagnosis: {trigger} is affecting {workload}, but the exact root cause could not be inferred.",
            root_cause="Local model reasoning timed out before a precise diagnosis was returned.",
            confidence="low",
            evidence=evidence,
            next_check=f"kubectl -n {namespace} describe deployment {workload}",
        )

    def _fallback_plan(self, state: WorkflowState) -> list[PlannedAction]:
        namespace = state.get("namespace", "default")
        workload = state.get("workload", "workload")
        trigger = state.get("trigger", "Unknown")
        diagnosis_text = json.dumps(state.get("diagnosis", {})).lower()
        configmap_name = self._find_configmap_name(state)
        combined = "\n".join(self._collect_strings(state)).lower()

        if "configmap" in diagnosis_text or (trigger == "CrashLoopBackOff" and "configmap" in combined):
            return [
                PlannedAction(
                    description="Check whether the expected ConfigMap exists.",
                    command=f"kubectl -n {namespace} get configmap {configmap_name}",
                    risk="low",
                    requires_approval=False,
                    rollback="",
                ),
                PlannedAction(
                    description="Restore the expected ConfigMap from the demo manifest.",
                    command=f"kubectl -n {namespace} apply -f k8s/demo/configmap.yaml",
                    risk="medium",
                    requires_approval=True,
                    rollback=f"kubectl -n {namespace} delete configmap {configmap_name}",
                ),
                PlannedAction(
                    description="Restart the deployment after configuration is restored.",
                    command=f"kubectl -n {namespace} rollout restart deployment/{workload}",
                    risk="medium",
                    requires_approval=True,
                    rollback=f"kubectl -n {namespace} rollout undo deployment/{workload}",
                ),
            ]

        if trigger == "ImagePullBackOff":
            return [
                PlannedAction(
                    description="Inspect the image configured on the deployment.",
                    command=f"kubectl -n {namespace} get deployment {workload} -o yaml",
                    risk="low",
                    requires_approval=False,
                    rollback="",
                ),
                PlannedAction(
                    description="Patch the deployment to a known-good image tag.",
                    command=f"kubectl -n {namespace} set image deployment/{workload} {workload}={self._known_good_image(state)}",
                    risk="high",
                    requires_approval=True,
                    rollback=f"kubectl -n {namespace} rollout undo deployment/{workload}",
                ),
                PlannedAction(
                    description="Wait for the rollout to complete after patching the image.",
                    command=f"kubectl -n {namespace} rollout status deployment/{workload} --timeout=120s",
                    risk="low",
                    requires_approval=False,
                    rollback="",
                ),
            ]

        return [
            PlannedAction(
                description="Describe the deployment for more detail.",
                command=f"kubectl -n {namespace} describe deployment {workload}",
                risk="low",
                requires_approval=False,
                rollback="",
            ),
            PlannedAction(
                description="Check rollout status after investigation.",
                command=f"kubectl -n {namespace} rollout status deployment/{workload} --timeout=60s",
                risk="low",
                requires_approval=False,
                rollback="",
            ),
        ]

    def _fallback_summary(self, state: WorkflowState) -> str:
        diagnosis = state.get("diagnosis", {})
        execution = state.get("execution_result", {})
        verification = state.get("verification", {})
        return "\n".join(
            [
                f"Trigger: {state.get('trigger')} on {state.get('namespace')}/{state.get('workload')}",
                f"Diagnosis: {diagnosis.get('summary', 'Unavailable')}",
                f"Root cause: {diagnosis.get('root_cause', 'Unavailable')}",
                f"Approval: {state.get('approval_status', 'unknown')} (risk={state.get('risk_level', 'unknown')})",
                f"Execution: {execution.get('status', 'not-run')} - {execution.get('details', 'No execution details')}",
                f"Verification: {verification.get('summary', 'No verification details')}",
            ]
        )

    def diagnose(self, state: WorkflowState) -> Diagnosis:
        system_prompt = (
            "You are a Kubernetes incident diagnosis agent. Return only valid JSON with keys: "
            "summary, root_cause, confidence, evidence, next_check."
        )
        user_prompt = (
            "Diagnose the most likely root cause from this incident state.\n"
            "Prefer concrete Kubernetes explanations over generic wording.\n"
            f"{self._state_snapshot(state)}"
        )
        try:
            raw = self.ollama.chat(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
            payload = DiagnosisPayload.model_validate_json(_extract_json_object(raw))
        except ValidationError as exc:
            return self._fallback_diagnosis(state, f"Failed to validate diagnosis payload: {exc}")
        except ConnectorError as exc:
            return self._fallback_diagnosis(state, str(exc))
        return Diagnosis(**payload.model_dump())

    def plan(self, state: WorkflowState) -> list[PlannedAction]:
        system_prompt = (
            "You are a Kubernetes remediation planner. Return only valid JSON with one top-level key "
            "'actions', whose value is an array of objects with keys: description, command, risk, "
            "requires_approval, rollback."
        )
        user_prompt = (
            "Create a remediation plan from this incident state. "
            "Commands must be executable via kubectl or helm, with explicit namespace when relevant. "
            "Read-only checks should be low risk. Mutating commands should require approval if medium or high risk.\n"
            f"{self._state_snapshot(state)}"
        )
        try:
            raw = self.ollama.chat(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
            payload = PlanPayload.model_validate_json(_extract_json_object(raw))
        except ValidationError as exc:
            return self._fallback_plan(state)
        except ConnectorError:
            return self._fallback_plan(state)
        return self._normalize_actions(
            state,
            [PlannedAction(**action.model_dump()) for action in payload.actions],
        )

    def summarize(self, state: WorkflowState) -> str:
        system_prompt = (
            "You are a concise SRE incident reporter. Produce a short operational summary in plain text."
        )
        user_prompt = (
            "Summarize this incident in 8 lines or fewer. Include trigger, diagnosis, evidence sources, "
            "approval status, execution result, and verification outcome.\n"
            f"{self._state_snapshot(state)}"
        )
        try:
            return self.ollama.chat(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=False)
        except ConnectorError:
            return self._fallback_summary(state)
