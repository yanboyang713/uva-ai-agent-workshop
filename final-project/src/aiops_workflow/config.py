from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_project_dotenv() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break


def _get_list_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if not inner:
                return []
            return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
        return shlex.split(value)
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    if isinstance(parsed, str):
        return shlex.split(parsed)
    raise ValueError(f"{name} must be a JSON array of strings or a shell-style string")


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ConnectorConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e4b"
    ollama_embedding_model: str = "embeddinggemma"
    ollama_timeout_seconds: int = 600

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "k8s-rag"
    qdrant_api_key: str | None = None
    qdrant_top_k: int = 5
    qdrant_embedding_dim: int | None = None

    org_roam_mcp_command: str = "org-roam-mcp"
    org_roam_mcp_args: list[str] | None = None
    org_roam_db_path: str | None = None
    org_roam_dir: str | None = None
    k8sgpt_mcp_command: str = "k8sgpt"
    k8sgpt_mcp_args: list[str] | None = None
    k8sgpt_backend: str = "ollama"

    kubectl_command: str = "kubectl"
    kubectl_context: str | None = None
    kubeconfig_path: str | None = None
    allow_mutations: bool = False

    @classmethod
    def from_env(cls) -> "ConnectorConfig":
        _load_project_dotenv()
        return cls(
            ollama_base_url=os.getenv("AI_OPS_OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_chat_model=os.getenv("AI_OPS_OLLAMA_CHAT_MODEL", "gemma4:e4b"),
            ollama_embedding_model=os.getenv("AI_OPS_OLLAMA_EMBED_MODEL", "embeddinggemma"),
            ollama_timeout_seconds=int(os.getenv("AI_OPS_OLLAMA_TIMEOUT_SECONDS", "600")),
            qdrant_url=os.getenv("AI_OPS_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("AI_OPS_QDRANT_COLLECTION", "k8s-rag"),
            qdrant_api_key=os.getenv("AI_OPS_QDRANT_API_KEY"),
            qdrant_top_k=int(os.getenv("AI_OPS_QDRANT_TOP_K", "5")),
            qdrant_embedding_dim=(
                int(os.getenv("AI_OPS_QDRANT_EMBEDDING_DIM"))
                if os.getenv("AI_OPS_QDRANT_EMBEDDING_DIM")
                else None
            ),
            org_roam_mcp_command=os.getenv("AI_OPS_ORG_ROAM_MCP_COMMAND", "org-roam-mcp"),
            org_roam_mcp_args=_get_list_env("AI_OPS_ORG_ROAM_MCP_ARGS", []),
            org_roam_db_path=os.getenv("ORG_ROAM_DB_PATH"),
            org_roam_dir=os.getenv("ORG_ROAM_DIR"),
            k8sgpt_mcp_command=os.getenv("AI_OPS_K8SGPT_MCP_COMMAND", "k8sgpt"),
            k8sgpt_mcp_args=_get_list_env("AI_OPS_K8SGPT_MCP_ARGS", ["serve", "--mcp"]),
            k8sgpt_backend=os.getenv("AI_OPS_K8SGPT_BACKEND", "ollama"),
            kubectl_command=os.getenv("AI_OPS_KUBECTL_COMMAND", "kubectl"),
            kubectl_context=os.getenv("AI_OPS_KUBECTL_CONTEXT"),
            kubeconfig_path=os.getenv("AI_OPS_KUBECONFIG"),
            allow_mutations=_get_bool_env("AI_OPS_ALLOW_MUTATIONS", False),
        )

    def mcp_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.kubeconfig_path:
            env["KUBECONFIG"] = self.kubeconfig_path
        if self.org_roam_db_path:
            env["ORG_ROAM_DB_PATH"] = self.org_roam_db_path
        if self.org_roam_dir:
            env["ORG_ROAM_DIR"] = self.org_roam_dir
        return env

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "ollama_base_url": self.ollama_base_url,
            "ollama_chat_model": self.ollama_chat_model,
            "ollama_embedding_model": self.ollama_embedding_model,
            "ollama_timeout_seconds": self.ollama_timeout_seconds,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            "org_roam_mcp_command": self.org_roam_mcp_command,
            "org_roam_mcp_args": self.org_roam_mcp_args or [],
            "org_roam_db_path": self.org_roam_db_path,
            "org_roam_dir": self.org_roam_dir,
            "k8sgpt_mcp_command": self.k8sgpt_mcp_command,
            "k8sgpt_mcp_args": self.k8sgpt_mcp_args or [],
            "k8sgpt_backend": self.k8sgpt_backend,
            "kubectl_command": self.kubectl_command,
            "kubectl_context": self.kubectl_context,
            "kubeconfig_path": self.kubeconfig_path,
            "allow_mutations": self.allow_mutations,
        }
