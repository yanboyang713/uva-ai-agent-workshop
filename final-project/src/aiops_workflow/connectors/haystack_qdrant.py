from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "false")

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from ..config import ConnectorConfig
from ..models import RetrievedPassage
from ..state import WorkflowState
from .ollama import OllamaClient

try:
    from haystack.utils import Secret
except ImportError:  # pragma: no cover - older Haystack integration shape
    Secret = None  # type: ignore[assignment]


class HaystackQdrantRetriever:
    def __init__(self, config: ConnectorConfig, ollama: OllamaClient):
        self.config = config
        self.ollama = ollama
        self._embedding_dim = config.qdrant_embedding_dim or len(
            ollama.embed("embedding dimension probe")[0]
        )
        self.document_store = self._build_document_store()
        self.retriever = QdrantEmbeddingRetriever(document_store=self.document_store, top_k=config.qdrant_top_k)

    def _build_document_store(self) -> QdrantDocumentStore:
        kwargs: dict[str, Any] = {
            "url": self.config.qdrant_url,
            "index": self.config.qdrant_collection,
            "embedding_dim": self._embedding_dim,
            "return_embedding": False,
            "wait_result_from_api": True,
        }
        if self.config.qdrant_api_key:
            if Secret is not None:
                kwargs["api_key"] = Secret.from_token(self.config.qdrant_api_key)
            else:
                kwargs["api_key"] = self.config.qdrant_api_key
        return QdrantDocumentStore(**kwargs)

    def _build_query(self, state: WorkflowState) -> str:
        parts = [
            f"trigger: {state.get('trigger', '')}",
            f"namespace: {state.get('namespace', '')}",
            f"workload: {state.get('workload', '')}",
        ]
        events = state.get("evidence", {}).get("events", [])
        logs = state.get("evidence", {}).get("logs", [])
        if events:
            parts.append(f"events: {events[0][:400]}")
        if logs:
            parts.append(f"logs: {logs[0][:400]}")
        return "\n".join(parts)

    def retrieve(self, state: WorkflowState) -> list[RetrievedPassage]:
        query_embedding = self.ollama.embed(self._build_query(state))[0]
        response = self.retriever.run(query_embedding=query_embedding, top_k=self.config.qdrant_top_k)
        documents = response.get("documents", [])
        passages: list[RetrievedPassage] = []
        for document in documents:
            score = float(getattr(document, "score", 0.0) or 0.0)
            passages.append(
                RetrievedPassage(
                    source_id=document.id or document.meta.get("source_id", "unknown"),
                    title=str(document.meta.get("title", document.id or "Untitled document")),
                    content=document.content or "",
                    score=score,
                    metadata=document.meta or {},
                )
            )
        return passages

    def write_documents(
        self,
        documents: list[Document],
        *,
        policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> int:
        embedded_documents: list[Document] = []
        for document in documents:
            embedding = self.ollama.embed(document.content or "")[0]
            embedded_documents.append(replace(document, embedding=embedding))
        return self.document_store.write_documents(embedded_documents, policy=policy)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[Document]:
        documents: list[Document] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                documents.append(
                    Document(
                        id=record.get("id"),
                        content=record["content"],
                        meta={
                            "title": record.get("title", "Untitled"),
                            **record.get("meta", {}),
                        },
                    )
                )
        return documents
