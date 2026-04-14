from __future__ import annotations

import argparse
from typing import Sequence

from haystack.document_stores.types import DuplicatePolicy

from .config import ConnectorConfig
from .connectors.haystack_qdrant import HaystackQdrantRetriever
from .connectors.ollama import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a JSONL RAG corpus into Qdrant with Ollama embeddings.")
    parser.add_argument("input_path", help="Path to a JSONL file with records containing content, title, and meta.")
    parser.add_argument(
        "--duplicate-policy",
        choices=("overwrite", "skip", "fail"),
        default="overwrite",
        help="How to handle documents whose IDs already exist in Qdrant. Default: overwrite.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ConnectorConfig.from_env()
    ollama = OllamaClient(config)
    retriever = HaystackQdrantRetriever(config, ollama)
    documents = retriever.load_jsonl(args.input_path)
    policy = DuplicatePolicy[args.duplicate_policy.upper()]
    written = retriever.write_documents(documents, policy=policy)
    print(f"Ingested {written} documents into Qdrant collection '{config.qdrant_collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
