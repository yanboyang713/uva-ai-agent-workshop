#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

NOTEBOOK_URL="https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/manual_rag_pipeline_universal.ipynb"
CORPORA_URL="https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/Corpora.zip"

mkdir -p resources

echo "Downloading manual_rag_pipeline_universal.ipynb ..."
curl -L "$NOTEBOOK_URL" -o resources/manual_rag_pipeline_universal.ipynb

echo "Downloading Corpora.zip (large file) ..."
curl -L "$CORPORA_URL" -o resources/Corpora.zip

echo "Unzipping corpora ..."
unzip -o resources/Corpora.zip -d resources/

echo "Done. Corpora extracted under: $ROOT_DIR/resources/Corpora"
