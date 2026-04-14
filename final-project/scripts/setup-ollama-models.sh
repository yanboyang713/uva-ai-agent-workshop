#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_env
require_cmd ollama

CHAT_MODEL="${AI_OPS_OLLAMA_CHAT_MODEL:-gemma4:e4b}"
EMBED_MODEL="${AI_OPS_OLLAMA_EMBED_MODEL:-embeddinggemma}"

ollama pull "${CHAT_MODEL}"
ollama pull "${EMBED_MODEL}"

echo "Pulled Ollama models: ${CHAT_MODEL}, ${EMBED_MODEL}"
