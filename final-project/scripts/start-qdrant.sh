#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ENGINE="$(container_engine)"
IMAGE="${AI_OPS_QDRANT_IMAGE:-docker.io/qdrant/qdrant:latest}"

DATA_DIR="${HOME}/.local/share/aiops-qdrant"
mkdir -p "${DATA_DIR}"

if "${ENGINE}" ps -a --format '{{.Names}}' | grep -qx aiops-qdrant; then
  "${ENGINE}" rm -f aiops-qdrant >/dev/null
fi

"${ENGINE}" run -d \
  --name aiops-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "${DATA_DIR}:/qdrant/storage" \
  "${IMAGE}" >/dev/null

echo "Qdrant is running on http://localhost:6333 via ${ENGINE} using ${IMAGE}"
