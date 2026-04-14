#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd kubectl

kubectl -n payments set image deployment/api-server api-server=ghcr.io/example/api-server:bad-tag
kubectl -n payments rollout status deployment/api-server --timeout=30s || true
kubectl -n payments get pods

echo "Injected bad image fault into payments/api-server."
