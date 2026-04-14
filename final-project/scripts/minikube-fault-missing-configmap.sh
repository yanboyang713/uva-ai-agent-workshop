#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd kubectl

kubectl -n payments delete configmap api-config --ignore-not-found
kubectl -n payments rollout restart deployment/api-server
kubectl -n payments get pods

echo "Injected missing ConfigMap fault into payments/api-server."
