#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd kubectl

kubectl apply -f "${ROOT_DIR}/k8s/demo/namespace.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/demo/configmap.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/demo/deployment.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/demo/service.yaml"
kubectl -n payments rollout status deployment/api-server --timeout=120s

echo "Demo app deployed in namespace payments."
