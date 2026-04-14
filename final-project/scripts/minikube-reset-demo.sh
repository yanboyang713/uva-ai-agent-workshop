#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd kubectl

kubectl apply -f "${ROOT_DIR}/k8s/demo"
kubectl -n payments set image deployment/api-server api-server=busybox:1.36

rollout_ok=true
if ! kubectl -n payments rollout status deployment/api-server --timeout=120s; then
  rollout_ok=false
fi

kubectl -n payments wait --for=condition=Available deployment/api-server --timeout=120s
kubectl -n payments wait --for=jsonpath='{.status.readyReplicas}'=1 deployment/api-server --timeout=120s

if [[ "${rollout_ok}" == "false" ]]; then
  echo "rollout status reported a failure, but the deployment is now healthy; continuing." >&2
fi

echo "Reset demo resources to a healthy baseline."
