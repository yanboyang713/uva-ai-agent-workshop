#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd minikube
require_cmd kubectl

PROFILE="${MINIKUBE_PROFILE:-minikube}"
MEMORY="${MINIKUBE_MEMORY:-8192}"
CPUS="${MINIKUBE_CPUS:-4}"
K8S_VERSION="${MINIKUBE_KUBERNETES_VERSION:-stable}"
DRIVER="${MINIKUBE_DRIVER:-$(minikube_driver)}"
ROOTLESS="${MINIKUBE_ROOTLESS:-true}"
CONTAINER_RUNTIME="${MINIKUBE_CONTAINER_RUNTIME:-containerd}"

MINIKUBE_ARGS=(
  --profile "${PROFILE}"
  --driver "${DRIVER}"
  --memory "${MEMORY}"
  --cpus "${CPUS}"
  --kubernetes-version "${K8S_VERSION}"
  --container-runtime "${CONTAINER_RUNTIME}"
)

MINIKUBE_PROFILE_ARGS=(
  --profile "${PROFILE}"
)

if [[ "${DRIVER}" == "podman" && "${ROOTLESS}" == "true" ]]; then
  MINIKUBE_ARGS+=(--rootless)
  MINIKUBE_PROFILE_ARGS+=(--rootless)
fi

minikube start "${MINIKUBE_ARGS[@]}"

kubectl config use-context "${PROFILE}"
minikube addons enable metrics-server "${MINIKUBE_PROFILE_ARGS[@]}"
minikube addons enable ingress "${MINIKUBE_PROFILE_ARGS[@]}"

echo "minikube profile ${PROFILE} is ready with driver ${DRIVER}."
