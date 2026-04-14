#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${AI_OPS_CONDA_ENV_NAME:-aiops-workflow}"

load_env() {
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${ROOT_DIR}/.env"
    set +a
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

run_in_conda() {
  require_cmd conda
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" "$@"
}

container_engine() {
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return
  fi
  if command -v docker >/dev/null 2>&1; then
    echo "docker"
    return
  fi
  echo "No supported container engine found. Install podman or docker." >&2
  exit 1
}

minikube_driver() {
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return
  fi
  if command -v docker >/dev/null 2>&1; then
    echo "docker"
    return
  fi
  echo "No supported minikube driver found. Install podman or docker." >&2
  exit 1
}
