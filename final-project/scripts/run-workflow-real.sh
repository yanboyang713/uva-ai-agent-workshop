#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_env

cd "${ROOT_DIR}"
run_in_conda python -m aiops_workflow.cli --runtime real --progress "$@"
