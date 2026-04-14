#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd conda

cd "${ROOT_DIR}"

PYTHON_VERSION="${AI_OPS_PYTHON_VERSION:-3.12}"

if ! conda run -n "${CONDA_ENV_NAME}" python --version >/dev/null 2>&1; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

run_in_conda python -m pip install --upgrade pip
run_in_conda python -m pip install -e .

echo "Python environment is ready in Conda env '${CONDA_ENV_NAME}'"
echo "Use it with: conda activate ${CONDA_ENV_NAME}"
