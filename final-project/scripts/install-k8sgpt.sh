#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd curl
require_cmd jq
require_cmd tar

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) ASSET_NAME="k8sgpt_Linux_x86_64.tar.gz" ;;
  aarch64|arm64) ASSET_NAME="k8sgpt_Linux_arm64.tar.gz" ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

INSTALL_DIR="${HOME}/.local/bin"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${INSTALL_DIR}"

RELEASE_JSON="$(curl -fsSL https://api.github.com/repos/k8sgpt-ai/k8sgpt/releases/latest)"
DOWNLOAD_URL="$(printf '%s' "${RELEASE_JSON}" | jq -r --arg name "${ASSET_NAME}" '.assets[] | select(.name == $name) | .browser_download_url')"

if [[ -z "${DOWNLOAD_URL}" || "${DOWNLOAD_URL}" == "null" ]]; then
  echo "Could not find release asset ${ASSET_NAME}" >&2
  exit 1
fi

curl -fsSL "${DOWNLOAD_URL}" -o "${TMP_DIR}/k8sgpt.tar.gz"
tar -xzf "${TMP_DIR}/k8sgpt.tar.gz" -C "${TMP_DIR}"
install -m 0755 "${TMP_DIR}/k8sgpt" "${INSTALL_DIR}/k8sgpt"

echo "Installed k8sgpt to ${INSTALL_DIR}/k8sgpt"
echo "Ensure ${INSTALL_DIR} is in PATH."
