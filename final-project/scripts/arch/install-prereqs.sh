#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root: sudo $0"
  exit 1
fi

TARGET_USER="${SUDO_USER:-${USER}}"
USE_PODMAN=0
USE_SYSTEM_OLLAMA=0

if pacman -Q podman-docker >/dev/null 2>&1 || pacman -Q podman >/dev/null 2>&1; then
  USE_PODMAN=1
fi

if ! pacman -Q ollama >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1 || [[ -e /usr/share/ollama ]]; then
    USE_SYSTEM_OLLAMA=1
  fi
fi

ensure_service_enabled_and_running() {
  local service="$1"

  if systemctl is-enabled --quiet "${service}"; then
    echo "${service} is already enabled."
  else
    systemctl enable "${service}"
    echo "Enabled ${service}."
  fi

  if systemctl is-active --quiet "${service}"; then
    echo "${service} is already running."
  else
    systemctl start "${service}"
    echo "Started ${service}."
  fi
}

PACKAGES=(
  base-devel
  conntrack-tools
  curl
  git
  jq
  kubectl
  minikube
  python
  tar
  unzip
  uv
)

if [[ "${USE_PODMAN}" -eq 1 ]]; then
  echo "Detected podman/podman-docker. Skipping docker package installation."
else
  PACKAGES+=(docker)
fi

if [[ "${USE_SYSTEM_OLLAMA}" -eq 1 ]]; then
  echo "Detected an existing non-pacman Ollama installation. Skipping ollama package installation."
else
  PACKAGES+=(ollama)
fi

pacman -Syu --needed --noconfirm "${PACKAGES[@]}"

if [[ "${USE_PODMAN}" -eq 0 ]]; then
  ensure_service_enabled_and_running docker.service
else
  echo "Using Podman-compatible container workflow. docker.service setup skipped."
fi

if systemctl list-unit-files ollama.service >/dev/null 2>&1; then
  ensure_service_enabled_and_running ollama.service
elif command -v ollama >/dev/null 2>&1; then
  echo "ollama command exists, but no ollama.service unit was found. Manage your existing Ollama install manually."
else
  echo "Ollama was not installed and no existing command was found."
fi

if [[ "${USE_PODMAN}" -eq 0 ]]; then
  if id -nG "${TARGET_USER}" | grep -qw docker; then
    echo "User ${TARGET_USER} is already in the docker group."
  else
    usermod -aG docker "${TARGET_USER}"
    echo "Added ${TARGET_USER} to the docker group."
    echo "Log out and back in before using Docker without sudo."
  fi
else
  echo "Docker group setup skipped because Podman is being used."
fi
