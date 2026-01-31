#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IGX_DATA_AGENT_IMAGE:-ghcr.io/mornville/data-agent:latest}"
DOCKERFILE="${ROOT_DIR}/packaging/data-agent/Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Missing Dockerfile at ${DOCKERFILE}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the Data Agent image."
  exit 1
fi

echo "Building Data Agent image: ${IMAGE}"
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" -f "${DOCKERFILE}" "${ROOT_DIR}/packaging/data-agent"

echo "Build complete: ${IMAGE}"
