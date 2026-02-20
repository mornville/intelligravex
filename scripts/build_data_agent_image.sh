#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IGX_DATA_AGENT_IMAGE:-igx-data-agent:local}"
DOCKERFILE="${ROOT_DIR}/packaging/data-agent/Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Missing Dockerfile at ${DOCKERFILE}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the Isolated Workspace image."
  exit 1
fi

echo "Building Isolated Workspace image: ${IMAGE}"
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" -f "${DOCKERFILE}" "${ROOT_DIR}/packaging/data-agent"

echo "Build complete: ${IMAGE}"
