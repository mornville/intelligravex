FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Minimal dependencies for Codex CLI + typical API scripting.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    jq \
    nodejs \
    npm \
    openssh-client \
  && rm -rf /var/lib/apt/lists/*

# Codex CLI (Node-based).
RUN npm install -g @openai/codex@latest

# OpenVSCode Server (VS Code in the browser).
# Download the latest release for the current architecture.
ARG OPENVSCODE_VERSION=""
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64|amd64) ovs_arch="x64" ;; \
      aarch64|arm64) ovs_arch="arm64" ;; \
      *) echo "Unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    ovs_version="${OPENVSCODE_VERSION:-}"; \
    if [ -z "$ovs_version" ]; then \
      ovs_version="$(curl -fsSL -o /dev/null -w '%{url_effective}' https://github.com/gitpod-io/openvscode-server/releases/latest | awk -F/ '{print $NF}')"; \
    fi; \
    if [ -z "$ovs_version" ]; then \
      echo "Failed to resolve OpenVSCode Server version." >&2; exit 1; \
    fi; \
    ovs_tag="${ovs_version}"; \
    ovs_base="${ovs_version#openvscode-server-}"; \
    curl -fsSL "https://github.com/gitpod-io/openvscode-server/releases/download/${ovs_tag}/openvscode-server-${ovs_base}-linux-${ovs_arch}.tar.gz" -o /tmp/openvscode-server.tgz; \
    tar -xzf /tmp/openvscode-server.tgz -C /opt; \
    mv "/opt/openvscode-server-${ovs_base}-linux-${ovs_arch}" /opt/openvscode-server; \
    ln -s /opt/openvscode-server/bin/openvscode-server /usr/local/bin/openvscode-server; \
    rm -f /tmp/openvscode-server.tgz

# Convenience libs for agent-authored scripts.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir requests

WORKDIR /work
