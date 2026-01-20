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

# Convenience libs for agent-authored scripts.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir requests

WORKDIR /work
