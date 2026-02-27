<p align="center">
  <img src="marketing/static/igx-logo.svg" width="120" alt="IGX logo" />
</p>
<p align="center">
  <a href="https://gravex-agent.vercel.app">gravex-agent.vercel.app</a>
</p>

# Gravex

- **Local‑first automation agents** that run on your machine - not in a black box.
- **Now with fast ChatGPT sign‑in** for instant GPT access, no API key copy‑paste required.
- **Or just use openai keys** for accessing all the features like TTS, ASR

Gravex gives you a dashboard to create agents, run real tasks, and keep files, configs, and context on-device. Use local LLMs or OpenAI, add tools, and scale from one agent to teams.

## Default Assistants

- **GravexStudio Guide** (system helper)
- **IGX Showcase** (demo agent)

## Run Locally (No Overlay)

> The overlay is macOS‑only. The dashboard works everywhere.

```bash
./start.sh web -v
```

Open:
- `http://localhost:8000/dashboard`

First‑time setup in the UI:
- Choose **ChatGPT (OAuth)**, **Local**, **OpenAI**, or **OpenRouter**.
- Voice (ASR/TTS) is optional.
- Isolated Workspace is optional and requires Docker.
- For Local models, you can pick a bundled model or use **Custom URL** to provide a direct download link.
 - The Isolated Workspace image is built locally on first use (or run `./scripts/build_data_agent_image.sh`).

## Why Local‑First

- Your data stays on your machine by default.
- Local models work without API keys.
- You control what each agent can access.

## Features

- Dedicated dashboard to spin up agents and teams fast.
- Local LLM runtime with **0 API keys** and **0 data export**.
- ChatGPT OAuth sign‑in for instant GPT access (no API key copy‑paste).
- Summarization to keep context stable with less drift.
- System tools with 1‑click enablement and approvals.
- HTTP request tool (ad‑hoc) for zero‑setup API calls.
- Integration tools + response mapper for schema‑driven APIs.
- Codex post‑processing for cleaner structured outputs.
- Isolated Workspaces per agent (optional, Docker).
- Local or OpenAI models for chat and automation.
- Always‑on mic overlay (macOS‑only).

## Screenshots

<img src="marketing/static/dashboard.png" width="520" alt="Dashboard" style="border-radius:12px;border:1px solid rgba(255,255,255,0.12);margin:8px 0;" />
<img src="marketing/static/agent-configs.png" width="520" alt="Agent configs" style="border-radius:12px;border:1px solid rgba(255,255,255,0.12);margin:8px 0;" />
<img src="marketing/static/system-tools.png" width="520" alt="System tools" style="border-radius:12px;border:1px solid rgba(255,255,255,0.12);margin:8px 0;" />
<img src="marketing/static/local-llm-setup.png" width="520" alt="Local LLM setup" style="border-radius:12px;border:1px solid rgba(255,255,255,0.12);margin:8px 0;" />

## Desktop Build (Optional)

```bash
./scripts/package_macos.sh
```

```bash
./scripts/package_linux_appimage.sh
```

```powershell
.\scripts\package_windows.ps1
```

Outputs:
- macOS: `dist/GravexStudio.app`
- Linux: `dist/GravexStudio-x86_64.AppImage`
- Windows: `dist/GravexStudio.exe`

## Notes

- Docker is only required for Isolated Workspaces.
- Local models download automatically when selected.
