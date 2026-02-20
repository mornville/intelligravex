# Gravex

**Local‑first automation agents** that run on your machine - not in a black box.

Gravex gives you a dashboard to create agents, run real tasks, and keep files, configs, and context on-device. Use local LLMs or OpenAI, add tools, and scale from one agent to teams.

## Run Locally (No Overlay)

> The overlay is macOS‑only. The dashboard works everywhere.

```bash
./start.sh web -v
```

Open:
- `http://localhost:8000/dashboard`

First‑time setup in the UI:
- Choose **Local**, **OpenAI**, or **OpenRouter**.
- Voice (ASR/TTS) is optional.
- Isolated Workspace is optional and requires Docker.
- For Local models, you can pick a bundled model or use **Custom URL** to provide a direct download link.
 - The Isolated Workspace image is built locally on first use (or run `./scripts/build_data_agent_image.sh`).

## Why Local‑First

- Your data stays on your machine by default.
- Local models work without API keys.
- You control what each agent can access.

## Features (At a Glance)

- Dedicated dashboard to spin up agents and teams fast.
- Permissioned actions with explicit approvals.
- Built‑in system tools + HTTP integrations (schemas + response mapping).
- Isolated Workspaces per agent (optional, Docker).
- Local or OpenAI models for chat and automation.
- Always‑on mic overlay (macOS‑only).

## Default Assistants

- **GravexStudio Guide** (system helper)
- **IGX Showcase** (demo agent)

## Screenshots

![Dashboard](marketing/static/dashboard.png)
![Agent configs](marketing/static/agent-configs.png)
![System tools](marketing/static/system-tools.png)
![Local LLM setup](marketing/static/local-llm-setup.png)

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
