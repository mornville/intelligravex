<p align="center">
  <img src="marketing/static/igx-logo.svg" width="120" alt="IGX logo" />
</p>
<p align="center">
  <a href="https://gravex-agent.vercel.app">gravex-agent.vercel.app</a>
</p>

# Gravex
Run multi-agent AI workflows locally, with isolated workspaces and real tool access. No black boxes. No vendor lock-in

## How Gravex is different
- Runs agents on *your* machine, not someone else’s cloud
- One agent = one isolated workspace (no collisions)
- Real tools (Git, Jira, DBs), not just chat
- Works with local LLMs *or* OpenAI

## Who is Gravex for?
- Engineers handling multiple Jira tickets daily
- Founders who want AI agents without leaking code or data
- Teams experimenting with multi-agent workflows locally
- Power users tired of SaaS copilots with zero control

  
## How do i use it?
- Made an agent by using the Studio and gave it access to gmail, github, jira, gitlab, databases.
- Logged in with my Openai account so no need of key if i have to use smaller models, else i provide my key.
- For my day to day tasks: It spins up multiple agents and I give each ticker Jira ticket links, that's it. If the description is sufficient,
- It clones the repo in isolated workspaces (docker container), does  the work and just gives me URLs to test.
- The Multi - Agent framework never collide with each other as its in seperate Docker Container. 
- This is just one of the infinite use cases.
- Others for fun: I just spin up agent swarm and let them collab and come up with the code/test/reviews after i give them a task
  
Gravex gives you a dashboard to create agents, run real tasks, and keep files, configs, and context on-device. Use local LLMs or OpenAI, add tools, and scale from one agent to teams.

## Integrations

<table>
  <tr>
    <td align="center"><img src="frontend/public/brands/openai.svg" width="80" alt="OpenAI" /><br/>OpenAI</td>
    <td align="center"><img src="https://cdn.simpleicons.org/github/181717" width="80" alt="GitHub" /><br/>GitHub</td>
    <td align="center"><img src="https://cdn.simpleicons.org/gitlab/fc6d26" width="80" alt="GitLab" /><br/>GitLab</td>
    <td align="center"><img src="frontend/public/brands/jira.svg" width="80" alt="Jira" /><br/>Jira</td>
    <td align="center"><img src="https://cdn.simpleicons.org/gmail/ea4335" width="80" alt="Gmail" /><br/>Gmail</td>
    <td align="center"><img src="https://cdn.simpleicons.org/postgresql/4169e1" width="80" alt="Database" /><br/>Database</td>
  </tr>
</table>

## Default Assistants

- **GravexStudio Guide** (system helper)
- **IGX Showcase** (demo agent)

## Run Locally

- Docker is only required for Isolated Workspaces.
- Download Docker Desktop for full features (Isolated Workspaces): https://www.docker.com/products/docker-desktop/
- Local models download automatically when selected.

> The dashboard works on macOS, Linux, and Windows.

```bash
./start.sh web -v
```
This command builds the Studio UI first, then starts the web server.

Windows options:
- Git Bash: `./start.sh web -v`
- PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[web]"
.\.venv\Scripts\python.exe -m voicebot web -v
```

Open:
- `http://localhost:8000/dashboard`

macOS auto-start (optional):
- Install at login/restart:

```bash
./scripts/install_macos_autostart.sh
```

- If you stop port `8000` by mistake, restart the service:

```bash
launchctl kickstart -k gui/$(id -u)/com.intelligravex.voicebot.web
```

- Disable auto-start:

```bash
./scripts/uninstall_macos_autostart.sh
```

First‑time setup in the UI:
- Choose **ChatGPT (OAuth)**, **Local**, **OpenAI**, or **OpenRouter**.
- Voice (ASR/TTS) is optional.
- Isolated Workspace is optional and requires Docker.
- For Local models, you can pick a bundled model or use **Custom URL** to provide a direct download link.
 - The Isolated Workspace image is built locally on first use (or run `./scripts/build_data_agent_image.sh`).
- Set up **Connected apps** if needed (GitHub, GitLab, Jira, Gmail OAuth, DB credentials).

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
- Connected apps hub for GitHub/GitLab, Jira, Gmail OAuth, and database profiles.

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
