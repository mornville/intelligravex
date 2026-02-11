# GravexStudio

GravexStudio is a desktop studio for building assistants that listen, reason, speak, and run real work across tools and long‑running tasks—without juggling separate servers.

## One‑line build (macOS / Linux / Windows)

Choose your platform and run the single build command:

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

## Quick start

1) Open the app (`dist/GravexStudio.app` or the AppImage).
2) The UI walks you through first‑time setup.
3) Create your first assistant and start a conversation.

## What GravexStudio can do

### Assistant studio
- Create, configure, and manage multiple assistants.
- Per‑assistant personality, system prompt, and response style.
- Central dashboard for assistants, chats, and developer settings.

### Multi‑model orchestration (per assistant)
- **LLM model** for main chat and reasoning.
- **ASR model** for voice input.
- **TTS model + voice** for spoken replies.
- **Web search model** for live information retrieval.
- **Codex model** for post‑processing tool responses.
- **Summary model** for long‑context rollups.
- Adjustable history window (keep last N turns verbatim).

### Real‑time conversations
- Streamed responses (text and audio) with latency metrics.
- Live mic test and talk loop built into the UI.
- Continuous session support (start / stop, new conversation).

### Isolated Workspace (optional, Docker)
- Run each conversation in its **own isolated workspace**.
- Long‑running tasks that persist across turns.
- Files, logs, and outputs stored per workspace.
- Multiple concurrent containers for parallel work.
- Container monitor with status and kill controls.
- Automatic respawn of the workspace when needed.

### Tools & integrations
- Built‑in system tools: **metadata** + **web search**.
- HTTP integration tools that let assistants call external APIs.
- Tool args schemas (JSON Schema or required args lists).
- Response schemas for validation and structured extraction.
- Response‑to‑metadata mapping for durable memory.
- Static reply templates (Jinja2) for deterministic outputs.
- Optional “Use Codex for response” post‑processing.

### Variables (metadata templating)
Prompts and replies can reference conversation metadata using `{{...}}` placeholders.

- Shorthand metadata lookup:
  - `{{.first_name}}`
  - `{{.user.profile.firstName}}` (nested)
- Other contexts (used by integrations):
  - `{{args.user_id}}` / `{{params.user_id}}` (tool‑call arguments)
  - `{{response.data.first_name}}` (HTTP response JSON)

Where variables are applied:
- Assistant `system_prompt` is rendered with metadata before every LLM call.
- Message history is rendered with metadata before sending to the LLM.
- Tool `next_reply` is rendered with metadata before sending back to the user.

Missing variables resolve to an empty string.

### Integration tools (HTTP → metadata)
You can attach HTTP API “integration tools” to an assistant. When a tool is called, GravexStudio will:
1) Execute the HTTP request.
2) Map selected response fields into conversation metadata (via a response mapper).
3) Return a tool result to the chat model, which then replies to the user.

Important:
- Raw HTTP responses are **not** sent back to the LLM.
- Only the mapped keys are merged into conversation metadata.

#### Tool call schema (LLM‑facing)
Each integration tool is exposed to the LLM as a function tool with a top‑level shape:

```json
{
  "args": { "... tool args ..." },
  "wait_reply": "optional short filler while the tool runs",
  "next_reply": "optional (only when Codex is NOT enabled)"
}
```

Notes:
- `args` is always required.
- `next_reply` is required when **Codex is NOT enabled** (to avoid a second LLM call).
- If “Use Codex for response” is enabled, `next_reply` is not required; the backend runs a Codex post‑processor and returns a result to the main chat model to rephrase.

#### Args schema (required args + JSON schema)
You can define `args` in two ways:
- **Required args list** (comma‑separated in UI): the backend generates a permissive schema and requires those keys.
- **Args schema (JSON Schema)**: the backend uses this as the schema for `args`.

If both are present, required args are appended to the schema (they are not removed), so the integration still has all the fields needed to call the API.

#### Static reply (optional, Jinja2) — takes priority
Integrations can be configured with an optional `static_reply_template`.

If `static_reply_template` is set:
- The backend ignores `next_reply`.
- The backend also ignores “Use Codex for response” (static template wins).
- The backend renders `static_reply_template` using **Jinja2** (supports `{% if %}`, `{% for %}`, etc).

Template context:
- `meta`: current conversation metadata (after response mapping)
- `response`: raw HTTP response JSON
- `args` / `params`: tool‑call arguments (excluding `next_reply`)

#### Use Codex for response (optional)
If enabled, the backend runs a separate Codex “one‑shot” agent after the HTTP request:
1) The HTTP response is saved to a temp file.
2) The Codex model receives:
   - the response JSON schema (from the tool config, or a best‑effort derived schema)
   - the response file path (so its generated Python script can read locally)
   - intent fields inside `args`:
     - `fields_required`: what fields are needed to build the response
     - `why_api_was_called`: why this API call happened (user intent)
3) Codex returns a Python script that extracts/aggregates the required info and writes a `result.txt`.
4) The tool result includes `codex_result_text` and file paths; the **main chat model** rephrases the final user‑facing reply.

### Context control
- Metadata templating in prompts and tool replies.
- Automatic conversation summarization as context grows.
- Keeps long sessions fast while retaining key facts.

### Local‑first and secure
- Workspaces, configs, and conversation history stored locally.
- Provider secrets encrypted at rest.
- You decide what gets shared externally.

### Local LLMs (no API key)
- Run local GGUF models via the bundled llama.cpp server.
- Auto‑download models on first use (no manual setup).
- Tool‑calling aware model catalog with compatibility labels.
- Keep sensitive workflows fully on device.

### Embeddable chat widget
- Public text‑chat widget you can embed in websites.
- Client key support and WebSocket transport.

### Git + SSH tooling (inside Isolated Workspace)
- Secure repo access for data‑agent workflows.
- Useful for code tasks, automation, and structured work.

## First‑time setup (no CLI)

The app guides you through everything:

1) **LLM choice** — OpenAI, OpenRouter, or **Local model (no API key)**.
2) **Voice (optional)** — OpenAI is still required for ASR/TTS.
3) **Isolated Workspace** (optional) — requires Docker installed and running.

Default Isolated Workspace image:
- `ghcr.io/mornville/data-agent:latest`

Override the Isolated Workspace image if needed:
- `IGX_DATA_AGENT_IMAGE=...`

## Notes

- Docker is only needed if you enable the Isolated Workspace.
- All core features work without Docker.
- Local models are downloaded automatically and stored on your machine.
