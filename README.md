# Intelligravex VoiceBot

Continuous, end-to-end AI voice bot + Studio:
- **ASR:** OpenAI (`gpt-4o-mini-transcribe`)
- **LLM:** OpenAI (configurable)
- **TTS:** OpenAI (`gpt-4o-mini-tts`)

## Features

- **Studio UI (React)** to create bots, manage API keys, run mic tests, and inspect conversations.
- **Multiple model slots per bot**:
  - `openai_model` (main chat)
  - `web_search_model` (system tool: web_search)
  - `codex_model` (HTTP tools with “Use Codex for response”)
- **System tools** (built-in): `set_metadata`, `web_search`, `recall_http_response`, `export_http_response`.
- **Per-bot tool enable/disable** from the bot page:
  - Disable/enable built-in system tools (except `set_metadata`).
  - Disable/enable each HTTP integration tool.
- **HTTP integration tools** (LLM tool-calling → HTTP → metadata templating), with optional:
  - JSON Schema for tool args
  - JSON Schema for the HTTP response
  - Response-to-metadata mapping
  - Static reply templates (Jinja2)
  - “Use Codex for response” post-processing
- **Saved response recall + exports**:
  - Recall previously saved HTTP responses to answer follow-ups without re-calling the API.
  - Export prior results to CSV/JSON and serve via a short-lived download token URL.
- **Embeddable widget** (text chat) with client keys and a public WebSocket API.

## End‑user setup (no CLI)

On first launch, the Studio UI walks you through setup:
1) **OpenAI API key** (required for ASR, LLM, and TTS).
2) **ScrapingBee key** (optional; enables `web_search`).
3) **Data Agent** (optional; requires Docker installed). The UI will detect Docker and guide you. The first run pulls a prebuilt container image (override with `IGX_DATA_AGENT_IMAGE` if needed).

Default Data Agent image:
- `ghcr.io/mornville/data-agent:latest`

### Download URL host (exports)

`export_http_response` returns a `download_url` that points at the Studio server.

Set the base URL (host[:port] or full URL) using:

```bash
VOICEBOT_DOWNLOAD_BASE_URL=127.0.0.1:8000
```

### Test from UI / API

- Mic conversation test is in the React Studio bot page. Conversations are stored in the DB with `test_flag=true`.
- The bot page uses a **WebSocket** for mic audio + live status + streamed TTS audio:
  - `WS /ws/bots/<uuid>/talk`
  - Client protocol:
    - send JSON `{type:"start", req_id, conversation_id?, test_flag, speak}`
    - send binary **PCM16 @ 16kHz mono** (one or more frames)
    - send JSON `{type:"stop", req_id}`
  - Server events (JSON): `status`, `conversation`, `asr`, `text_delta`, `audio_wav`, `error`, `done`.
- REST endpoints:
- `POST /api/bots/<uuid>/talk/stream` (WAV upload → OpenAI ASR → streaming LLM → streaming TTS, NDJSON)
  - `POST /api/bots/<uuid>/chat` (text → LLM → optional TTS; JSON)

## Variables (Metadata Templating)

Prompts and replies can reference conversation metadata using `{{...}}` placeholders.

- Shorthand metadata lookup:
  - `{{.first_name}}`
  - `{{.user.profile.firstName}}` (nested)
- Other contexts (used by integrations):
  - `{{args.user_id}}` / `{{params.user_id}}` (tool-call arguments)
  - `{{response.data.first_name}}` (HTTP response JSON)

Where variables are applied:
- Bot `system_prompt` is rendered with metadata before every LLM call.
- Message history is rendered with metadata before sending to the LLM.
- Tool `next_reply` is rendered with metadata before sending back to the user.

Missing variables resolve to an empty string.

## Integration Tools (HTTP → Metadata)

You can attach HTTP API “integration tools” to a bot. The LLM can call them, and the backend will:
1) Execute the HTTP request.
2) Map selected response fields into conversation metadata (via a response mapper).
3) Return a tool result to the chat model, which then replies to the user.

Important:
- Raw HTTP responses are **not** sent back to the LLM.
- Only the mapped keys are merged into conversation metadata.

### Tool call schema (LLM-facing)

Each integration tool is exposed to the LLM as a function tool with a top-level shape:

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
- If “Use Codex for response” is enabled, `next_reply` is not required; the backend runs a Codex post-processor and returns a result to the main chat model to rephrase.

### Args schema (required args + JSON schema)

You can define `args` in two ways:
- **Required args list** (comma-separated in UI): the backend generates a permissive schema and requires those keys.
- **Args schema (JSON Schema)**: the backend uses this as the schema for `args`.

If both are present, required args are appended to the schema (they are not removed), so the integration still has all the fields needed to call the API.

### Static reply (optional, Jinja2) — takes priority

Integrations can be configured with an optional `static_reply_template`.

If `static_reply_template` is set:
- The backend ignores `next_reply`.
- The backend also ignores “Use Codex for response” (static template wins).
- The backend renders `static_reply_template` using **Jinja2** (supports `{% if %}`, `{% for %}`, etc).

Template context:
- `meta`: current conversation metadata (after response mapping)
- `response`: raw HTTP response JSON
- `args` / `params`: tool-call arguments (excluding `next_reply`)

### Use Codex for response (optional)

If enabled, the backend runs a separate Codex “one-shot” agent after the HTTP request:
1) The HTTP response is saved to a temp file.
2) The Codex model receives:
   - the response JSON schema (from the tool config, or a best-effort derived schema)
   - the response file path (so its generated Python script can read locally)
   - intent fields inside `args`:
     - `fields_required`: what fields are needed to build the response
     - `why_api_was_called`: why this API call happened (user intent)
3) Codex returns a Python script that extracts/aggregates the required info and writes a `result.txt`.
4) The tool result includes `codex_result_text` and file paths; the **main chat model** rephrases the final user-facing reply.

### Tool enabling/disabling

From the bot page in Studio:
- System tools can be toggled per bot (click “Update tools” to save). `set_metadata` cannot be disabled.
- Each HTTP integration tool has an `Enabled` toggle; disabled tools are not exposed to the LLM.

### Response mapper

Response mapper is a JSON object: `metadata_key -> template`.

Example:
```json
{
  "firstName": "{{response.data.first_name}}",
  "user.id": "{{response.data.id}}"
}
```

Notes:
- Keys with dots (e.g. `user.id`) are stored as nested objects in `metadata_json`.
- Templates can return raw JSON values if the whole value is a single placeholder.

### GET query params (explicit in URL)

For `GET` integrations, query parameters are taken from the URL template itself (the backend does not auto-append
unused tool args as query params). This prevents accidental duplication like:

`/individuals/1730174632?npi=1730174632`

Example:

`https://analytics.candorhealth.com/api/individuals/{{args.npi}}?include_reviews={{args.include_reviews}}`

### UI / API

- React Studio: Bot page → “Integrations (HTTP tools)” → Add integration.
- REST:
  - `GET /api/bots/<bot_uuid>/tools`
  - `POST /api/bots/<bot_uuid>/tools`
  - `PUT /api/bots/<bot_uuid>/tools/<tool_uuid>`
  - `DELETE /api/bots/<bot_uuid>/tools/<tool_uuid>`

## Context / Architecture (for handoff)

- Local loop (mic): `voicebot/dialog/session.py` (VAD → OpenAI ASR → OpenAI Responses stream → OpenAI TTS → playback)
- Studio server: `voicebot/web/app.py` (FastAPI JSON + WebSocket APIs)
- DB: SQLite by default (`VOICEBOT_DB_URL`), models in `voicebot/models.py` (`Bot`, `ApiKey`, `Conversation`, `ConversationMessage`)
- Secrets: encrypted with Fernet using `VOICEBOT_SECRET_KEY` (auto-generated unless overridden); UI only shows masked hints for keys.

## Production notes

- This is a **local** voice loop (mic + speakers). For web/telephony deployment, build a streaming server
  (WebRTC/WebSocket) around the same pipeline.
- For safety/quality, add policy checks and user authentication if you ship beyond local use.

## Maintainer packaging

These steps are for maintainers building distributable binaries (end users should not need a terminal).

### macOS (PyInstaller .app)

```bash
./scripts/package_macos.sh
```

Codesign + notarize (Developer ID required):

```bash
codesign --force --deep --options runtime --sign "Developer ID Application: <Team Name>" dist/IntelligravexStudio.app
xcrun notarytool submit dist/IntelligravexStudio.app --keychain-profile "<profile>" --wait
xcrun stapler staple dist/IntelligravexStudio.app
```

### Linux (AppImage)

```bash
./scripts/package_linux_appimage.sh
```

Output: `dist/IntelligravexStudio-x86_64.AppImage` (AppImage tool is downloaded by the script).

Note: AppImage packaging is currently scripted for x86_64 Linux. For arm64, swap in the appropriate appimagetool binary.

### Data Agent image (prebuilt)

Build the Data Agent container image and push it to your registry:

```bash
IGX_DATA_AGENT_IMAGE=ghcr.io/mornville/data-agent:latest ./scripts/build_data_agent_image.sh
docker push ghcr.io/mornville/data-agent:latest
```

## Embed (Text-only)

You can embed a bot on a third-party website using:
- A **Client Key** (generated in Studio → Keys → “Add client key”)
- A WebSocket endpoint that streams assistant replies
- A plug-and-play widget script (`/public/widget.js` or `/static/embed-widget.js`)

### Client Key

Client keys are separate from OpenAI keys and are used to authenticate embeds.
They can be restricted by allowed origins and (optionally) allowed bot ids.

### WebSocket API

Endpoint:
- `GET /public/v1/ws/bots/<bot_uuid>/chat?key=<client_key>&user_conversation_id=<stable_user_id>`

Messages (client → server):
- Start (assistant speaks first): `{"type":"start","req_id":"<uuid>"}`
- User reply: `{"type":"chat","req_id":"<uuid>","text":"Hello"}`

Events (server → client):
- `conversation`: includes `conversation_id` (internal UUID)
- `status`: `llm` / `idle`
- `text_delta`: streaming assistant tokens
- `done`: final turn text + `metrics` (model, token estimates, cost, latencies)

Tool calls/results are executed server-side but **not exposed** over the public WebSocket.

## Downloads (Exports)

`export_http_response` creates an export file (CSV/JSON) from a previously saved integration response, and returns:
- `download_url`: absolute URL to `GET /api/downloads/<token>`
- `download_token`: token used by the downloads endpoint

The downloads endpoint:
- `GET /api/downloads/<token>` (serves the exported file)

Configure the base host/URL for `download_url` via:
- `VOICEBOT_DOWNLOAD_BASE_URL` (default `127.0.0.1:8000`)

### Widget script

Load:
- `GET /public/widget.js` (same contents as `/static/embed-widget.js`)

Auto-init (script tag):
```html
<div id="igx-voicebot"></div>
<script
  src="http://127.0.0.1:8000/public/widget.js"
  data-target="#igx-voicebot"
  data-bot-id="BOT_UUID"
  data-api-key="igx_..."
  data-user-conversation-id="user_123"
></script>
```

Demo page:
- `examples/embed_demo.html`
