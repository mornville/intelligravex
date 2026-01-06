# Intelligravex VoiceBot

Continuous, local, end-to-end AI voice bot + Studio:
- **ASR:** Whisper (local)
- **LLM:** OpenAI `gpt-4o`
- **TTS:** Coqui **XTTS v2** (local)

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

## Quickstart

### 1) Create a venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[asr,tts]"
```

Notes:
- `sounddevice` requires PortAudio (macOS: `brew install portaudio`).
- Use a modern CPython (3.10+) on macOS (Homebrew Python recommended) to avoid SSL and wheel issues.
- Whisper and XTTS will download model weights on first run.
- Use headphones to avoid feedback (no echo cancellation by default).

### 2) Configure

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY
```

TTS voice:
- Set `VOICEBOT_SPEAKER_WAV` to a short clean voice sample (recommended), or
- Set `VOICEBOT_SPEAKER_ID` to one of the model speakers (see `voicebot doctor` output).

Language:
- Set ASR language to a 2-letter code (e.g. `en`, `hi`) or `auto` to let Whisper detect it.

Latency tuning:
- Use a smaller Whisper model (`VOICEBOT_WHISPER_MODEL=tiny` or `base`) for lower ASR latency.
- XTTS is compute-heavy on CPU; reduce first-audio delay by lowering `VOICEBOT_TTS_CHUNK_MAX_CHARS`.

### 3) Run

```bash
voicebot run
```

## CLI

```bash
voicebot --help
voicebot run --help
voicebot devices
voicebot doctor
voicebot tts-speakers
voicebot web --help
```

## Studio API server

1) Set a DB URL + secret key in `.env`:
   - `VOICEBOT_DB_URL=sqlite:///voicebot.db`
   - `VOICEBOT_SECRET_KEY=...` (Fernet key; used to encrypt API keys at rest)

2) Run:

```bash
voicebot web --host 127.0.0.1 --port 8000
# or:
./start.sh web --host 127.0.0.1 --port 8000
```

Then open the React Studio UI (see below) to create keys + bots, and run locally by UUID:

```bash
voicebot run --bot <uuid>
```

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
  - `POST /api/bots/<uuid>/talk/stream` (WAV upload → Whisper → streaming LLM → streaming TTS, NDJSON)
  - `POST /api/bots/<uuid>/chat` (text → LLM → optional TTS; JSON)

## React Studio (Two-Server Dev)

The repo also includes a React (Vite) frontend under `frontend/`.

1) Start the backend:

```bash
./start.sh web --host 127.0.0.1 --port 8000
```

2) Start the React frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

Notes:
- Backend JSON APIs used by the React app:
  - `GET/POST/PUT/DELETE /api/bots`
  - `GET/POST/DELETE /api/keys`
  - `GET /api/conversations` (paginated), `GET /api/conversations/<uuid>`
  - `GET/POST/PUT/DELETE /api/bots/<uuid>/tools` (integration tools)
  - `WS /ws/bots/<uuid>/talk` (mic audio + streamed status/audio)
- Dev CORS is enabled for local hosts/ports by default (including `localhost:5173`).
  - Override with `VOICEBOT_CORS_ORIGINS` (comma-separated), e.g.:
  - `VOICEBOT_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://ashutosh-jha-macbook-pro:3001`
  - React can also be pointed at a different backend using `VITE_BACKEND_URL` in `frontend/.env`.

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

### UI / API

- React Studio: Bot page → “Integrations (HTTP tools)” → Add integration.
- REST:
  - `GET /api/bots/<bot_uuid>/tools`
  - `POST /api/bots/<bot_uuid>/tools`
  - `PUT /api/bots/<bot_uuid>/tools/<tool_uuid>`
  - `DELETE /api/bots/<bot_uuid>/tools/<tool_uuid>`

## Context / Architecture (for handoff)

- Local loop (mic): `voicebot/dialog/session.py` (VAD → Whisper → OpenAI Responses stream → chunked XTTS → playback)
- Studio server: `voicebot/web/app.py` (FastAPI JSON + WebSocket APIs)
- DB: SQLite by default (`VOICEBOT_DB_URL`), models in `voicebot/models.py` (`Bot`, `ApiKey`, `Conversation`, `ConversationMessage`)
- Secrets: encrypted with Fernet using `VOICEBOT_SECRET_KEY` (`voicebot/crypto.py`); UI only shows masked hints for keys.

## Production notes

- This is a **local** voice loop (mic + speakers). For web/telephony deployment, build a streaming server
  (WebRTC/WebSocket) around the same pipeline.
- For safety/quality, add policy checks and user authentication if you ship beyond local use.

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
