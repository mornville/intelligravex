# Intelligravex VoiceBot

Continuous, local, end-to-end AI voice bot:
- **ASR:** Whisper (local)
- **LLM:** OpenAI `gpt-4o`
- **TTS:** Coqui **XTTS v2** (local)

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

## VoiceBot Studio (Web UI)

1) Set a DB URL + secret key in `.env`:
   - `VOICEBOT_DB_URL=sqlite:///voicebot.db`
   - `VOICEBOT_SECRET_KEY=...` (Fernet key; used to encrypt API keys at rest)

2) Run:

```bash
voicebot web --host 127.0.0.1 --port 8000
# or:
./start.sh web --host 127.0.0.1 --port 8000
```

Then open the UI, create keys + bots, and run locally by UUID:

```bash
voicebot run --bot <uuid>
```

### Test from UI / API

- Mic conversation test is on each bot page (`/bots/<uuid>`). Conversations are stored in the DB with `test_flag=true`.
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
  - `WS /ws/bots/<uuid>/talk` (mic audio + streamed status/audio)
- Dev CORS is enabled for `localhost:5173` by default. Override with:
  - `VOICEBOT_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
  - React can also be pointed at a different backend using `VITE_BACKEND_URL` in `frontend/.env`.

## Context / Architecture (for handoff)

- Local loop (mic): `voicebot/dialog/session.py` (VAD → Whisper → OpenAI Responses stream → chunked XTTS → playback)
- Studio server: `voicebot/web/app.py` (FastAPI + Jinja templates + NDJSON streaming)
- DB: SQLite by default (`VOICEBOT_DB_URL`), models in `voicebot/models.py` (`Bot`, `ApiKey`, `Conversation`, `ConversationMessage`)
- Secrets: encrypted with Fernet using `VOICEBOT_SECRET_KEY` (`voicebot/crypto.py`); UI only shows masked hints for keys.

## Production notes

- This is a **local** voice loop (mic + speakers). For web/telephony deployment, build a streaming server
  (WebRTC/WebSocket) around the same pipeline.
- For safety/quality, add policy checks and user authentication if you ship beyond local use.
