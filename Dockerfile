FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libportaudio2 \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY voicebot /app/voicebot

# NOTE: XTTS/Whisper pull large deps (torch) and model weights.
RUN pip install -U pip && pip install -e ".[asr,tts,web]"

CMD ["voicebot", "run"]
