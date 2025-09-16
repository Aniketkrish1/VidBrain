VidBrain Web UI
================

Run locally
-----------

1) Create and activate a virtual environment (recommended).

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set optional environment variables (or use the form inputs):

```bash
# .env (optional)
YOUTUBE_URL=
WHISPER_MODEL=base
SPACY_MODEL=en_core_web_sm
OUTPUT_VIDEO_NAME=summary.mp4
TEMP_DIR=.vidbrain_tmp
```

4) Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open the UI at `http://localhost:8000`.


Deploy
------

- Uvicorn/Gunicorn example command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Dockerfile sketch:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes
-----

- The UI triggers the existing pipeline in a background task and exposes status and download endpoints.
- Outputs are written under `app/outputs/`.
- Ensure models (Whisper, spaCy) are available or downloadable on first run.

