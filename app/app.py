from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import uuid
import asyncio
import logging

# Import existing pipeline entry points
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from main import setup_directories, cleanup_files, segment_and_realign_sentences, write_srt, generate_voiceovers, assemble_video
from utils import downloader as dl
from utils import transcriber as tb
from utils import topic_clustering as tc
from utils import summarizer as sz
from utils.database import VectorDB, parse_srt

logger = logging.getLogger("uvicorn")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VidBrain", description="AI Video Summarizer UI", version="1.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory job store for simplicity
jobs = {}


async def run_pipeline(job_id: str, youtube_url: str, whisper_model: str | None, spacy_model: str | None, output_name: str | None):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 0

        # Reflect env overrides for existing code that reads env vars
        if youtube_url:
            os.environ["YOUTUBE_URL"] = youtube_url
        if whisper_model:
            os.environ["WHISPER_MODEL"] = whisper_model
        if spacy_model:
            os.environ["SPACY_MODEL"] = spacy_model
        if output_name:
            os.environ["OUTPUT_VIDEO_NAME"] = output_name

        setup_directories()
        jobs[job_id]["progress"] = 5

        # 1) Download and extract
        video_path, audio_path = dl.download_and_extract_audio(os.getenv("YOUTUBE_URL"))
        jobs[job_id]["progress"] = 25

        # 2) Transcribe
        transcription_data = tb.transcribe_audio(audio_path)
        jobs[job_id]["progress"] = 40

        # 3) Segment + SRT
        sentences = segment_and_realign_sentences(transcription_data)
        write_srt(sentences, str(OUTPUTS_DIR / f"{job_id}.srt"))
        jobs[job_id]["progress"] = 50

        # 3b) Build vector DB
        segments = parse_srt(str(OUTPUTS_DIR / f"{job_id}.srt"))
        db = VectorDB(db_path=str(OUTPUTS_DIR / f"{job_id}.pkl"))
        db.build(segments)
        jobs[job_id]["progress"] = 60

        # 4) Clustering
        topic_clusters = tc.cluster_topics(sentences)
        if not topic_clusters:
            raise RuntimeError("No topic clusters found")
        jobs[job_id]["progress"] = 70

        # 5) Summaries
        summaries = sz.summarize_topics(topic_clusters)
        jobs[job_id]["progress"] = 80

        # 6) Voiceovers
        voiceover_paths = generate_voiceovers(summaries)
        jobs[job_id]["progress"] = 90

        # 7) Assemble
        output_video_path = OUTPUTS_DIR / (output_name or f"summary_{job_id}.mp4")
        final_video_path = assemble_video(
            video_path,
            topic_clusters,
            voiceover_paths,
            str(output_video_path)
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = str(final_video_path)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
    finally:
        cleanup_files()


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def start_job(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    youtube_url = str(form.get("youtube_url") or "").strip()
    whisper_model = str(form.get("whisper_model") or "").strip() or None
    spacy_model = str(form.get("spacy_model") or "").strip() or None
    output_name = str(form.get("output_name") or "").strip() or None

    if not youtube_url:
        return JSONResponse({"error": "youtube_url is required"}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0}

    background_tasks.add_task(run_pipeline, job_id, youtube_url, whisper_model, spacy_model, output_name)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed" or not job.get("result"):
        return JSONResponse({"error": "result not available"}, status_code=404)
    return FileResponse(path=job["result"], filename=Path(job["result"]).name, media_type="video/mp4")


