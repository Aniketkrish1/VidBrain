from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import uuid, asyncio, os

# import your main pipeline
from main import process_video, OUTPUT_VIDEO_NAME, TEMP_DIR

app = FastAPI()

# Serve static assets for CSS and JS
app.mount("/static/css", StaticFiles(directory="static/css"), name="css")
app.mount("/static/js", StaticFiles(directory="static/js"), name="js")

# Where to save finished videos
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

jobs = {}

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/start")
async def start(
    youtube_url: str = Form(""),
    output_name: str = Form(""),
    query: str = Form(""),
    target_language: str = Form("en"),
    local_file: UploadFile = File(None)
):
    """Start a new summarization job."""
    job_id = str(uuid.uuid4())
    if not output_name:
        output_name = OUTPUT_VIDEO_NAME
    output_path = OUTPUT_DIR / output_name
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": "queued", "error": None, "result": None}
    # Accept local uploaded file and save to temp if provided
    saved_local_path = None
    if local_file is not None:
        try:
            Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
            suffix = Path(str(local_file.filename)).suffix or ""
            saved_local_path = Path(TEMP_DIR) / f"uploaded_{job_id}{suffix}"
            with open(saved_local_path, "wb") as out_f:
                # local_file.file is a SpooledTemporaryFile
                import shutil
                shutil.copyfileobj(local_file.file, out_f)
            # ensure file flushed
            out_f.close()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"failed to save uploaded file: {e}"
            return JSONResponse({"job_id": job_id, "error": jobs[job_id]["error"]}, status_code=500)

    async def run_job():
        try:
            jobs[job_id]["status"] = "processing"
            # progress updater for the background pipeline
            def _progress_updater(percent, status=None, stage=None):
                j = jobs.get(job_id)
                if not j:
                    return
                try:
                    if percent is not None:
                        j["progress"] = int(percent)
                except Exception:
                    pass
                if status:
                    j["status"] = status
                if stage:
                    j["stage"] = stage

            # choose inputs: prefer uploaded file, else youtube_url
            video_input = None
            if saved_local_path:
                video_input = str(saved_local_path)
            else:
                video_input = None

            await run_in_threadpool(
                process_video,
                video_input,
                youtube_url,
                query,
                str(output_path),
                target_language,
                _progress_updater
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = str(output_path)
            # Try to attach a textual summary produced by the pipeline, if available
            try:
                import json
                from pathlib import Path
                summary_path = Path("aaa") / "summaries.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        jobs[job_id]["summary"] = json.load(sf)
                else:
                    jobs[job_id]["summary"] = None
            except Exception:
                jobs[job_id]["summary"] = None
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["progress"] = 100

    asyncio.create_task(run_job())
    return {"job_id": job_id}


@app.get("/api/summary/{job_id}")
async def get_summary(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "notfound"}, status_code=404)
    # summary may be None or dict
    summary = job.get("summary")
    if summary is None:
        return JSONResponse({"error": "no_summary"}, status_code=404)
    return JSONResponse(summary)

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "notfound"}, status_code=404)
    return job

@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("result"):
        return JSONResponse({"error": "No result"}, status_code=404)
    return FileResponse(job["result"], filename=Path(job["result"]).name, media_type="video/mp4")
