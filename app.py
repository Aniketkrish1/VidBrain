from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import uuid, asyncio, os

# import your main pipeline
from main import process_video, OUTPUT_VIDEO_NAME

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
    youtube_url: str = Form(...),
    output_name: str = Form(""),
    query: str = Form("")
):
    """Start a new summarization job."""
    job_id = str(uuid.uuid4())
    if not output_name:
        output_name = OUTPUT_VIDEO_NAME
    output_path = OUTPUT_DIR / output_name
    jobs[job_id] = {"status": "queued", "progress": 0, "error": None, "result": None}

    async def run_job():
        try:
            jobs[job_id]["status"] = "processing"
            await run_in_threadpool(
                process_video,
                None,  # no local file
                youtube_url,
                query,  # query optional
                str(output_path),
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = str(output_path)
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["progress"] = 100

    asyncio.create_task(run_job())
    return {"job_id": job_id}

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
