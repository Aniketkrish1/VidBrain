from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import uuid, asyncio, os, logging, shutil

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(override=True)

# import your main pipeline
from main import process_video, OUTPUT_VIDEO_NAME

logger = logging.getLogger(__name__)

# Startup cleanup - clear vector database cache if enabled
CLEAR_VECTOR_CACHE = os.getenv("CLEAR_VECTOR_CACHE", "true").lower() == "true"
if CLEAR_VECTOR_CACHE:
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db.pkl")
    if os.path.exists(vector_db_path):
        os.remove(vector_db_path)
        logger.info(f"🗑️  Cleared vector database cache on startup: {vector_db_path}")

# Clear video processing cache on startup
temp_dir = os.getenv("TEMP_DIR", "temp_processing")
if os.path.exists(temp_dir):
    import shutil
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"🗑️  Cleared video processing cache: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not clear temp directory: {e}")

# Clear old output files
output_dir = "outputs"
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith(('.mp4', '.mp3', '.wav')):
            try:
                os.remove(os.path.join(output_dir, file))
                logger.info(f"🗑️  Cleared old output: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")

app = FastAPI()

# Serve static assets for CSS and JS
app.mount("/static/css", StaticFiles(directory="static/css"), name="css")
app.mount("/static/js", StaticFiles(directory="static/js"), name="js")

# Where to save finished videos and uploads
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    whisper_model: str = Form("")
):
    """Start a new summarization job from YouTube URL."""
    if not youtube_url:
        return JSONResponse({"error": "YouTube URL is required"}, status_code=400)
    job_id = str(uuid.uuid4())
    if not output_name:
        output_name = OUTPUT_VIDEO_NAME
    output_path = OUTPUT_DIR / output_name
    jobs[job_id] = {
        "status": "queued", 
        "progress": 0, 
        "error": None, 
        "result": None, 
        "summaries": [],
        "query": query  # Store query for reference
    }

    async def run_job():
        try:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 0
            jobs[job_id]["stage"] = "Initializing..."
            jobs[job_id]["details"] = ""

            def progress_callback(stage: str, percent: int, details: str):
                jobs[job_id]["stage"] = stage
                jobs[job_id]["progress"] = percent
                jobs[job_id]["details"] = details
                logger.info(f"Progress update: {stage} - {percent}% - {details}")

            def summaries_callback(summaries_list):
                jobs[job_id]["summaries"] = summaries_list
                logger.info(f"Stored {len(summaries_list)} summaries for job {job_id}")

            # Run with extended timeout for long video processing
            await run_in_threadpool(
                process_video,
                None,  # no local file
                youtube_url,
                query,  # query optional
                str(output_path),
                whisper_model or None,  # whisper model optional
                progress_callback,  # progress callback
                summaries_callback,  # summaries callback
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = str(output_path)
            logger.info(f"Video processing completed successfully: {output_path}")
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["progress"] = 100

    asyncio.create_task(run_job())
    return {"job_id": job_id}

@app.post("/api/upload")
async def upload(
    video: UploadFile = File(...),
    query: str = Form(""),
    output_name: str = Form(""),
    whisper_model: str = Form("")
):
    """Start a new summarization job from uploaded video."""
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    file_ext = Path(video.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return JSONResponse({
            "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        }, status_code=400)
    
    # Create job
    job_id = str(uuid.uuid4())
    if not output_name:
        output_name = OUTPUT_VIDEO_NAME
    output_path = OUTPUT_DIR / output_name
    
    jobs[job_id] = {
        "status": "queued", 
        "progress": 0, 
        "error": None, 
        "result": None, 
        "summaries": [],
        "query": query
    }
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    
    try:
        # Save file to disk
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"Uploaded file saved: {upload_path} ({upload_path.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"Upload failed: {str(e)}"
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)
    
    async def run_job():
        try:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 0
            jobs[job_id]["stage"] = "Processing uploaded video..."
            jobs[job_id]["details"] = ""

            def progress_callback(stage: str, percent: int, details: str):
                jobs[job_id]["stage"] = stage
                jobs[job_id]["progress"] = percent
                jobs[job_id]["details"] = details
                logger.info(f"Progress update: {stage} - {percent}% - {details}")

            def summaries_callback(summaries_list):
                jobs[job_id]["summaries"] = summaries_list
                logger.info(f"Stored {len(summaries_list)} summaries for job {job_id}")

            # Run processing with uploaded video path
            await run_in_threadpool(
                process_video,
                str(upload_path),  # local video file
                None,  # no YouTube URL
                query,
                str(output_path),
                whisper_model or None,
                progress_callback,
                summaries_callback,
            )
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = str(output_path)
            logger.info(f"Video processing completed successfully: {output_path}")
            
            # Clean up uploaded file after processing
            try:
                upload_path.unlink()
                logger.info(f"Cleaned up uploaded file: {upload_path}")
            except Exception as e:
                logger.warning(f"Could not delete uploaded file: {e}")
                
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["progress"] = 100
            
            # Clean up on error
            try:
                if upload_path.exists():
                    upload_path.unlink()
            except:
                pass

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
