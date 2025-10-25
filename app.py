from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import uuid, asyncio, os, logging, tempfile, shutil

# import your main pipeline
from main import process_video, OUTPUT_VIDEO_NAME

# Import simple summarizer for direct transcript processing
from simple_summarizer import process_simple_query
from utils import transcriber as tb

logger = logging.getLogger(__name__)

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
    query: str = Form(""),
    whisper_model: str = Form("")
):
    """Start a new summarization job."""
    job_id = str(uuid.uuid4())
    if not output_name:
        output_name = OUTPUT_VIDEO_NAME
    output_path = OUTPUT_DIR / output_name
    jobs[job_id] = {"status": "queued", "progress": 0, "error": None, "result": None, "summaries": []}

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


@app.post("/api/simple-summary")
async def simple_summary(
    youtube_url: str = Form(""),
    video_file: UploadFile = File(None),
    query: str = Form(""),
    language: str = Form("english")  # For future multilingual support
):
    """
    Simple endpoint: Extract transcript → Pass to OpenRouter → Return summary.
    No video processing, no clustering - just clean summary.
    """
    if not query or not query.strip():
        return JSONResponse({"error": "Query is required"}, status_code=400)
    
    try:
        # 1. Get video file (either upload or YouTube)
        video_path = None
        temp_files = []
        
        if video_file and video_file.filename:
            # Handle uploaded file
            suffix = Path(video_file.filename).suffix
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_files.append(temp_video.name)
            
            with open(temp_video.name, "wb") as f:
                shutil.copyfileobj(video_file.file, f)
            video_path = temp_video.name
            
        elif youtube_url and youtube_url.strip():
            # Handle YouTube URL - try simple approach first
            return JSONResponse({
                "error": "YouTube download requires dependencies. Please upload a video file instead."
            }, status_code=400)
        else:
            return JSONResponse({"error": "Either video file or YouTube URL is required"}, status_code=400)
        
        # 2. Extract audio from video
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        temp_files.append(audio_path)
        
        # Simple ffmpeg command to extract audio
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return JSONResponse({
                "error": f"Audio extraction failed: {result.stderr}"
            }, status_code=500)
        
        # 3. Transcribe audio using Whisper
        logger.info(f"Transcribing audio for query: '{query}'")
        trans_data = await run_in_threadpool(tb.transcribe_audio, audio_path, "base")
        sentences = trans_data.get("sentences", [])
        
        if not sentences:
            return JSONResponse({"error": "Transcription failed - no text found"}, status_code=500)
        
        # 4. Generate summary using simple summarizer
        logger.info(f"Generating summary for {len(sentences)} sentences")
        summary_result = await run_in_threadpool(process_simple_query, sentences, query)
        
        # 5. Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # 6. Return result
        return JSONResponse({
            "success": True,
            "summary": summary_result["summary"],
            "query": query,
            "confidence": summary_result["confidence"],
            "transcript_length": len(sentences),
            "error": summary_result.get("error")
        })
        
    except Exception as e:
        # Clean up temp files on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        logger.error(f"Simple summary failed: {e}")
        return JSONResponse({
            "error": f"Processing failed: {str(e)}"
        }, status_code=500)
