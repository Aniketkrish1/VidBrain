#!/usr/bin/env python3
"""
main.py — orchestrates:
 - download (optional)
 - transcription (expects transcriber.transcribe_audio -> {sentences, srt, language})
 - topic retrieval (VectorDB search if user query; otherwise cluster_topics)
 - scene detection (scene_detector.detect_scenes)
 - summarization/classification (utils.summarizer.summarize_topics)
 - TTS generation
 - assemble condensed video
"""

import os
import sys
import json
import shutil
import logging
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

from dotenv import load_dotenv
load_dotenv()

# media libs
import torch
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

# TTS: use your existing TTS (Coqui TTS imported below). If your environment uses a different TTS, swap accordingly.
from TTS.api import TTS

# local utils (assumed present)
from utils import topic_clustering as tc
from utils import downloader as dl
from utils import transcriber as tb
from utils import summarizer as sz
from utils.database import VectorDB, parse_srt
from utils import scene_detector as sd
from utils import assmble_video as av
from utils import translator as tr
# ==== CONFIG ====
TEMP_DIR = os.getenv("TEMP_DIR", "temp_processing")
OUTPUT_VIDEO_NAME = os.getenv("OUTPUT_VIDEO_NAME", "summary_output.mp4")
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
EMBEDDING_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db.pkl")
KEEP_CLUSTER_PERCENTILE = float(os.getenv("KEEP_CLUSTER_PERCENTILE", "10.0"))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "2"))

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

# ==== helpers ====
def setup_dirs():
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Temp dir: %s", TEMP_DIR)
    # Ensure ffmpeg is available on PATH; audio/video operations require it.
    try:
        check_ffmpeg()
    except Exception as e:
        logger.warning("ffmpeg check failed: %s", e)

def cleanup():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info("Cleaned up temp files")
    except Exception as e:
        logger.warning("Cleanup failed: %s", e)


def check_ffmpeg():
    """Raise a helpful exception if ffmpeg is not available on PATH."""
    import shutil
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not found on PATH. Install ffmpeg and ensure it's available in your PATH.")

# safe bounding for None end times
def normalize_end(end: Optional[float], video_duration: float) -> float:
    return video_duration if end is None else float(end)

# ==== Retrieval utilities ====
def retrieve_by_query(db: VectorDB, query: str, top_k: int = 50) -> List[Dict]:
    """
    Use VectorDB search to fetch top_k transcript segments relevant to `query`.
    Expected that db.search returns list of dicts with keys: 'text','start','end','score' (score optional).
    """
    hits = db.search(query, top_k=top_k)  # adjust per your VectorDB API
    # ensure shape
    result = []
    for h in hits:
        # handle both dict and tuple shaped returns
        if isinstance(h, dict):
            text = h.get("text") or h.get("content") or ""
            start = float(h.get("start", 0.0))
            end = float(h.get("end", start))
            result.append({"text": text, "start": start, "end": end})
        elif isinstance(h, (list, tuple)) and len(h) >= 3:
            text, start, end = h[0], float(h[1]), float(h[2])
            result.append({"text": text, "start": start, "end": end})
    # sort temporally
    return sorted(result, key=lambda x: x["start"])

def merge_adjacent_segments(segments: List[Dict], gap_threshold: float = 2.0) -> List[List[Dict]]:
    """
    Merge segments into groups: if consecutive segments are within `gap_threshold` seconds,
    they are merged into same logical group. Returns list of grouped segment-lists.
    """
    if not segments:
        return []
    groups = []
    current = [segments[0]]
    for s in segments[1:]:
        prev = current[-1]
        if s["start"] <= prev["end"] + gap_threshold:
            current.append(s)
        else:
            groups.append(current)
            current = [s]
    if current:
        groups.append(current)
    return groups

def make_topic_clusters_from_db_hits(groups: List[List[Dict]]) -> Dict[int, List[Dict]]:
    """
    Convert grouped hits into a cluster dict {0: [sentences...], 1: [...]}
    Each sentence item keeps text,start,end.
    """
    clusters = {}
    for idx, grp in enumerate(groups):
        # flatten into sentence-like dicts (keep original items)
        clusters[idx] = [ {"text": s["text"], "start": s["start"], "end": s["end"]} for s in grp ]
    return clusters

# ==== TTS helper ====
def generate_voiceovers_from_summaries(
    summaries: Dict[int, Dict], 
    tts_model: str = TTS_MODEL,
    target_language: Optional[str] = None
) -> Dict[int, str]:
    """
    Input summaries: {cluster_id: {"summary": str, "start": float, "end": float, ...}}
    Returns: {cluster_id: audio_path}
    
    Uses gTTS for all languages (reliable and supports 50+ languages).
    Falls back to Coqui TTS only if gTTS fails.
    """
    logger.info("Starting TTS generation for %d summaries", len(summaries))
    
    # Determine language for gTTS
    lang_name = target_language.lower() if target_language else "english"
    
    # Try gTTS first for all languages (it's more reliable)
    use_gtts = True
    logger.info(f"Using gTTS for language: {lang_name}")
    
    # Initialize Coqui TTS as fallback
    tts = None
    
    out = {}
    for cid, data in summaries.items():
        summary_text = data.get("summary", "")
        if not summary_text:
            logger.warning("Empty summary for cluster %s, skipping TTS", cid)
            continue
        
        file_path = os.path.join(TEMP_DIR, f"voiceover_{cid}.mp3")  # gTTS outputs MP3
        success = False
        
        try:
            if use_gtts:
                # Try gTTS first (supports multiple languages, very reliable)
                success = tr.generate_tts_gtts(
                    summary_text, 
                    file_path, 
                    target_language=lang_name
                )
                if success:
                    logger.info("gTTS written for cluster %s -> %s", cid, file_path)
                else:
                    logger.warning("gTTS failed for cluster %s, falling back to Coqui", cid)
            
            # Fallback to Coqui TTS if gTTS failed
            if not success:
                # Coqui only supports English, so only use as fallback
                file_path = os.path.join(TEMP_DIR, f"voiceover_{cid}.wav")  # Coqui outputs WAV
                if tts is None:
                    tts = TTS(model_name=tts_model, progress_bar=False, gpu=torch.cuda.is_available())
                tts.tts_to_file(text=summary_text, file_path=file_path)
                success = True
                logger.info("Coqui TTS written for cluster %s -> %s", cid, file_path)
            
            if success:
                out[cid] = file_path
                
        except Exception as e:
            logger.error("TTS failed for cluster %s: %s", cid, e)
    
    return out

# ==== assemble (uses summaries dict) ====
def assemble_from_summaries(video_path: str, summaries: Dict[int, Dict], voiceover_paths: Dict[int, str], output_path: str) -> str:
    logger.info("Assembling final video using %d summaries", len(summaries))
    original = None
    segments = []
    try:
        original = VideoFileClip(video_path)
        duration = original.duration
        for cid in sorted(summaries.keys()):
            data = summaries[cid]
            start = float(data.get("start", 0.0))
            end = data.get("end", None)
            end = normalize_end(end, duration)

            if end <= start:
                logger.warning("Cluster %s has non-positive duration (%s-%s), skipping", cid, start, end)
                continue

            voice_path = voiceover_paths.get(cid)
            if not voice_path or not os.path.exists(voice_path):
                raise FileNotFoundError(f"Voiceover for cluster {cid} not found: {voice_path}")

            # Load voiceover to get its duration
            audio_clip = AudioFileClip(voice_path)
            voiceover_duration = audio_clip.duration
            
            logger.info("Cluster %s: Original video %.2f-%.2f (%.1fs), Voiceover: %.1fs", 
                       cid, start, end, end - start, voiceover_duration)

            # KEY OPTIMIZATION: Use voiceover duration, NOT original video duration!
            # Extract video starting at timestamp, but only for voiceover length
            # This ensures output = sum of voiceover lengths (condensed!)
            video_extract_duration = min(voiceover_duration, end - start)
            
            clip = original.subclip(start, start + video_extract_duration)
            
            # If voiceover is longer than available video, loop video to match
            if voiceover_duration > video_extract_duration:
                clip = clip.loop(duration=voiceover_duration)
            # If voiceover is shorter, trim video to match (already done above)
            
            clip = clip.set_audio(audio_clip)
            segments.append(clip)

        if not segments:
            raise RuntimeError("No segments to concatenate; nothing to assemble.")

        final = concatenate_videoclips(segments, method="compose")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac",
                              temp_audiofile=os.path.join(TEMP_DIR, "temp-audio.m4a"),
                              remove_temp=True, verbose=False, logger=None)

        logger.info("Wrote final video: %s", output_path)
        return output_path

    finally:
        # try to close everything to release handles
        try:
            if original:
                original.close()
        except Exception:
            pass
        for s in segments:
            try:
                s.close()
            except Exception:
                pass
        gc.collect()

# ==== pipeline entry point ====
def process_video(
    video_path: Optional[str], 
    youtube_url: Optional[str], 
    query: Optional[str], 
    output_path: str,
    target_language: Optional[str] = None,
    progress_hook: Optional[Callable[[int, Optional[str], Optional[str]], None]] = None
):
    """
    If youtube_url provided -> download video -> set video_path accordingly.
    If video_path provided -> use it.
    query: optional user query string (topic to extract); if None, auto-detect topics (clustering)
    target_language: optional target language for translation and TTS (e.g., 'hindi', 'tamil', 'en-IN')
    """
    def _update_progress(percent: Optional[int], status: Optional[str] = None, stage: Optional[str] = None):
        if progress_hook:
            try:
                progress_hook(percent or 0, status, stage)
            except Exception:
                pass

    setup_dirs()
    _update_progress(0, "queued", "start")

    # 1) Download or use local
    if youtube_url:
        try:
            logger.info("Downloading video from URL")
            video_path, audio_path = dl.download_and_extract_audio(youtube_url)
            _update_progress(10, "processing", "download")
        except Exception as e:
            logger.error("Download failed: %s", e)
            raise
    elif video_path:
        video_path = str(video_path)
        audio_path = os.path.join(TEMP_DIR, "audio.mp3")
        # try to extract audio if not provided
        if not os.path.exists(audio_path):
            import subprocess
            try:
                # Ensure ffmpeg exists (setup_dirs also checks, but be defensive)
                check_ffmpeg()
                subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path], check=True)
            except subprocess.CalledProcessError as e:
                logger.error("ffmpeg failed to extract audio: %s", e)
                raise RuntimeError("ffmpeg failed to extract audio from local video") from e
            except Exception as e:
                logger.error("Audio extraction setup failed: %s", e)
                raise
    else:
        raise ValueError("Either video_path or youtube_url must be provided")

    # 2) Transcribe
    logger.info("Transcribing audio")
    tb.load_models()
    trans_data = tb.transcribe_audio(audio_path)
    _update_progress(30, "processing", "transcribe")
    print(trans_data,file=open("aaa/transcription_data.json","w"))
    # trans_data expected: { "sentences": [ {"text","start","end","words"}, ... ], "srt": "...", "language": "en" }
    sentences = trans_data.get("sentences", [])
    with open("aaa/sentences.json","w") as f:
        json.dump(sentences,f,indent=2)
    if not sentences:
        raise RuntimeError("Transcription produced no sentences")

    # Build / load vector DB
    logger.info("Building or loading vector DB")
    db = VectorDB(db_path=EMBEDDING_DB_PATH)
    # Build expects segments list - adapt parse_srt or sentences shape as required
    try:
        segments_for_db = [ {"text": s["text"], "start": s["start"], "end": s["end"]} for s in sentences ]
        db.build(segments_for_db)
        _update_progress(45, "processing", "embed-db")
    except Exception as e:
        logger.warning("VectorDB build warning: %s", e)
        # If build failed, make sure we don't accidentally use an older/stale DB that
        # was loaded from disk earlier. Clear embeddings and metadata so searches
        # will return no hits and the pipeline will fall back to clustering.
        try:
            db.embeddings = None
            db.metadata = []
        except Exception:
            pass

    # 3) Determine clusters (either via query retrieval or automatic clustering)
    if query and query.strip():
        logger.info("User query provided; retrieving relevant transcript segments")
        hits = retrieve_by_query(db, query, top_k=15)
        print("Retrieved segments:", len(hits))
        with open("aaa/retrieved_segments.json","w") as f:
            json.dump(hits,f,indent=2)
        if not hits:
            logger.warning("No results from vector DB for query; falling back to clustering entire transcript")
            topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)
        else:
            # merge temporally proximate hits into groups; this keeps explanation parts separated by filler merged
            groups = merge_adjacent_segments(hits, gap_threshold=3.0)
            with open("aaa/retrieved_groups.json","w") as f:
                json.dump(groups,f,indent=2)
            topic_clusters = make_topic_clusters_from_db_hits(groups)
            
    else:
        logger.info("No query: clustering entire transcript")
        topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)
    _update_progress(55, "processing", "cluster")
    with open("aaa/topic_clusters_from_retrieval.json","w") as f:
                json.dump(topic_clusters,f,indent=2)
    if not topic_clusters:
        raise RuntimeError("No topic clusters found after retrieval/clustering")

    logger.info("Clusters prepared: %d", len(topic_clusters))
    print("Topic clusters:", len(topic_clusters))
    # 4) Scene detection (GPU-accelerated ffmpeg hybrid)
    scenes = sd.detect_scenes(video_path)
    # scenes is list of (start,end) where end may be None for last segment
    logger.info("Detected %d scenes", len(scenes))
    _update_progress(60, "processing", "scene-detect")

    # 5) Summarize clusters (classification + summarization; returns cluster_id -> {summary,start,end,sentences})
    # pass video duration so summarizer can normalize None -> duration
    clip = VideoFileClip(video_path)
    video_duration = clip.duration
    clip.close()

    logger.info("Video duration: %.2f", video_duration)
    summaries = sz.summarize_topics(
        query or "",
        topic_clusters, 
        scenes, 
        video_duration=video_duration, 
        require_classification=True, 
        use_openrouter=True,
        target_language=target_language
    )
    _update_progress(75, "processing", "summarize")
    with open("aaa/summaries.json","w") as f:
        json.dump(summaries,f,indent=2)
    if not summaries:
        raise RuntimeError("Summarizer returned no summaries (all clusters filtered)")

    # 6) TTS generation
    voiceover_paths = generate_voiceovers_from_summaries(
        summaries, 
        tts_model=TTS_MODEL,
        target_language=target_language
    )
    _update_progress(88, "processing", "tts")

    # sanity check
    missing = [cid for cid in summaries.keys() if cid not in voiceover_paths]
    if missing:
        logger.warning("Missing voiceovers for clusters: %s", missing)

    # 7) Assemble condensed video
    final = av.assemble_video(video_path, summaries, voiceover_paths, output_path)
    _update_progress(95, "processing", "assemble")

    # 8) cleanup
    cleanup()
    _update_progress(100, "completed", "done")
    logger.info("Pipeline finished successfully")
    return final

# ==== CLI ====
if __name__ == "__main__":
    try:
        print("="*60)
        print("🎥  AI Video Summarizer with Multilingual Support")
        print("="*60)

        youtube_url = input("Enter YouTube URL (leave blank if using a local file): ").strip()
        local_video = ""
        if not youtube_url:
            local_video = input("Enter path to local video file: ").strip()

        query = input("Enter topic query (optional, leave blank to condense full video): ").strip()
        
        # Language selection
        print("\nSupported languages:")
        supported_langs = tr.get_supported_languages()
        for lang_name, lang_code in supported_langs.items():
            print(f"  - {lang_name.capitalize()} ({lang_code})")
        
        target_lang = input("\nEnter target language (leave blank for English): ").strip()
        if target_lang and not tr.get_language_code(target_lang):
            print(f"Warning: '{target_lang}' not recognized, defaulting to English")
            target_lang = None
        
        output_path = input("Enter output file name (default summary_output.mp4): ").strip() or OUTPUT_VIDEO_NAME

        if youtube_url:
            video_path = None
        else:
            video_path = local_video if local_video else None

        if not (youtube_url or video_path):
            print("You must enter either a YouTube URL or a local video path.")
            sys.exit(1)

        process_video(
            video_path, 
            youtube_url, 
            query if query else None, 
            output_path,
            target_language=target_lang if target_lang else None
        )

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)

