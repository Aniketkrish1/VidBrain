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
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

# media libs
import torch
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# TTS: use pyttsx3 for cross-platform compatibility
import pyttsx3

# local utils (assumed present)
from utils import topic_clustering as tc
from utils import downloader as dl
from utils import transcriber as tb
from utils import summarizer as sz
from utils.database import VectorDB, parse_srt
from utils import scene_detector as sd
from utils.assmble_video import assemble_video

# ==== CONFIG ====
TEMP_DIR = os.getenv("TEMP_DIR", "temp_processing")
OUTPUT_VIDEO_NAME = os.getenv("OUTPUT_VIDEO_NAME", "summary_output.mp4")
# TTS_MODEL not used with pyttsx3, but keeping for compatibility
TTS_MODEL = os.getenv("TTS_MODEL", "default")
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

def cleanup():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info("Cleaned up temp files")
    except Exception as e:
        logger.warning("Cleanup failed: %s", e)

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

    # Filter hits to only include those that are highly relevant to the specific query
    # Keep only hits with score above a threshold and that contain query keywords
    query_lower = query.lower()
    filtered_hits = []
    for h in hits:
        score = h.get("score", 0)
        text = h.get("text", "").lower()

        # Keep hits with high similarity score OR that contain the query terms
        if score > 0.3 or query_lower in text:
            filtered_hits.append(h)

    logger.info(f"Query '{query}': found {len(hits)} total hits, {len(filtered_hits)} after filtering")

    # ensure shape
    result = []
    for h in filtered_hits:
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
def generate_voiceovers_from_summaries(summaries: Dict[int, Dict], tts_model: str = TTS_MODEL) -> Dict[int, str]:
    """
    Input summaries: {cluster_id: {"summary": str, "start": float, "end": float, ...}}
    Returns: {cluster_id: audio_path}
    """
    logger.info("Starting TTS generation for %d summaries", len(summaries))
    out = {}

    for cid, data in summaries.items():
        summary_text = data.get("summary", "")
        if not summary_text:
            logger.warning("Empty summary for cluster %s, skipping TTS", cid)
            continue

        file_path = os.path.join(TEMP_DIR, f"voiceover_{cid}.mp3")
        logger.info(f"Generating TTS for cluster {cid}: {len(summary_text)} chars")

        try:
            # Create a new engine instance for each cluster to avoid issues
            engine = pyttsx3.init()
            # Configure voice settings
            voices = engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available, otherwise use default
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break

            engine.setProperty('rate', 180)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

            # Generate TTS for this cluster
            engine.save_to_file(summary_text, file_path)
            engine.runAndWait()
            engine.stop()

            # Verify the file was created
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                out[cid] = file_path
                logger.info("TTS completed for cluster %s -> %s (%.1f KB)",
                           cid, file_path, os.path.getsize(file_path) / 1024)
            else:
                logger.error("TTS file not created or empty for cluster %s", cid)

        except Exception as e:
            logger.error("TTS failed for cluster %s: %s", cid, e)
            # Try to clean up any partial file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

        finally:
            # Ensure engine is properly closed
            try:
                engine.stop()
                del engine
            except:
                pass

    logger.info("TTS generation completed: %d/%d voiceovers created", len(out), len(summaries))
    return out

# ==== pipeline entry point ====
def process_video(video_path: Optional[str], youtube_url: Optional[str], query: Optional[str], output_path: str, whisper_model: Optional[str] = None, progress_callback: Optional[callable] = None, summaries_callback: Optional[callable] = None):
    """
    If youtube_url provided -> download video -> set video_path accordingly.
    If video_path provided -> use it.
    query: optional user query string (topic to extract); if None, auto-detect topics (clustering)
    """
    def update_progress(stage: str, percent: int, details: str = ""):
        if progress_callback:
            progress_callback(stage, percent, details)
        logger.info(f"Progress: {stage} - {percent}% - {details}")

    setup_dirs()

    # 1) Download or use local
    update_progress("download", 5, "Starting video download...")
    if youtube_url:
        try:
            logger.info("Downloading video from URL")
            video_path, audio_path = dl.download_and_extract_audio(youtube_url)
            update_progress("download", 15, "Video downloaded successfully")
        except Exception as e:
            logger.error("Download failed: %s", e)
            raise
    elif video_path:
        video_path = str(video_path)
        audio_path = os.path.join(TEMP_DIR, "audio.mp3")
        # try to extract audio if not provided
        if not os.path.exists(audio_path):
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path], check=False)
        update_progress("download", 15, "Using local video file")
    else:
        raise ValueError("Either video_path or youtube_url must be provided")

    # 2) Transcribe
    update_progress("transcription", 20, "Starting audio transcription...")
    logger.info("Transcribing audio")
    trans_data = tb.transcribe_audio(audio_path, whisper_model)
    # trans_data expected: { "sentences": [ {"text","start","end","words"}, ... ], "srt": "...", "language": "en" }
    sentences = trans_data.get("sentences", [])
    if not sentences:
        raise RuntimeError("Transcription produced no sentences")
    update_progress("transcription", 35, f"Transcription completed - {len(sentences)} sentences")

    # Build / load vector DB
    update_progress("vectordb", 40, "Building vector database...")
    logger.info("Building or loading vector DB")
    db = VectorDB(db_path=EMBEDDING_DB_PATH)
    # Build expects segments list - adapt parse_srt or sentences shape as required
    try:
        segments_for_db = [ {"text": s["text"], "start": s["start"], "end": s["end"]} for s in sentences ]
        db.build(segments_for_db)
        update_progress("vectordb", 45, "Vector database built")
    except Exception as e:
        logger.warning("VectorDB build warning: %s", e)

    # 3) Determine clusters (either via query retrieval or automatic clustering)
    update_progress("clustering", 50, "Analyzing topics and clustering...")
    if query and query.strip():
        logger.info("User query provided; retrieving relevant transcript segments")
        hits = retrieve_by_query(db, query, top_k=10)  # Get more hits for better coverage
        if not hits:
            logger.warning("No results from vector DB for query; falling back to clustering entire transcript")
            topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)
        else:
            # For specific queries, create ONE cluster with all relevant segments
            # Sort hits by start time to maintain temporal order
            sorted_hits = sorted(hits, key=lambda x: x['start'])

            # Filter out segments that are too short or don't contain relevant keywords
            query_keywords = set(query.lower().split())
            filtered_hits = []
            for hit in sorted_hits:
                text_lower = hit["text"].lower()
                # Include segments that contain query keywords or are highly relevant
                if any(keyword in text_lower for keyword in query_keywords) or len(hit["text"].split()) > 5:
                    filtered_hits.append(hit)

            # Create a single cluster containing all relevant segments
            topic_clusters = {0: [{"text": hit["text"], "start": hit["start"], "end": hit["end"]} for hit in filtered_hits]}
            logger.info(f"Created single cluster with {len(filtered_hits)} segments for query: {query} (filtered from {len(sorted_hits)} total hits)")
    else:
        logger.info("No query: clustering entire transcript")
        topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)

    if not topic_clusters:
        raise RuntimeError("No topic clusters found after retrieval/clustering")

    logger.info("Clusters prepared: %d", len(topic_clusters))
    update_progress("clustering", 55, f"Found {len(topic_clusters)} topic clusters")

    # 4) Scene detection (GPU-accelerated ffmpeg hybrid)
    update_progress("scenes", 60, "Detecting video scenes...")
    scenes = sd.detect_scenes(video_path)
    # scenes is list of (start,end) where end may be None for last segment
    update_progress("scenes", 65, f"Detected {len(scenes)} video scenes")

    # 5) Summarize clusters (classification + summarization; returns cluster_id -> {summary,start,end,sentences})
    # pass video duration so summarizer can normalize None -> duration
    with VideoFileClip(video_path) as v:
        video_duration = v.duration

    update_progress("summarization", 70, "Generating AI summaries...")
    summaries = sz.summarize_topics(topic_clusters, scenes, video_duration=video_duration, require_classification=True, use_openrouter=True)

    if not summaries:
        raise RuntimeError("Summarizer returned no summaries (all clusters filtered)")

    update_progress("summarization", 75, f"Generated {len(summaries)} summaries")

    # Store summaries for frontend display
    if summaries_callback:
        summary_list = []
        for cid, data in summaries.items():
            summary_list.append({
                "cluster_id": int(cid),
                "summary": data.get("summary", ""),
                "start": data.get("start", 0),
                "end": data.get("end", 0)
            })
        summaries_callback(summary_list)

    # 6) TTS generation
    update_progress("tts", 80, "Generating voiceovers...")
    voiceover_paths = generate_voiceovers_from_summaries(summaries, tts_model=TTS_MODEL)

    # sanity check
    missing = [cid for cid in summaries.keys() if cid not in voiceover_paths]
    if missing:
        logger.warning("Missing voiceovers for clusters: %s", missing)
        update_progress("tts", 85, f"Generated {len(voiceover_paths)}/{len(summaries)} voiceovers")

    update_progress("tts", 90, f"All {len(voiceover_paths)} voiceovers generated")

    # 7) Assemble condensed video
    update_progress("assembly", 95, "Assembling final video...")

    # Convert summaries format to topic_clusters format expected by assemble_video
    topic_clusters_for_assembly = {}
    for cid, summary_data in summaries.items():
        if cid in voiceover_paths:  # Only include clusters that have voiceovers
            topic_clusters_for_assembly[cid] = summary_data["sentences"]

    if not topic_clusters_for_assembly:
        raise RuntimeError("No voiceovers generated - cannot assemble video")

    logger.info("Assembling video with %d clusters (out of %d summaries)",
               len(topic_clusters_for_assembly), len(summaries))

    final = assemble_video(video_path, topic_clusters_for_assembly, voiceover_paths, output_path)
    update_progress("assembly", 100, f"Final video saved to {final}")

    # 8) cleanup
    cleanup()
    logger.info("Pipeline finished successfully")
    return final

# ==== CLI ====
if __name__ == "__main__":
    try:
        print("="*60)
        print("🎥  AI Video Summarizer")
        print("="*60)

        youtube_url = input("Enter YouTube URL (leave blank if using a local file): ").strip()
        local_video = ""
        if not youtube_url:
            local_video = input("Enter path to local video file: ").strip()

        query = input("Enter topic query (optional, leave blank to condense full video): ").strip()
        output_path = input("Enter output file name (default summary_output.mp4): ").strip() or OUTPUT_VIDEO_NAME

        if youtube_url:
            video_path = None
        else:
            video_path = local_video if local_video else None

        if not (youtube_url or video_path):
            print("You must enter either a YouTube URL or a local video path.")
            sys.exit(1)

        process_video(video_path, youtube_url, query if query else None, output_path)

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)

