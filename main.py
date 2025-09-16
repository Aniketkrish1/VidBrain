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
def generate_voiceovers_from_summaries(summaries: Dict[int, Dict], tts_model: str = TTS_MODEL) -> Dict[int, str]:
    """
    Input summaries: {cluster_id: {"summary": str, "start": float, "end": float, ...}}
    Returns: {cluster_id: audio_path}
    """
    logger.info("Starting TTS generation for %d summaries", len(summaries))
    tts = TTS(model_name=tts_model, progress_bar=False, gpu=torch.cuda.is_available())
    out = {}
    for cid, data in summaries.items():
        summary_text = data.get("summary", "")
        if not summary_text:
            logger.warning("Empty summary for cluster %s, skipping TTS", cid)
            continue
        file_path = os.path.join(TEMP_DIR, f"voiceover_{cid}.wav")
        try:
            tts.tts_to_file(text=summary_text, file_path=file_path)
            out[cid] = file_path
            logger.info("TTS written for cluster %s -> %s", cid, file_path)
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

            logger.info("Cutting cluster %s: %.2f - %.2f", cid, start, end)
            clip = original.subclip(start, end)
            audio_clip = AudioFileClip(voice_path)

            # sync durations
            if audio_clip.duration < clip.duration:
                clip = clip.subclip(0, audio_clip.duration)
            elif audio_clip.duration > clip.duration:
                clip = clip.loop(duration=audio_clip.duration)

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
def process_video(video_path: Optional[str], youtube_url: Optional[str], query: Optional[str], output_path: str):
    """
    If youtube_url provided -> download video -> set video_path accordingly.
    If video_path provided -> use it.
    query: optional user query string (topic to extract); if None, auto-detect topics (clustering)
    """
    setup_dirs()

    # 1) Download or use local
    if youtube_url:
        try:
            logger.info("Downloading video from URL")
            video_path, audio_path = dl.download_and_extract_audio(youtube_url)
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
    else:
        raise ValueError("Either video_path or youtube_url must be provided")

    # 2) Transcribe
    logger.info("Transcribing audio")
    trans_data = tb.transcribe_audio(audio_path)
    # trans_data expected: { "sentences": [ {"text","start","end","words"}, ... ], "srt": "...", "language": "en" }
    sentences = trans_data.get("sentences", [])
    if not sentences:
        raise RuntimeError("Transcription produced no sentences")

    # Build / load vector DB
    logger.info("Building or loading vector DB")
    db = VectorDB(db_path=EMBEDDING_DB_PATH)
    # Build expects segments list - adapt parse_srt or sentences shape as required
    try:
        segments_for_db = [ {"text": s["text"], "start": s["start"], "end": s["end"]} for s in sentences ]
        db.build(segments_for_db)
    except Exception as e:
        logger.warning("VectorDB build warning: %s", e)

    # 3) Determine clusters (either via query retrieval or automatic clustering)
    if query and query.strip():
        logger.info("User query provided; retrieving relevant transcript segments")
        hits = retrieve_by_query(db, query, top_k=5)
        if not hits:
            logger.warning("No results from vector DB for query; falling back to clustering entire transcript")
            topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)
        else:
            # merge temporally proximate hits into groups; this keeps explanation parts separated by filler merged
            groups = merge_adjacent_segments(hits, gap_threshold=3.0)
            topic_clusters = make_topic_clusters_from_db_hits(groups)
    else:
        logger.info("No query: clustering entire transcript")
        topic_clusters = tc.cluster_topics(sentences, embedding_model=None, min_cluster_size=MIN_CLUSTER_SIZE, keep_percentile=KEEP_CLUSTER_PERCENTILE)

    if not topic_clusters:
        raise RuntimeError("No topic clusters found after retrieval/clustering")

    logger.info("Clusters prepared: %d", len(topic_clusters))

    # 4) Scene detection (GPU-accelerated ffmpeg hybrid)
    scenes = sd.detect_scenes(video_path)
    # scenes is list of (start,end) where end may be None for last segment

    # 5) Summarize clusters (classification + summarization; returns cluster_id -> {summary,start,end,sentences})
    # pass video duration so summarizer can normalize None -> duration
    with VideoFileClip(video_path) as v:
        video_duration = v.duration

    summaries = sz.summarize_topics(topic_clusters, scenes, video_duration=video_duration, require_classification=True, use_openrouter=True)

    if not summaries:
        raise RuntimeError("Summarizer returned no summaries (all clusters filtered)")

    # 6) TTS generation
    voiceover_paths = generate_voiceovers_from_summaries(summaries, tts_model=TTS_MODEL)

    # sanity check
    missing = [cid for cid in summaries.keys() if cid not in voiceover_paths]
    if missing:
        logger.warning("Missing voiceovers for clusters: %s", missing)

    # 7) Assemble condensed video
    final = assemble_from_summaries(video_path, summaries, voiceover_paths, output_path)

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

