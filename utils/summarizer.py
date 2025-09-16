"""
utils/summarizer.py

Functions:
 - summarize_topics(topic_clusters, scenes, video_duration, require_classification, use_openrouter)
   returns { cluster_id: {"summary": str, "start": float, "end": float, "sentences": [...] } }
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# OpenRouter client via openai package as used earlier
from openai import OpenAI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CLASSIFY_MODEL = os.getenv("OPENROUTER_CLASSIFY_MODEL", "nvidia/nemotron-nano-9b-v2:free")
SUMMARIZE_MODEL = os.getenv("OPENROUTER_SUMMARIZE_MODEL", "nvidia/nemotron-nano-9b-v2:free")
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

_client = None
if USE_OPENROUTER:
    _client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)

# local summarizer fallback
_local_summarizer = None
try:
    from transformers import pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    _local_summarizer = pipeline("summarization", model=os.getenv("LOCAL_SUMMARY_MODEL", "facebook/bart-large-cnn"), device=device)
except Exception as e:
    logger.warning("Local summarizer not available: %s", e)
    _local_summarizer = None

# ---- helpers ----
def _safe_openrouter_call(model: str, messages: List[Dict[str, str]], max_retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    """
    Call OpenRouter chat completions and return the text content string (not a list).
    """
    if _client is None:
        return None
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(model=model, messages=messages)
            # resp.choices[0].message.content is a plain string according to openrouter client usage
            content = resp.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            # else try str conversion
            return str(content).strip()
        except Exception as e:
            last_err = e
            logger.warning("OpenRouter call failed attempt %d: %s", attempt+1, e)
            time.sleep(backoff * (2 ** attempt))
    logger.error("OpenRouter failed after retries: %s", last_err)
    return None

def _classify_cluster_openrouter(cluster_text: str) -> Tuple[bool, str]:
    """
    Ask OpenRouter whether this cluster is meaningful (importance).
    Expects a JSON-like answer but handles plain text.
    Returns: (important_bool, reason_text)
    """
    prompt = (
        "You are an assistant that classifies transcript chunks. "
        "Return JSON only with keys: important (true/false) and reason (string). "
        "Important means this chunk is a real technical explanation or an essential part "
        "of a concept that should be kept in a condensed educational video. "
        "Filler includes subscribe requests, greetings, short chit-chat, or unrelated tangents.\n\n"
        "TRANSCRIPT:\n" + cluster_text
    )
    content = _safe_openrouter_call(CLASSIFY_MODEL, [{"role":"user","content":prompt}])
    if not content:
        # fallback heuristic: if cluster length > ~30 words, keep; else maybe drop
        reason = "openrouter-failed-fallback"
        return (len(cluster_text.split()) > 30, reason)
    # try parse JSON
    try:
        parsed = json.loads(content)
        important = parsed.get("important", True)
        reason = parsed.get("reason", "") or content
        return (bool(important), reason)
    except Exception:
        lc = content.lower()
        if "false" in lc or "not important" in lc or "filler" in lc or "subscribe" in lc:
            return (False, content)
        return (True, content)

def _summarize_cluster_openrouter(cluster_text: str) -> str:
    prompt = (
        "Rewrite the following transcript into a concise, clear, and complete educational explanation. "
        "Keep technical terms, steps and examples. Make it substantially shorter than the original but "
        "preserve the meaning and the example (i.e., a viewer should understand and be able to apply the concept). "
        "Output 3-6 short sentences suitable for a 2-3 minute spoken voiceover.\n\n"
        "TRANSCRIPT:\n" + cluster_text
    )
    content = _safe_openrouter_call(SUMMARIZE_MODEL, [{"role":"user","content":prompt}])
    if content:
        return content
    # fallback to local summarizer if available
    if _local_summarizer:
        try:
            out = _local_summarizer(cluster_text, max_length=150, min_length=40, do_sample=False)
            return out[0]["summary_text"]
        except Exception as e:
            logger.warning("Local summarizer failed: %s", e)
    # last fallback: return original chunk (trimmed)
    return (cluster_text[:1000] + "...") if len(cluster_text) > 1000 else cluster_text

def _merge_scenes_for_cluster(cluster_start: float, cluster_end: float, scenes: List[Tuple[Optional[float], Optional[float]]], video_duration: float) -> Tuple[float, float]:
    """
    Scenes may include (start, None) — normalize None to video_duration here.
    Return the merged (start,end) covering all scenes that overlap cluster range.
    If no overlapping scene found, default to cluster_start/cluster_end.
    """
    # normalize scenes and collect overlapping
    overlaps = []
    for s0, s1 in scenes:
        s1n = video_duration if s1 is None else s1
        if (s0 < cluster_end) and (s1n > cluster_start):
            overlaps.append((s0, s1n))
    if not overlaps:
        # no overlap -> clamp to cluster range but ensure within video bounds
        cs = max(0.0, cluster_start)
        ce = min(video_duration, cluster_end if cluster_end is not None else video_duration)
        return cs, ce
    starts = [s[0] for s in overlaps]
    ends = [s[1] for s in overlaps]
    return min(starts), max(ends)

# ---- main summarizer function ----
def summarize_topics(
    topic_clusters: Dict[int, List[Dict]],
    scenes: List[Tuple[Optional[float], Optional[float]]],
    video_duration: float,
    require_classification: bool = True,
    use_openrouter: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Summarize clusters and return:
      { cluster_id: { "summary": str, "start": float, "end": float, "sentences": [ ... ] } }
    - topic_clusters: {id: [ {text,start,end,...}, ... ] }
    - scenes: list of (start,end) from scene detector (end may be None)
    - video_duration: needed to normalize None ends
    """
    results: Dict[int, Dict[str, Any]] = {}
    if not topic_clusters:
        return results

    for cid, sentences in topic_clusters.items():
        # ensure list of dicts
        if not sentences or not isinstance(sentences, list):
            logger.debug("cluster %s empty or bad shape, skipping", cid)
            continue

        # Combine cluster text (keep original sentence order)
        cluster_text = " ".join([str(s.get("text","")).strip() for s in sentences]).strip()
        if not cluster_text:
            continue

        cluster_start = min(float(s.get("start", 0.0)) for s in sentences)
        cluster_end = max(float(s.get("end", cluster_start)) for s in sentences)

        # Classification step
        keep = True
        cls_reason = ""
        if require_classification and use_openrouter and USE_OPENROUTER:
            try:
                keep, cls_reason = _classify_cluster_openrouter(cluster_text)
            except Exception as e:
                logger.warning("Classification exception for cluster %s: %s", cid, e)
                keep = True
        elif require_classification and not use_openrouter:
            # heuristic: require at least 2 sentences or > 30 words
            keep = (len(sentences) >= 2) or (len(cluster_text.split()) > 30)

        if not keep:
            logger.info("Dropping cluster %s as filler: %s", cid, cls_reason)
            continue

        # Merge overlapping scenes -> visual start/end
        merged_start, merged_end = _merge_scenes_for_cluster(cluster_start, cluster_end, scenes, video_duration)
        merged_start = max(0.0, merged_start)
        merged_end = min(video_duration, merged_end)

        # Summarize (prefer OpenRouter)
        if use_openrouter and USE_OPENROUTER:
            summary = _summarize_cluster_openrouter(cluster_text)
        else:
            # local fallback
            if _local_summarizer:
                try:
                    out = _local_summarizer(cluster_text, max_length=150, min_length=40, do_sample=False)
                    summary = out[0]["summary_text"]
                except Exception as e:
                    logger.warning("Local summarizer failed: %s", e)
                    summary = cluster_text[:1000]
            else:
                summary = cluster_text[:1000]

        results[int(cid)] = {
            "summary": summary.strip(),
            "start": float(merged_start),
            "end": float(merged_end),
            "sentences": sentences
        }
        logger.info("Cluster %s kept: %.2f-%.2f summary len=%d", cid, merged_start, merged_end, len(summary.split()))

    return results
