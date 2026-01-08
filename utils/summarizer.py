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
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

# Load .env from project root (parent of utils/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import translation utility
try:
    from utils.translator import translate_text, is_translation_available
except ImportError:
    # Fallback if running as standalone
    try:
        from translator import translate_text, is_translation_available
    except ImportError:
        logger.warning("Translation utility not available")
        translate_text = None
        is_translation_available = lambda: False

# OpenRouter client via openai package as used earlier
from openai import OpenAI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
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

def _classify_cluster_openrouter(cluster_text: str, topic: str) -> Tuple[bool, str]:
    """
    Ask OpenRouter whether this cluster is meaningful (importance).
    Expects a JSON-like answer but handles plain text.
    Returns: (important_bool, reason_text)
    """
    prompt = (
        f"You are an assistant that classifies transcript chunks based on their relevance to a user's query.\n"
        f"Return JSON only with keys: 'important' (true/false) and 'reason' (string).\n\n"
        f"The user is specifically looking for content about: {topic}\n\n"
        f"CRITICAL RULES:\n"
        f"1. A chunk is 'important' (true) ONLY if it EXPLICITLY mentions or discusses '{topic}' by name.\n"
        f"2. A chunk is 'filler' (false) if:\n"
        f"   - It's a greeting, subscribe request, or chit-chat\n"
        f"   - It discusses a DIFFERENT algorithm/topic, even if related or similar\n"
        f"   - It only mentions '{topic}' in passing without explaining it\n"
        f"   - It compares other things to '{topic}' but doesn't explain '{topic}' itself\n\n"
        f"EXAMPLE: If user asks for 'Timsort':\n"
        f"  - 'Timsort was created in 2002...' → IMPORTANT (true)\n"
        f"  - 'Almost exactly like insertion sort...' → FILLER (false) - describes heap sort, not Timsort\n"
        f"  - 'Bubble sort is easy to understand...' → FILLER (false) - different algorithm\n\n"
        f"USER QUERY: {topic}\n\n"
        f"TRANSCRIPT:\n{cluster_text}"
    )
    content = _safe_openrouter_call(CLASSIFY_MODEL, [{"role":"user","content":prompt}])
    if not content:
        # fallback heuristic: check if topic keywords appear in text
        topic_lower = topic.lower()
        text_lower = cluster_text.lower()
        # Simple keyword matching as fallback
        if topic_lower in text_lower:
            return (True, "fallback-keyword-match")
        else:
            return (False, "fallback-no-keyword-match")
    # try parse JSON
    try:
        parsed = json.loads(content)
        important = parsed.get("important", True)
        reason = parsed.get("reason", "") or content
        
        # Additional safety check: if classified as important but topic keyword not in text, be suspicious
        if important:
            topic_keywords = topic.lower().replace(" ", "").replace("-", "")
            text_normalized = cluster_text.lower().replace(" ", "").replace("-", "")
            if topic_keywords not in text_normalized:
                # LLM said important but exact topic name not mentioned - double-check
                logger.warning(f"Cluster classified as important but '{topic}' not found in text. Reason: {reason}")
                # If LLM gave a weak reason, reject it
                if "similar" in reason.lower() or "related" in reason.lower() or "like" in reason.lower():
                    logger.info(f"Rejecting cluster: too vague - {reason}")
                    return (False, f"rejected-vague-match: {reason}")
        
        return (bool(important), reason)
    except Exception:
        lc = content.lower()
        if "false" in lc or "not important" in lc or "filler" in lc or "subscribe" in lc:
            return (False, content)
        return (True, content)

def _summarize_cluster_openrouter(cluster_text: str,topic: str) -> str:
    prompt = (
        f"You are an assistant creating an ULTRA-CONCISE summary for a video voiceover.\n\n"
        f"STRICT RULES:\n"
        f"- Output EXACTLY 1-2 short sentences (MAX 30 words total)\n"
        f"- Focus ONLY on the single most important point about '{topic}'\n"
        f"- Use simple, spoken language (this will be read aloud)\n"
        f"- NO filler words, NO introductions, NO conclusions\n"
        f"- Start directly with the key information\n\n"
        f"USER QUERY: {topic}\n\n"
        f"TRANSCRIPT:\n{cluster_text}\n\n"
        f"CONCISE SUMMARY (1-2 sentences, max 30 words):"
    )
    content = _safe_openrouter_call(SUMMARIZE_MODEL, [{"role":"user","content":prompt}])
    if content:
        # Enforce max length even from LLM response (truncate to ~50 words)
        words = content.split()
        if len(words) > 50:
            content = ' '.join(words[:50])
        return content
    # fallback to local summarizer if available
    if _local_summarizer:
        try:
            out = _local_summarizer(cluster_text, max_length=60, min_length=20, do_sample=False)
            return out[0]["summary_text"]
        except Exception as e:
            logger.warning("Local summarizer failed: %s", e)
    # last fallback: return first 50 words of original text
    words = cluster_text.split()[:50]
    return ' '.join(words)

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
    query: str,
    topic_clusters: Dict[int, List[Dict]],
    scenes: List[Tuple[Optional[float], Optional[float]]],
    video_duration: float,
    require_classification: bool = True,
    use_openrouter: bool = True,
    target_language: Optional[str] = None
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

    # Track classification results for safety fallback
    classification_results: Dict[int, Tuple[bool, str, str, float, float, List]] = {}
    
    # If no query provided, skip classification entirely - keep all clusters
    skip_classification = not query or not query.strip()
    if skip_classification:
        logger.info("No query provided - skipping classification, keeping all clusters")

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
        cls_reason = "no-classification"
        
        if skip_classification:
            # No query = keep everything (full video summarization mode)
            keep = True
            cls_reason = "no-query-keep-all"
        elif require_classification and use_openrouter and USE_OPENROUTER:
            try:
                keep, cls_reason = _classify_cluster_openrouter(cluster_text, query)
            except Exception as e:
                logger.warning("Classification exception for cluster %s: %s", cid, e)
                keep = True
                cls_reason = "exception-fallback-keep"
        elif require_classification:
            # Relaxed heuristic fallback: keep if has any meaningful content
            # (at least 1 sentence with >5 words, or total >15 words)
            word_count = len(cluster_text.split())
            keep = word_count > 15
            cls_reason = f"heuristic-wordcount-{word_count}"

        # Store result for potential fallback
        classification_results[cid] = (keep, cls_reason, cluster_text, cluster_start, cluster_end, sentences)

        if not keep:
            logger.info("Dropping cluster %s as filler: %s", cid, cls_reason)
            continue

        # Merge overlapping scenes -> visual start/end
        merged_start, merged_end = _merge_scenes_for_cluster(cluster_start, cluster_end, scenes, video_duration)
        merged_start = max(0.0, merged_start)
        merged_end = min(video_duration, merged_end)

        # Summarize (prefer OpenRouter)
        if use_openrouter and USE_OPENROUTER:
            summary = _summarize_cluster_openrouter(cluster_text, query)
        else:
            # local fallback
            if _local_summarizer:
                try:
                    out = _local_summarizer(cluster_text, max_length=60, min_length=20, do_sample=False)
                    summary = out[0]["summary_text"]
                except Exception as e:
                    logger.warning("Local summarizer failed: %s", e)
                    summary = ' '.join(cluster_text.split()[:50])
            else:
                summary = cluster_text[:1000]

        results[int(cid)] = {
            "summary": summary.strip(),
            "start": float(merged_start),
            "end": float(merged_end),
            "sentences": sentences
        }
        logger.info("Cluster %s kept: %.2f-%.2f summary len=%d", cid, merged_start, merged_end, len(summary.split()))

    # SAFETY FALLBACK: If all clusters were filtered out, keep the top N longest ones
    if not results and classification_results:
        logger.warning("All clusters were filtered! Applying safety fallback - keeping top clusters by content length")
        # Sort by text length (longest = most content) and keep up to 5
        sorted_by_length = sorted(
            classification_results.items(),
            key=lambda x: len(x[1][2]),  # x[1][2] is cluster_text
            reverse=True
        )
        fallback_count = min(5, len(sorted_by_length))
        
        for cid, (_, cls_reason, cluster_text, cluster_start, cluster_end, sentences) in sorted_by_length[:fallback_count]:
            logger.info("Fallback: keeping cluster %s (was filtered: %s)", cid, cls_reason)
            
            # Merge overlapping scenes -> visual start/end
            merged_start, merged_end = _merge_scenes_for_cluster(cluster_start, cluster_end, scenes, video_duration)
            merged_start = max(0.0, merged_start)
            merged_end = min(video_duration, merged_end)

            # Summarize
            if use_openrouter and USE_OPENROUTER:
                summary = _summarize_cluster_openrouter(cluster_text, query or "general summary")
            else:
                if _local_summarizer:
                    try:
                        out = _local_summarizer(cluster_text, max_length=60, min_length=20, do_sample=False)
                        summary = out[0]["summary_text"]
                    except Exception as e:
                        logger.warning("Local summarizer failed: %s", e)
                        summary = ' '.join(cluster_text.split()[:50])
                else:
                    summary = cluster_text[:1000]

            results[int(cid)] = {
                "summary": summary.strip(),
                "start": float(merged_start),
                "end": float(merged_end),
                "sentences": sentences
            }
            logger.info("Fallback cluster %s: %.2f-%.2f", cid, merged_start, merged_end)

    # Translate all summaries if target language specified
    if target_language and translate_text and is_translation_available():
        logger.info(f"Translating {len(results)} summaries to {target_language}")
        for cid, data in results.items():
            original_summary = data["summary"]
            try:
                translated = translate_text(original_summary, target_language, source_language="en-IN")
                if translated and translated != original_summary:
                    data["summary"] = translated
                    logger.info(f"Cluster {cid} translated successfully")
            except Exception as e:
                logger.warning(f"Translation failed for cluster {cid}: {e}, keeping original")
                # Keep original summary on error

    return results
