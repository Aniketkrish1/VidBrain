"""
utils/topic_query_processor.py

Process user queries about specific topics in the video:
1. Search transcription using VectorDB
2. Extract relevant segments about the topic
3. Generate comprehensive summary using OpenRouter
4. Return structured data for video generation
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Force reload .env file to override any system environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# OpenRouter configuration
from openai import OpenAI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SUMMARIZE_MODEL = os.getenv("OPENROUTER_SUMMARIZE_MODEL", "qwen/qwen3-vl-32b-instruct")

# Validate API key
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in .env file - OpenRouter features will not be available")
else:
    logger.info(f"OpenRouter API key loaded (starts with: {OPENROUTER_API_KEY[:15]}...)")
    logger.info(f"Using model: {SUMMARIZE_MODEL}")

_client = None
if OPENROUTER_API_KEY:
    _client = OpenAI(
        base_url=OPENROUTER_BASE, 
        api_key=OPENROUTER_API_KEY,
        timeout=30.0,  # 30 second timeout
        max_retries=2   # Only retry twice
    )


def search_topic_in_transcript(db, query: str, top_k: int = 20) -> List[Dict]:
    """
    Search for topic-related segments in the transcript.
    
    Args:
        db: VectorDB instance with transcript embeddings
        query: User query (e.g., "quick sort", "recursion", "binary search")
        top_k: Number of top results to retrieve
    
    Returns:
        List of segments with text, start, end, score (sorted by relevance)
    """
    logger.info(f"Searching transcript for topic: '{query}'")
    
    # Get vector search results
    hits = db.search(query, top_k=top_k)
    
    if not hits:
        logger.warning(f"No results found for query: {query}")
        return []
    
    # Filter and enhance results
    query_keywords = set(query.lower().split())
    filtered_hits = []
    
    for hit in hits:
        text_lower = hit.get("text", "").lower()
        score = hit.get("score", 0)
        
        # Boost score if query keywords are present
        keyword_boost = 0.0
        for keyword in query_keywords:
            if keyword in text_lower:
                keyword_boost += 0.1
        
        # Update score with boost
        hit["score"] = min(1.0, score + keyword_boost)
        
        # Keep high-relevance segments or those containing query keywords
        if hit["score"] > 0.25 or any(keyword in text_lower for keyword in query_keywords):
            filtered_hits.append(hit)
    
    # Sort by score (highest first) to get best matches at top
    filtered_hits = sorted(filtered_hits, key=lambda x: x["score"], reverse=True)
    
    logger.info(f"Found {len(filtered_hits)} relevant segments (from {len(hits)} total hits)")
    logger.info(f"Top segment score: {filtered_hits[0]['score']:.2f}, Bottom segment score: {filtered_hits[-1]['score']:.2f}")
    
    return filtered_hits


def merge_adjacent_segments(segments: List[Dict], gap_threshold: float = 3.0) -> List[List[Dict]]:
    """
    Group segments that are close together in time.
    
    Args:
        segments: List of segment dicts with start, end, text
        gap_threshold: Maximum gap in seconds to consider segments adjacent
    
    Returns:
        List of segment groups
    """
    if not segments:
        return []
    
    groups = []
    current_group = [segments[0]]
    
    for seg in segments[1:]:
        prev_seg = current_group[-1]
        
        # Convert timestamps to float if they're strings
        prev_end = float(prev_seg["end"]) if not isinstance(prev_seg["end"], (int, float)) else prev_seg["end"]
        curr_start = float(seg["start"]) if not isinstance(seg["start"], (int, float)) else seg["start"]
        
        # Check if segments are close enough to merge
        if curr_start <= prev_end + gap_threshold:
            current_group.append(seg)
        else:
            groups.append(current_group)
            current_group = [seg]
    
    if current_group:
        groups.append(current_group)
    
    logger.info(f"Merged {len(segments)} segments into {len(groups)} groups")
    return groups


def generate_summary_from_full_transcript(full_transcript: str, query: str, 
                                         relevant_segments: List[Dict] = None) -> Dict:
    """
    Generate summary using AI to analyze the full transcript for a specific query.
    This is the CLEAN approach - let AI read everything and extract what's relevant.
    
    Args:
        full_transcript: Complete video transcript text
        query: User's query (e.g., "quick sort", "explain recursion")
        relevant_segments: IGNORED - only kept for compatibility
    
    Returns:
        Dictionary with:
        - summary: AI-generated summary text (in English)
        - confidence: How confident the AI is about the topic
        - error: True if API error occurred
        - rate_limited: True if rate limit hit
    """
    if not full_transcript:
        logger.warning("No transcript to summarize")
        return {"summary": "No transcript available", "confidence": 0, "error": True}
    
    logger.info(f"Generating summary from full transcript ({len(full_transcript)} chars) for query: '{query}'")
    
    # Use OpenRouter if available
    if _client and OPENROUTER_API_KEY:
        try:
            # Clean, direct prompt for crisp summaries
            prompt = f"""You are a technical content editor. Read the video transcript and create a clean, direct summary about "{query}".

INSTRUCTIONS:
1. Read the ENTIRE transcript carefully
2. Find all information related to "{query}"
3. Write a DIRECT, factual summary (NO conversational phrases like "Let's dive into" or "The speaker describes")
4. Start directly with the topic: "{query} is..." or "{query} works by..."
5. Correct any spelling mistakes or transcription errors in the content
6. Include key concepts, steps, and technical details
7. Write in clear, educational language (like a textbook explanation)
8. If "{query}" is NOT discussed in the video, say: "This topic is not covered in this video."
9. Length: 4-8 sentences of pure technical content

VIDEO TRANSCRIPT:
{full_transcript}

---

DIRECT SUMMARY (about "{query}"):"""

            response = _client.chat.completions.create(
                model=SUMMARIZE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Check if topic was found
            not_found_phrases = [
                "not covered", "not discussed", "not mentioned", 
                "does not contain", "doesn't contain", "no information"
            ]
            
            if any(phrase in summary.lower() for phrase in not_found_phrases):
                logger.warning(f"AI indicates '{query}' not found in transcript")
                return {"summary": summary, "confidence": 0}
            
            logger.info(f"Generated AI summary: {len(summary)} chars")
            
            return {
                "summary": summary,
                "confidence": 0.8  # High confidence when AI successfully processes
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenRouter summarization failed: {e}")
            
            # Check error type
            is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower()
            
            if is_rate_limit:
                return {
                    "summary": f"Rate limit exceeded for OpenRouter API. Please add credits at https://openrouter.ai or wait for reset.", 
                    "confidence": 0,
                    "rate_limited": True
                }
            
            return {
                "summary": f"API Error: {error_msg}", 
                "confidence": 0,
                "error": True
            }
    
    # No API available
    logger.warning("OpenRouter not available")
    return {"summary": "OpenRouter API not configured. Cannot generate AI summary.", "confidence": 0}


def process_topic_query(db, query: str, sentences: List[Dict]) -> Dict:
    """
    SIMPLE approach: Just pass full transcript + query to AI.
    Let AI do ALL the work - find relevant parts AND generate summary.
    
    Args:
        db: VectorDB instance (used only for fallback clip extraction)
        query: User query (e.g., "explain quick sort")
        sentences: Full transcript sentences with timestamps
    
    Returns:
        Dictionary with:
        - summary: AI-generated summary from full transcript
        - segments: List of relevant segments with timestamps
        - groups: Merged segment groups
        - query: Original query
    """
    logger.info(f"Processing topic query: '{query}'")
    
    # 1. Build full transcript text from ALL sentences
    full_transcript = " ".join([sent.get("text", "").strip() for sent in sentences])
    logger.info(f"Full transcript: {len(full_transcript)} chars, {len(sentences)} sentences")
    
    # 2. Let AI analyze EVERYTHING and generate summary
    logger.info(f"Sending full transcript to AI for analysis...")
    summary_result = generate_summary_from_full_transcript(
        full_transcript, 
        query, 
        relevant_segments=None  # No pre-filtering, let AI decide
    )
    
    summary = summary_result.get("summary", "")
    confidence = summary_result.get("confidence", 0)
    is_rate_limited = summary_result.get("rate_limited", False)
    is_error = summary_result.get("error", False)
    
    # 3. For video clips: Use vector search as fallback to find approximate timestamps
    # (We need some way to extract relevant video clips)
    logger.info(f"Finding relevant segments for video clips...")
    relevant_segments = search_topic_in_transcript(db, query, top_k=15)
    
    if not relevant_segments:
        logger.warning(f"No relevant segments found for clips - using first 5 minutes")
        # Fallback: use first few minutes of video
        relevant_segments = []
        for i, sent in enumerate(sentences[:50]):  # First ~5 minutes
            relevant_segments.append({
                "text": sent.get("text", ""),
                "start": sent.get("start", i * 6),
                "end": sent.get("end", (i + 1) * 6),
                "score": 0.5
            })
    
    # Sort by timestamp for video clip extraction
    relevant_segments_chronological = sorted(relevant_segments, key=lambda x: x["start"])
    
    # 4. Merge adjacent segments for video clips
    segment_groups = merge_adjacent_segments(relevant_segments_chronological, gap_threshold=4.0)
    
    if not segment_groups:
        logger.warning("No segment groups - creating single group from all segments")
        segment_groups = [relevant_segments_chronological] if relevant_segments_chronological else []
    
    logger.info(f"Topic query processed: Summary={len(summary)} chars, {len(segment_groups)} clip groups")
    
    return {
        "summary": summary,
        "segments": relevant_segments_chronological,
        "groups": segment_groups,
        "query": query,
        "confidence": confidence,
        "ai_generated": True  # Flag that this is AI-generated, not vector-based
    }
    if confidence == 0 and not is_rate_limited and not is_error:
        if "not covered" in summary.lower():
            logger.error(f"Topic '{query}' not found in video")
            return {
                "summary": summary,
                "segments": [],
                "groups": [],
                "query": query
            }
    
    # 4. Merge adjacent segments for video clip extraction (use chronological order)
    segment_groups = merge_adjacent_segments(relevant_segments_chronological, gap_threshold=4.0)
    
    if not segment_groups:
        logger.warning("No segment groups found - using all relevant segments")
        segment_groups = [[seg] for seg in relevant_segments_chronological] if relevant_segments_chronological else []
    
    # 5. Prepare result
    result = {
        "summary": summary,
        "segments": relevant_segments_chronological,  # Return in chronological order
        "groups": segment_groups,
        "query": query,
        "total_segments": len(relevant_segments_chronological),
        "total_groups": len(segment_groups),
        "confidence": confidence
    }
    
    logger.info(f"Topic query processed successfully: {result['total_segments']} segments, {result['total_groups']} groups")
    return result


def extract_timestamps_from_groups(groups: List[List[Dict]]) -> List[Tuple[float, float]]:
    """
    Extract start and end timestamps from segment groups.
    
    Args:
        groups: List of segment groups
    
    Returns:
        List of (start, end) tuples
    """
    timestamps = []
    
    for group in groups:
        if not group:
            continue
        
        start = float(group[0]["start"]) if not isinstance(group[0]["start"], (int, float)) else group[0]["start"]
        end = float(group[-1]["end"]) if not isinstance(group[-1]["end"], (int, float)) else group[-1]["end"]
        
        timestamps.append((start, end))
    
    return timestamps


if __name__ == "__main__":
    # Test the topic query processor
    from database import VectorDB
    
    # Load or build database
    db = VectorDB()
    
    # Test query
    test_query = "quick sort algorithm"
    
    # Mock sentences for testing
    mock_sentences = [
        {"text": "Today we'll learn about quick sort", "start": 10.0, "end": 12.0},
        {"text": "Quick sort is a divide and conquer algorithm", "start": 12.5, "end": 15.0},
    ]
    
    result = process_topic_query(db, test_query, mock_sentences)
    
    print(f"\nQuery: {result['query']}")
    print(f"Summary: {result['summary']}")
    print(f"Found {result['total_segments']} segments in {result['total_groups']} groups")
