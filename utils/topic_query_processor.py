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

# Force reload environment variables to get latest API key
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# OpenRouter configuration - reload API key each time
from openai import OpenAI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SUMMARIZE_MODEL = os.getenv("OPENROUTER_SUMMARIZE_MODEL", "qwen/qwen-2.5-72b-instruct")  # More reliable model

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


def refresh_openrouter_client():
    """
    Refresh the OpenRouter client with the latest API key from environment.
    Call this if you've updated the .env file and need to reload the API key.
    """
    global _client, OPENROUTER_API_KEY, SUMMARIZE_MODEL
    
    # Reload environment variables
    load_dotenv(override=True)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    SUMMARIZE_MODEL = os.getenv("OPENROUTER_SUMMARIZE_MODEL", "qwen/qwen3-vl-32b-instruct")
    
    if OPENROUTER_API_KEY:
        _client = OpenAI(
            base_url=OPENROUTER_BASE, 
            api_key=OPENROUTER_API_KEY,
            timeout=30.0,
            max_retries=2
        )
        logger.info(f"OpenRouter client refreshed with new API key (starts with: {OPENROUTER_API_KEY[:15]}...)")
        return True
    else:
        _client = None
        logger.warning("No OpenRouter API key found after refresh")
        return False


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
                keyword_boost += 0.15  # Increase boost for keyword matches
        
        # Update score with boost
        hit["score"] = min(1.0, score + keyword_boost)
        
        # IMPROVED FILTERING: Only keep segments that are actually relevant
        # 1. High vector similarity score (>0.3)
        # 2. OR contains query keywords
        # 3. AND text is substantial (>20 characters)
        text_length_ok = len(hit.get("text", "")) > 20
        has_keywords = any(keyword in text_lower for keyword in query_keywords)
        high_similarity = hit["score"] > 0.3
        
        if text_length_ok and (high_similarity or has_keywords):
            filtered_hits.append(hit)
    
    # Sort by score (highest first) and take only the most relevant
    filtered_hits = sorted(filtered_hits, key=lambda x: x["score"], reverse=True)
    
    # IMPROVED: Take only top 5-8 most relevant segments instead of all
    top_segments = filtered_hits[:8]
    
    logger.info(f"Found {len(top_segments)} relevant segments (from {len(hits)} total hits)")
    if top_segments:
        logger.info(f"Top segment score: {top_segments[0]['score']:.2f}, Bottom segment score: {top_segments[-1]['score']:.2f}")
    
    return top_segments


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
    Generate a comprehensive summary from FULL transcript using OpenRouter.
    Also identifies which parts of the transcript are most relevant.
    
    Args:
        full_transcript: Complete video transcript text
        query: Original user query (e.g., "quick sort")
        relevant_segments: Optional pre-identified relevant segments
    
    Returns:
        Dictionary with:
        - summary: AI-generated summary text
        - timestamps: List of (start, end) tuples where topic is discussed
        - confidence: How confident the AI is about the topic presence
    """
    if not full_transcript:
        logger.warning("No transcript to summarize")
        return {"summary": "", "timestamps": [], "confidence": 0}
    
    logger.info(f"Generating summary for full transcript ({len(full_transcript)} chars) about: {query}")
    
    # Use OpenRouter if available
    if _client and OPENROUTER_API_KEY:
        try:
            # Smart prompt that asks AI to both summarize AND identify timestamps
            prompt = f"""You are analyzing a video transcript to create a focused summary about "{query}".

TASK:
1. Read the ENTIRE transcript carefully
2. Identify ALL parts where "{query}" is explained or discussed
3. Create a clear, comprehensive summary explaining "{query}" based on what's in the video
4. The summary should be suitable for a voiceover narration (natural spoken language)

TRANSCRIPT:
{full_transcript}

---

Please respond with ONLY the summary text that explains "{query}" based on the video content.

Requirements:
- Focus ONLY on "{query}" - ignore unrelated content
- Explain the concept clearly and comprehensively
- Include key steps, examples, and important details mentioned in the video
- Use simple, natural language (like you're speaking to someone)
- Length: 4-8 sentences (60-90 seconds when spoken)
- If "{query}" is not discussed in the video, say "This topic was not covered in the video."

Summary:"""

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
                return {"summary": summary, "timestamps": [], "confidence": 0}
            
            logger.info(f"Generated AI summary: {len(summary)} chars")
            
            # Extract timestamps from relevant segments if provided
            timestamps = []
            if relevant_segments:
                timestamps = [(seg["start"], seg["end"]) for seg in relevant_segments]
            
            return {
                "summary": summary,
                "timestamps": timestamps,
                "confidence": 0.8 if timestamps else 0.5
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenRouter summarization failed: {e}")
            
            # Check error type
            is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower()
            is_connection_error = "connection" in error_msg.lower() or "timeout" in error_msg.lower()
            
            # Handle rate limit or connection errors
            if is_rate_limit or is_connection_error:
                error_type = "Rate limit" if is_rate_limit else "Connection error"
                logger.warning(f"{error_type} - using improved fallback summarization")
                
                if relevant_segments:
                    # Filter segments by relevance score (only high-scoring ones)
                    high_quality_segments = [
                        seg for seg in relevant_segments 
                        if seg.get("score", 0) > 0.6  # Only segments with >60% relevance
                    ]
                    
                    if not high_quality_segments:
                        high_quality_segments = relevant_segments[:5]  # Take top 5 if none above threshold
                    
                    logger.info(f"Using {len(high_quality_segments)} high-quality segments for fallback")
                    
                    # Build focused summary from high-quality segments only
                    combined_text = " ".join([seg.get("text", "").strip() for seg in high_quality_segments])
                    
                    # Create a more focused summary
                    query_lower = query.lower()
                    
                    # Add context message
                    summary_parts = [
                        f"[Note: AI summary limited due to API usage. Showing transcript excerpt about '{query}']",
                        "",
                        combined_text[:700] + "..." if len(combined_text) > 700 else combined_text
                    ]
                    
                    summary = "\n".join(summary_parts)
                    timestamps = [(seg["start"], seg["end"]) for seg in relevant_segments]
                    
                    return {
                        "summary": summary, 
                        "timestamps": timestamps, 
                        "confidence": 0.5,
                        "rate_limited": True
                    }
            
            # Other errors - return error but with timestamps if we have them
            timestamps = []
            if relevant_segments:
                timestamps = [(seg["start"], seg["end"]) for seg in relevant_segments]
            
            return {
                "summary": f"API Error: Rate limit exceeded. Please add credits at https://openrouter.ai or wait for reset. Found {len(timestamps)} relevant segments in video.", 
                "timestamps": timestamps, 
                "confidence": 0 if not timestamps else 0.3,
                "error": True
            }
    
    # Fallback: return relevant segments text
    logger.warning("OpenRouter not available - using fallback")
    if relevant_segments:
        combined_text = " ".join([seg.get("text", "").strip() for seg in relevant_segments])
        max_length = 500
        summary = combined_text[:max_length] + "..." if len(combined_text) > max_length else combined_text
        timestamps = [(seg["start"], seg["end"]) for seg in relevant_segments]
        return {"summary": summary, "timestamps": timestamps, "confidence": 0.3}
    
    return {"summary": "Could not generate summary", "timestamps": [], "confidence": 0}


def process_topic_query_enhanced(query: str, transcript: List[Dict], srt_content: str, db_path: str) -> Dict:
    """
    ENHANCED APPROACH: Build vector database and use focused search.
    
    Args:
        query: User query (e.g., "merge sort", "bubble sort")  
        transcript: List of transcript segments with timestamps
        srt_content: SRT format content for building database
        db_path: Path to save/load vector database
    
    Returns:
        Dictionary with:
        - summary: AI-generated summary from relevant segments only
        - segments: List of relevant segments with timestamps
        - confidence: AI confidence level
        - query: Original query
    """
    from .database import VectorDB, parse_srt_content
    
    logger.info(f"🎯 Enhanced topic query processing: '{query}'")
    
    # 1. Build/load vector database with SRT content
    logger.info("Building vector database from SRT content...")
    try:
        db = VectorDB(db_path=db_path)
        
        # Parse SRT content to get segments for database
        srt_segments = parse_srt_content(srt_content)
        if not srt_segments:
            logger.warning("No segments parsed from SRT, using transcript segments")
            srt_segments = [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in transcript]
        
        # Build database
        db.build(srt_segments)
        logger.info(f"✅ Vector database built with {len(srt_segments)} segments")
        
    except Exception as e:
        logger.error(f"Vector database error: {e}")
        return {
            "summary": f"Error building search database: {e}",
            "segments": [],
            "confidence": 0,
            "query": query
        }
    
    # 2. Search for relevant segments
    logger.info(f"🔍 Searching for segments related to '{query}'...")
    relevant_segments = search_topic_in_transcript(db, query, top_k=15)
    
    if not relevant_segments:
        logger.warning(f"No relevant segments found for: {query}")
        return {
            "summary": f"The topic '{query}' was not found in this video.",
            "segments": [],
            "confidence": 0,
            "query": query
        }
    
    # 3. Generate focused summary
    focused_transcript = " ".join([seg.get("text", "").strip() for seg in relevant_segments])
    logger.info(f"📝 Focused transcript: {len(focused_transcript)} chars from {len(relevant_segments)} segments")
    
    # 4. Generate AI summary from relevant segments only
    summary_result = generate_focused_summary_from_segments(
        focused_transcript, 
        query, 
        relevant_segments
    )
    
    summary = summary_result.get("summary", "")
    confidence = summary_result.get("confidence", 0)
    is_error = summary_result.get("error", False)
    
    if is_error:
        logger.error(f"Summary generation failed for query: {query}")
        # Fallback to concatenated segment text  
        summary = focused_transcript[:500] + "..." if len(focused_transcript) > 500 else focused_transcript
        confidence = 30
    
    # Filter segments to ensure good confidence
    if confidence < 50:
        logger.info(f"Low confidence ({confidence}%), filtering segments...")
        # Keep only highly relevant segments
        filtered_segments = []
        query_words = set(query.lower().split())
        for seg in relevant_segments:
            seg_words = set(seg.get("text", "").lower().split())
            if query_words.intersection(seg_words) or len(seg_words.intersection(query_words)) > 0:
                filtered_segments.append(seg)
        
        if filtered_segments:
            relevant_segments = filtered_segments[:8]  # Limit to top 8
            confidence = min(85, confidence + 20)  # Boost confidence slightly
    
    logger.info(f"✅ Enhanced approach completed: confidence={confidence}%, segments={len(relevant_segments)}")
    
    return {
        "summary": summary,
        "segments": relevant_segments,
        "confidence": confidence,
        "query": query
    }


# Alias for compatibility
def process_topic_query(query: str = None, transcript: List[Dict] = None, srt_content: str = None, db_path: str = None, db=None, sentences: List[Dict] = None) -> Dict:
    """
    Compatibility wrapper that handles both old and new signatures.
    
    New signature: process_topic_query(query, transcript, srt_content, db_path)
    Old signature: process_topic_query(db, query, sentences)
    """
    # New enhanced approach
    if transcript is not None and srt_content is not None and db_path is not None:
        return process_topic_query_enhanced(query, transcript, srt_content, db_path)
    
    # Old approach for backward compatibility
    elif db is not None and sentences is not None:
        return process_topic_query_old(db, query, sentences)
    
    else:
        raise ValueError("Invalid arguments. Use either: process_topic_query(query, transcript, srt_content, db_path) or process_topic_query(db, query, sentences)")


def process_topic_query_old(db, query: str, sentences: List[Dict]) -> Dict:
    """
    NEW IMPROVED APPROACH: Use vector search to find relevant segments,
    then send ONLY those segments to OpenRouter for focused summarization.
    
    Args:
        db: VectorDB instance with transcript embeddings
        query: User query (e.g., "merge sort", "bubble sort")
        sentences: Full transcript sentences with timestamps
    
    Returns:
        Dictionary with:
        - summary: AI-generated summary from relevant segments only
        - segments: List of relevant segments with timestamps
        - groups: Merged segment groups for video clips
        - query: Original query
        - confidence: AI confidence level
    """
    logger.info(f"Processing topic query: '{query}'")
    
    # 1. Use vector search to find the most relevant segments for the topic
    logger.info(f"Searching for segments related to '{query}'...")
    relevant_segments = search_topic_in_transcript(db, query, top_k=15)
    
    if not relevant_segments:
        logger.warning(f"No relevant segments found for: {query}")
        return {
            "summary": f"The topic '{query}' was not found in this video.",
            "segments": [],
            "groups": [],
            "query": query,
            "confidence": 0
        }
    
    # 2. Extract text from relevant segments to create focused transcript
    focused_transcript = " ".join([seg.get("text", "").strip() for seg in relevant_segments])
    logger.info(f"Focused transcript: {len(focused_transcript)} chars from {len(relevant_segments)} segments")
    
    # 3. Generate summary using ONLY the relevant segments (not full transcript)
    summary_result = generate_focused_summary_from_segments(
        focused_transcript, 
        query, 
        relevant_segments
    )
    
    summary = summary_result.get("summary", "")
    confidence = summary_result.get("confidence", 0)
    is_error = summary_result.get("error", False)
    
    if is_error:
        logger.error(f"Summary generation failed for query: {query}")
        # Fallback to segment text
        if relevant_segments:
            fallback_text = " ".join([seg.get("text", "")[:100] for seg in relevant_segments[:3]])
            summary = f"Found relevant content about '{query}': {fallback_text}..."
            confidence = 0.3
        else:
            summary = f"Unable to generate summary for '{query}'"
            confidence = 0
    
    # 4. Sort segments by timestamp for video clip extraction
    relevant_segments_chronological = sorted(relevant_segments, key=lambda x: x["start"])
    
    # 5. Merge adjacent segments for better video clips
    segment_groups = merge_adjacent_segments(relevant_segments_chronological, gap_threshold=4.0)
    
    if not segment_groups:
        logger.warning("No segment groups - creating single group from all segments")
        segment_groups = [relevant_segments_chronological] if relevant_segments_chronological else []
    
    logger.info(f"Topic query processed: Summary={len(summary)} chars, {len(segment_groups)} clip groups, confidence={confidence}")
    
    return {
        "summary": summary,
        "segments": relevant_segments_chronological,
        "groups": segment_groups,
        "query": query,
        "confidence": confidence,
        "total_segments": len(relevant_segments_chronological),
        "total_groups": len(segment_groups),
        "focused_approach": True  # Flag to indicate this used focused vector search
    }


def generate_focused_summary_from_segments(focused_transcript: str, query: str, relevant_segments: List[Dict]) -> Dict:
    """
    Generate summary using OpenRouter from ONLY the relevant segments (not full transcript).
    This creates much more focused and accurate summaries.
    
    Args:
        focused_transcript: Text from only the relevant segments
        query: User's query (e.g., "merge sort")
        relevant_segments: List of relevant segment dicts with scores
    
    Returns:
        Dictionary with summary, confidence, error status
    """
    if not focused_transcript:
        logger.warning("No focused transcript to summarize")
        return {"summary": "No relevant content found", "confidence": 0, "error": True}
    
    logger.info(f"Generating focused summary from {len(relevant_segments)} relevant segments for query: '{query}'")
    
    if not _client or not OPENROUTER_API_KEY:
        logger.error("OpenRouter not available")
        return {"summary": "OpenRouter API not configured", "confidence": 0, "error": True}
    
    try:
        # Calculate average relevance score for confidence
        avg_score = sum(seg.get("score", 0) for seg in relevant_segments) / len(relevant_segments)
        
        # Create focused prompt for relevant segments only
        prompt = f"""You are a technical content editor. You have been given transcript segments from a video that are specifically relevant to "{query}". Create a clear, direct summary.

INSTRUCTIONS:
1. The transcript segments below are PRE-FILTERED to be relevant to "{query}"
2. Write a DIRECT, factual summary (NO conversational phrases like "Let's dive into" or "The speaker describes")
3. Start directly with the topic: "{query} is..." or "{query} works by..."
4. Correct any spelling mistakes or transcription errors in the content
5. Include key concepts, steps, and technical details from the segments
6. Write in clear, educational language (like a textbook explanation)
7. Length: 4-8 sentences of pure technical content
8. Focus ONLY on explaining "{query}" based on these relevant segments

RELEVANT TRANSCRIPT SEGMENTS (about "{query}"):
{focused_transcript}

---

DIRECT SUMMARY (about "{query}"):"""

        response = _client.chat.completions.create(
            model=SUMMARIZE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Calculate confidence based on segment relevance and AI response quality
        confidence = min(0.9, 0.5 + (avg_score * 0.4))  # Base 50% + up to 40% from relevance
        
        if len(summary) < 50:
            confidence *= 0.7  # Reduce confidence for very short summaries
        
        logger.info(f"Generated focused summary: {len(summary)} chars, confidence: {confidence:.2f}")
        
        return {
            "summary": summary,
            "confidence": confidence,
            "error": False
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OpenRouter focused summarization failed: {e}")
        
        # Check for rate limit
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return {
                "summary": f"Rate limit exceeded. Please try again in a few minutes.", 
                "confidence": 0,
                "error": True,
                "rate_limited": True
            }
        
        return {
            "summary": f"API Error: {error_msg}", 
            "confidence": 0,
            "error": True
        }


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
