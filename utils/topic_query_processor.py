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

load_dotenv()

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


def process_topic_query(db, query: str, sentences: List[Dict]) -> Dict:
    """
    Main function to process a topic query and prepare data for video generation.
    NOW USES FULL TRANSCRIPT for better AI understanding.
    
    Args:
        db: VectorDB instance
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
    
    # 1. Search for relevant segments using vector similarity (sorted by score)
    relevant_segments = search_topic_in_transcript(db, query, top_k=15)
    
    if not relevant_segments:
        logger.warning(f"Vector search found no relevant segments for: {query}")
        # Still try with full transcript - AI might find it
        relevant_segments = []
    
    # Keep a copy sorted by relevance for summary generation
    segments_by_relevance = relevant_segments.copy()
    
    # Sort by timestamp for video clip extraction (chronological order)
    relevant_segments_chronological = sorted(relevant_segments, key=lambda x: x["start"])
    
    # 2. Build full transcript text from ALL sentences
    full_transcript = " ".join([sent.get("text", "").strip() for sent in sentences])
    
    logger.info(f"Full transcript: {len(full_transcript)} chars, {len(sentences)} sentences")
    
    # 3. Generate summary from FULL TRANSCRIPT (not just segments!)
    # This gives AI complete context to understand and explain the query
    # Pass segments sorted by relevance (best matches first)
    summary_result = generate_summary_from_full_transcript(
        full_transcript, 
        query, 
        segments_by_relevance  # Use relevance-sorted segments for better fallback
    )
    
    summary = summary_result.get("summary", "")
    confidence = summary_result.get("confidence", 0)
    is_rate_limited = summary_result.get("rate_limited", False)
    is_error = summary_result.get("error", False)
    
    # Check if it's a rate limit or API error (but we have segments)
    if is_rate_limited or is_error:
        if segments_by_relevance:
            logger.warning(f"API issue, but continuing with {len(segments_by_relevance)} segments found")
            # Continue processing with segments even if summary is fallback
        else:
            logger.error(f"API error and no segments found for: {query}")
            return {
                "summary": summary,
                "segments": [],
                "groups": [],
                "query": query
            }
    
    # Check if topic truly not found (confidence 0 and not an API error)
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
