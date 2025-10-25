#!/usr/bin/env python3
"""
Simple summarizer that takes full transcript + user query and returns clean summary.
No vector DB, no clustering - just clean OpenRouter summarization.
"""

import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUMMARIZE_MODEL = "qwen/qwen-2.5-72b-instruct"  # Fast and good model

# Initialize OpenRouter client
_client = None
if OPENROUTER_API_KEY:
    _client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    )
    logger.info("OpenRouter client initialized")
else:
    logger.warning("OpenRouter API key not found")


def generate_simple_summary(transcript_sentences: List[Dict], user_query: str) -> Dict:
    """
    Generate a clean, direct summary from full transcript using OpenRouter.
    
    Args:
        transcript_sentences: List of sentence dicts with 'text', 'start', 'end'
        user_query: User's query (e.g., "merge sort", "bubble sort")
    
    Returns:
        Dict with:
        - summary: Clean summary text
        - confidence: AI confidence level
        - error: Error message if any
    """
    if not transcript_sentences:
        return {"summary": "No transcript available", "confidence": 0, "error": "Empty transcript"}
    
    if not user_query or not user_query.strip():
        return {"summary": "No query provided", "confidence": 0, "error": "Empty query"}
    
    # Build full transcript
    full_transcript = " ".join([sent.get("text", "").strip() for sent in transcript_sentences])
    
    if len(full_transcript) < 50:
        return {"summary": "Transcript too short to analyze", "confidence": 0, "error": "Short transcript"}
    
    logger.info(f"Generating summary for query: '{user_query}' from {len(transcript_sentences)} sentences ({len(full_transcript)} chars)")
    
    if not _client:
        return {"summary": "OpenRouter API not configured", "confidence": 0, "error": "No API key"}
    
    try:
        # Create clean, direct prompt
        prompt = f"""You are a technical content editor. Read the video transcript and create a clean, direct summary about "{user_query}".

INSTRUCTIONS:
1. Read the ENTIRE transcript carefully
2. Find all information related to "{user_query}"
3. Write a DIRECT, factual summary (NO conversational phrases like "Let's dive into" or "The speaker describes")
4. Start directly with the topic: "{user_query} is..." or "{user_query} works by..."
5. Correct any spelling mistakes or transcription errors in the content
6. Include key concepts, steps, and technical details
7. Write in clear, educational language (like a textbook explanation)
8. If "{user_query}" is NOT discussed in the video, say: "This topic is not covered in this video."
9. Length: 4-8 sentences of pure technical content

VIDEO TRANSCRIPT:
{full_transcript}

---

DIRECT SUMMARY (about "{user_query}"):"""

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
            logger.warning(f"Topic '{user_query}' not found in transcript")
            return {"summary": summary, "confidence": 0}
        
        logger.info(f"Generated summary: {len(summary)} chars")
        
        return {
            "summary": summary,
            "confidence": 0.9,  # High confidence for direct AI processing
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OpenRouter summarization failed: {e}")
        
        # Check for rate limit
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return {
                "summary": "Rate limit exceeded. Please try again in a few minutes.",
                "confidence": 0,
                "error": "rate_limit"
            }
        
        return {
            "summary": f"API Error: {error_msg}",
            "confidence": 0,
            "error": error_msg
        }


def process_simple_query(transcript_sentences: List[Dict], user_query: str) -> Dict:
    """
    Simple wrapper that processes a query and returns formatted result.
    
    Args:
        transcript_sentences: Transcript sentences from Whisper
        user_query: User's topic query
    
    Returns:
        Dict with summary, query, timestamps, etc. (compatible with existing frontend)
    """
    result = generate_simple_summary(transcript_sentences, user_query)
    
    # Find approximate timestamps for the topic (simple keyword search)
    timestamps = []
    query_words = user_query.lower().split()
    
    for sent in transcript_sentences:
        text_lower = sent.get("text", "").lower()
        if any(word in text_lower for word in query_words):
            timestamps.append((sent.get("start", 0), sent.get("end", 0)))
    
    # Limit to first few matches for video clips
    timestamps = timestamps[:5]
    
    return {
        "summary": result["summary"],
        "query": user_query,
        "confidence": result["confidence"],
        "error": result.get("error"),
        "timestamps": timestamps,
        "segments": [
            {
                "text": sent["text"],
                "start": sent["start"], 
                "end": sent["end"],
                "score": 0.8  # Default relevance score
            } 
            for sent in transcript_sentences 
            if any(word in sent.get("text", "").lower() for word in user_query.lower().split())
        ][:10]  # Top 10 relevant segments
    }


if __name__ == "__main__":
    # Test with dummy data
    test_sentences = [
        {"text": "Today we'll learn about merge sort algorithm", "start": 0, "end": 3},
        {"text": "Merge sort is a divide and conquer algorithm", "start": 3, "end": 6},
        {"text": "It works by dividing the array into two halves", "start": 6, "end": 9},
    ]
    
    result = process_simple_query(test_sentences, "merge sort")
    print("Test Result:")
    print(f"Summary: {result['summary']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Found {len(result['segments'])} relevant segments")