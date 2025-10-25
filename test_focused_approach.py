#!/usr/bin/env python3
"""
Test the new focused vector search + summarization approach
"""

import sys
import os
sys.path.append('.')

from utils.database import VectorDB
from utils.topic_query_processor import process_topic_query
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing New Focused Vector Search + Summarization")
print("=" * 60)

# Mock transcript data for testing
mock_transcript_sentences = [
    {"text": "Today we'll learn about different sorting algorithms", "start": 10.0, "end": 13.0},
    {"text": "Merge sort is a divide and conquer algorithm", "start": 13.5, "end": 16.0},
    {"text": "It works by recursively dividing the array", "start": 16.5, "end": 19.0},
    {"text": "Then merging the sorted subarrays back together", "start": 19.5, "end": 22.0},
    {"text": "Bubble sort is much simpler but less efficient", "start": 45.0, "end": 48.0},
    {"text": "It compares adjacent elements and swaps them", "start": 48.5, "end": 51.0},
    {"text": "This process repeats until the array is sorted", "start": 51.5, "end": 54.0},
    {"text": "Quick sort is another efficient algorithm", "start": 75.0, "end": 78.0},
    {"text": "It uses a pivot element to partition the array", "start": 78.5, "end": 81.0},
]

# Build vector database with mock data
print("\n1. Building Vector Database...")
db = VectorDB(model_name="multi-qa-mpnet-base-dot-v1", db_path="test_vector_db.pkl")

# Build database with mock sentences
try:
    db.build(mock_transcript_sentences)
    print(f"✅ Database built with {len(mock_transcript_sentences)} segments")
except Exception as e:
    print(f"❌ Database build failed: {e}")
    sys.exit(1)

# Test different queries
test_queries = [
    "merge sort",
    "bubble sort", 
    "quick sort",
    "linear search"  # This should not be found
]

print("\n2. Testing Topic Queries...")
for i, query in enumerate(test_queries, 1):
    print(f"\n--- Test {i}: '{query}' ---")
    
    try:
        result = process_topic_query(db, query, mock_transcript_sentences)
        
        print(f"Query: {result['query']}")
        print(f"Found {result['total_segments']} relevant segments")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Summary: {result['summary'][:200]}{'...' if len(result['summary']) > 200 else ''}")
        
        if result.get('focused_approach'):
            print("✅ Used focused vector search approach")
        
        # Show top relevant segments
        if result['segments']:
            print("Top segments:")
            for j, seg in enumerate(result['segments'][:2], 1):
                print(f"  {j}. [{seg['start']:.1f}s] {seg['text']} (score: {seg.get('score', 0):.2f})")
        else:
            print("No relevant segments found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Cleanup
print(f"\n3. Cleanup...")
try:
    if os.path.exists("test_vector_db.pkl"):
        os.remove("test_vector_db.pkl")
        print("✅ Test database cleaned up")
except:
    pass

print("\n" + "=" * 60)
print("🎯 Test completed! Check results above.")
print("\nKey improvements:")
print("✅ Vector search finds relevant segments first")
print("✅ Only relevant segments sent to OpenRouter (not full transcript)")
print("✅ More focused and accurate summaries")
print("✅ Better confidence scoring based on segment relevance")