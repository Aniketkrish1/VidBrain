#!/usr/bin/env python3
"""
Demo: New Focused Vector Search + OpenRouter Summarization

This demonstrates the NEW approach:
1. Vector search finds relevant segments for user's query
2. Only those segments are sent to OpenRouter (not full transcript)
3. AI generates focused, accurate summary
4. Display result with confidence and timestamps
"""

import sys
import os
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv(override=True)

print("🎯 VidBrain: Focused Vector Search + AI Summarization")
print("=" * 60)

# Test the simple API first
from simple_api_test import *
print("\n1. Testing OpenRouter API...")
api_key = os.getenv("OPENROUTER_API_KEY", "")

if api_key:
    print(f"✅ API Key loaded: {api_key[:15]}...")
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Test with reliable model
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct", 
            messages=[{"role": "user", "content": "Explain bubble sort in one sentence."}],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API Test Result: {result}")
        
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        sys.exit(1)
else:
    print("❌ No API key found")
    sys.exit(1)

print("\n2. Testing Vector Search + Focused Summarization...")

# Import our modules
from utils.database import VectorDB
from utils.topic_query_processor import process_topic_query

# Mock realistic transcript data
mock_transcript = [
    {"text": "Welcome to our computer science tutorial", "start": 0.0, "end": 3.0},
    {"text": "Today we will cover sorting algorithms", "start": 3.5, "end": 6.0},
    {"text": "Merge sort is a divide and conquer algorithm", "start": 10.0, "end": 13.0},
    {"text": "It recursively divides the array into halves", "start": 13.5, "end": 16.0},
    {"text": "Then merges the sorted halves back together", "start": 16.5, "end": 19.0},
    {"text": "The time complexity is O(n log n)", "start": 19.5, "end": 22.0},
    {"text": "Bubble sort is much simpler but less efficient", "start": 30.0, "end": 33.0},
    {"text": "It compares adjacent elements and swaps them if needed", "start": 33.5, "end": 37.0},
    {"text": "This process repeats until no more swaps are needed", "start": 37.5, "end": 41.0},
    {"text": "Bubble sort has O(n²) time complexity in worst case", "start": 41.5, "end": 45.0},
    {"text": "Let's also discuss binary search algorithms", "start": 50.0, "end": 53.0},
    {"text": "Binary search works on sorted arrays only", "start": 53.5, "end": 56.0},
    {"text": "It eliminates half the search space each iteration", "start": 56.5, "end": 60.0},
]

# Build vector database
print("Building vector database...")
db = VectorDB(model_name="multi-qa-mpnet-base-dot-v1", db_path="demo_vector_db.pkl")
db.build(mock_transcript)

# Test focused queries
test_queries = ["merge sort", "bubble sort", "binary search"]

for query in test_queries:
    print(f"\n--- Query: '{query}' ---")
    
    try:
        result = process_topic_query(db, query, mock_transcript)
        
        print(f"✅ Found {result.get('total_segments', 0)} relevant segments")
        print(f"✅ Confidence: {result.get('confidence', 0):.1%}")
        print(f"✅ Summary ({len(result.get('summary', ''))} chars):")
        print(f"   {result.get('summary', 'No summary')[:150]}...")
        
        # Show relevant segments
        segments = result.get('segments', [])[:3]  # Top 3
        if segments:
            print("✅ Top relevant segments:")
            for i, seg in enumerate(segments, 1):
                print(f"   {i}. [{seg['start']:.1f}s] {seg['text'][:60]}... (score: {seg.get('score', 0):.2f})")
        
    except Exception as e:
        print(f"❌ Query failed: {e}")

# Cleanup
print(f"\nCleaning up...")
try:
    if os.path.exists("demo_vector_db.pkl"):
        os.remove("demo_vector_db.pkl")
except:
    pass

print("\n" + "=" * 60)
print("🎯 NEW APPROACH BENEFITS:")
print("✅ Vector search identifies relevant content first")
print("✅ Only focused segments sent to AI (not full transcript)")
print("✅ More accurate, topic-specific summaries")
print("✅ Confidence scoring based on relevance")
print("✅ Faster processing (less data to analyze)")
print("✅ Better resource usage (fewer API tokens)")

print("\n🚀 Ready for frontend integration!")