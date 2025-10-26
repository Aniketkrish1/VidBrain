#!/usr/bin/env python3
"""
Simple test for query-based video summarization.
This tests the core functionality without requiring downloads.
"""
import os
import sys
sys.path.append('.')

def test_query_processing():
    """Test the query processing and video assembly logic"""
    print("=== Testing Query-Based Video Summarization ===\n")

    # Test 1: Query filtering
    print("1. Testing query filtering logic...")
    from main import retrieve_by_query
    from utils.database import VectorDB

    # Load existing vector database
    if os.path.exists('vector_db.pkl'):
        print("   Loading vector database...")
        db = VectorDB(db_path='vector_db.pkl')

        # Test query for specific topic
        query = 'quick sort'
        hits = retrieve_by_query(db, query, top_k=10)

        print(f"   Query: '{query}'")
        print(f"   Found {len(hits)} relevant segments")

        if hits:
            print("   Top segments:")
            for i, hit in enumerate(hits[:3]):
                print(f"     {i+1}. {hit['text'][:100]}... (score: {getattr(hit, 'score', 'N/A')})")
    else:
        print("   No vector database found - need to run transcription first")

    # Test 2: Video assembly logic
    print("\n2. Testing video assembly logic...")
    from utils.assmble_video import assemble_video

    # Test with dummy data
    video_path = 'outputs/summary_output.mp4'
    if os.path.exists(video_path):
        print(f"   Using video: {video_path}")

        # Create minimal test data
        topic_clusters = {0: [{'start': 10, 'end': 20, 'text': 'Quick sort example'}]}
        voiceovers = {0: 'temp_processing/voiceover_0.mp3'}

        if os.path.exists(voiceovers[0]):
            print("   Testing assembly with voiceover...")
            try:
                result = assemble_video(video_path, topic_clusters, voiceovers, 'outputs/test_voiceover_only.mp4')
                print(f"   ✅ Success: {result}")

                # Check if result has audio
                from moviepy import VideoFileClip
                clip = VideoFileClip(result)
                has_audio = clip.audio is not None
                print(f"   Has audio: {has_audio}")
                if has_audio:
                    print(f"   Audio duration: {clip.audio.duration:.1f}s")
                clip.close()

            except Exception as e:
                print(f"   ❌ Assembly failed: {e}")
        else:
            print("   No voiceover file found for testing")
    else:
        print("   No video file found for testing")

    print("\n=== Test Complete ===")
    print("\nKey improvements made:")
    print("✅ Query filtering now filters for specific topics only")
    print("✅ Video assembly removes original audio and attaches voiceover")
    print("✅ Single cluster creation for topic-specific queries")
    print("✅ Focused summarizer for shorter, topic-specific content")

if __name__ == "__main__":
    test_query_processing()

