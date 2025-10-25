"""
test_topic_query.py

Test script for the new topic-based video summarization system.
Run this to verify all components are working correctly.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.topic_query_processor import process_topic_query, search_topic_in_transcript
        print("✓ topic_query_processor imported")
    except Exception as e:
        print(f"✗ topic_query_processor import failed: {e}")
        return False
    
    try:
        from utils.clip_extractor import prepare_clips_for_topic, extract_clips_from_timestamps
        print("✓ clip_extractor imported")
    except Exception as e:
        print(f"✗ clip_extractor import failed: {e}")
        return False
    
    try:
        from utils.voiceover_generator import VoiceoverGenerator, generate_voiceover
        print("✓ voiceover_generator imported")
    except Exception as e:
        print(f"✗ voiceover_generator import failed: {e}")
        return False
    
    try:
        from utils.video_assembler import assemble_topic_video
        print("✓ video_assembler imported")
    except Exception as e:
        print(f"✗ video_assembler import failed: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_voiceover_generation():
    """Test voiceover generation."""
    print("Testing voiceover generation...")
    
    try:
        from utils.voiceover_generator import generate_voiceover
        
        test_text = "This is a test of the voiceover generation system. It should create clear, natural speech."
        test_output = "test_voiceover.mp3"
        
        # Try to generate voiceover
        result = generate_voiceover(test_text, test_output, engine="auto")
        
        if os.path.exists(result) and os.path.getsize(result) > 0:
            print(f"✓ Voiceover generated successfully: {result}")
            print(f"  File size: {os.path.getsize(result)} bytes")
            
            # Clean up test file
            try:
                os.remove(result)
                print("  Test file cleaned up")
            except:
                pass
            
            return True
        else:
            print("✗ Voiceover file not created or empty")
            return False
            
    except Exception as e:
        print(f"✗ Voiceover generation failed: {e}")
        return False


def test_topic_query_processor():
    """Test topic query processor with mock data."""
    print("\nTesting topic query processor...")
    
    try:
        from utils.topic_query_processor import process_topic_query, merge_adjacent_segments
        from utils.database import VectorDB
        
        # Test segment merging
        test_segments = [
            {"text": "Quick sort is fast", "start": 10.0, "end": 12.0},
            {"text": "It uses divide and conquer", "start": 12.5, "end": 15.0},
            {"text": "Very efficient algorithm", "start": 20.0, "end": 22.0},
        ]
        
        groups = merge_adjacent_segments(test_segments, gap_threshold=3.0)
        
        if len(groups) == 2:  # Should merge first two, keep third separate
            print("✓ Segment merging works correctly")
            print(f"  Merged {len(test_segments)} segments into {len(groups)} groups")
            return True
        else:
            print(f"✗ Segment merging failed: expected 2 groups, got {len(groups)}")
            return False
            
    except Exception as e:
        print(f"✗ Topic query processor test failed: {e}")
        return False


def test_clip_extractor():
    """Test clip extraction utilities."""
    print("\nTesting clip extractor...")
    
    try:
        from utils.clip_extractor import convert_timestamp_to_seconds, expand_clip_duration, merge_overlapping_clips
        
        # Test timestamp conversion
        seconds = convert_timestamp_to_seconds("00:01:30,500")
        if abs(seconds - 90.5) < 0.01:
            print("✓ Timestamp conversion works")
        else:
            print(f"✗ Timestamp conversion failed: {seconds} != 90.5")
            return False
        
        # Test clip expansion
        start, end = expand_clip_duration(10.0, 11.0, min_duration=3.0)
        if end - start >= 3.0:
            print("✓ Clip expansion works")
        else:
            print(f"✗ Clip expansion failed: duration {end-start} < 3.0")
            return False
        
        # Test clip merging
        clips = [(10, 15), (16, 20), (30, 35)]
        merged = merge_overlapping_clips(clips, gap_threshold=2.0)
        if len(merged) == 2:  # First two should merge
            print("✓ Clip merging works")
            return True
        else:
            print(f"✗ Clip merging failed: expected 2 clips, got {len(merged)}")
            return False
            
    except Exception as e:
        print(f"✗ Clip extractor test failed: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        "moviepy": "Video processing",
        "openai": "OpenRouter API",
        "sentence_transformers": "Vector embeddings",
        "faster_whisper": "Transcription",
        "torch": "GPU acceleration",
        "spacy": "NLP processing",
    }
    
    optional_dependencies = {
        "edge_tts": "High-quality TTS (optional)",
        "pyttsx3": "Offline TTS (fallback)",
    }
    
    all_good = True
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package:25} - {description}")
        except ImportError:
            print(f"✗ {package:25} - {description} (REQUIRED)")
            all_good = False
    
    print("\nOptional dependencies:")
    for package, description in optional_dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package:25} - {description}")
        except ImportError:
            print(f"○ {package:25} - {description} (optional)")
    
    return all_good


def main():
    """Run all tests."""
    print("="*60)
    print("VidBrain Topic Query System - Test Suite")
    print("="*60)
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Some required dependencies are missing!")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n" + "="*60)
    print("Running Component Tests")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Topic Query Processor", test_topic_query_processor),
        ("Clip Extractor", test_clip_extractor),
        ("Voiceover Generation", test_voiceover_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print("\n" + "-"*60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
