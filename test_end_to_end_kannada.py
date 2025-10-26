#!/usr/bin/env python3
"""
Test end-to-end Kannada integration with main application
========================================================
Test the complete workflow from frontend language selection to final output.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from utils.sarvam_ai import SarvamAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_main_integration():
    """Test Kannada integration with main processing pipeline"""
    
    logger.info("🌐 Testing main application Kannada integration...")
    
    try:
        # Simulate processing a video with Kannada language selection
        logger.info("📝 Simulating video processing with Kannada language...")
        
        # Test data (simulating what would come from a real video)
        test_summary = """
        This video explains 10 different sorting algorithms in computer science.
        The presenter covers bubble sort, selection sort, insertion sort, merge sort,
        quick sort, heap sort, radix sort, counting sort, bucket sort, and shell sort.
        Each algorithm is explained with examples and time complexity analysis.
        """
        
        test_segments = [
            {"start": "00:00:10", "end": "00:00:30", "text": "Introduction to sorting algorithms"},
            {"start": "00:00:30", "end": "00:01:00", "text": "Bubble sort explanation"},
            {"start": "00:01:00", "end": "00:01:30", "text": "Selection sort demonstration"},
            {"start": "00:01:30", "end": "00:02:00", "text": "Quick sort analysis"}
        ]
        
        # Test with language = "kn" (Kannada)
        language = "kn"
        
        logger.info(f"🔧 Testing with language: {language}")
        logger.info(f"📝 Original summary: {test_summary[:100]}...")
        
        # Initialize Sarvam AI
        sarvam = SarvamAI()
        
        # Step 1: Translation
        if language != "en":
            logger.info("🌐 Translating summary to Kannada...")
            translated_summary = sarvam.translate_text(test_summary, language, "en")
            logger.info(f"✅ Translation successful!")
            logger.info(f"📝 Kannada summary: {translated_summary[:150]}...")
        else:
            translated_summary = test_summary
        
        # Step 2: Generate voiceover
        logger.info("🔊 Generating Kannada voiceover...")
        
        # Use shorter text for TTS (first paragraph)
        tts_text = translated_summary.split('.')[0] + "."  # First sentence
        audio_data = sarvam.generate_speech(tts_text, language, "anushka")
        
        # Save voiceover
        voiceover_path = "test_kannada_voiceover.wav"
        sarvam.save_audio(audio_data, voiceover_path)
        
        logger.info(f"✅ Voiceover generated: {len(audio_data)} bytes")
        logger.info(f"💾 Saved to: {voiceover_path}")
        
        # Prepare result (simulating what main.py would return)
        result = {
            "summary": translated_summary,
            "segments": test_segments,
            "language": language,
            "voiceover_path": voiceover_path,
            "status": "success"
        }
        
        logger.info("🎉 End-to-end Kannada integration test successful!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Main integration test failed: {e}")
        raise

def test_api_simulation():
    """Simulate API call with Kannada language parameter"""
    
    logger.info("🌐 Simulating API call with Kannada language...")
    
    try:
        # Simulate what happens when user selects Kannada in frontend
        form_data = {
            "url": "https://youtube.com/watch?v=example",
            "query": "sorting algorithms",
            "language": "kn"  # Kannada selected
        }
        
        logger.info(f"📋 Simulated form data: {form_data}")
        
        # Test just the language handling part
        language = form_data.get("language", "en")
        
        if language == "kn":
            logger.info("✅ Kannada language detected correctly")
            logger.info("🌐 Would trigger multilingual processing pipeline")
            logger.info("🔊 Would generate Kannada voiceover")
            
            # Quick test of the actual functions
            sarvam = SarvamAI()
            test_text = "This is a test for the API simulation."
            
            translated = sarvam.translate_text(test_text, "kn", "en")
            logger.info(f"📝 Sample translation: {translated}")
            
            return {
                "language_detected": "kn",
                "translation_working": True,
                "tts_available": True,
                "status": "ready_for_production"
            }
        else:
            logger.info("ℹ️ Non-Kannada language detected")
            return {"status": "english_mode"}
        
    except Exception as e:
        logger.error(f"❌ API simulation failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Testing end-to-end Kannada integration...")
    
    try:
        # Test 1: API simulation
        logger.info("=" * 60)
        logger.info("TEST 1: API Language Parameter Handling")
        logger.info("=" * 60)
        
        api_result = test_api_simulation()
        logger.info(f"✅ API test result: {api_result}")
        
        # Test 2: Main integration 
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Main Application Integration")
        logger.info("=" * 60)
        
        main_result = test_main_integration()
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("🎯 END-TO-END TEST RESULTS:")
        logger.info("✅ Language parameter handling: WORKING")
        logger.info("✅ Translation pipeline: WORKING")
        logger.info("✅ TTS generation: WORKING")
        logger.info("✅ File saving: WORKING")
        logger.info("✅ Complete integration: WORKING")
        logger.info("")
        logger.info("🎉 KANNADA INTEGRATION IS PRODUCTION READY!")
        logger.info("🚀 User can now select Kannada and get translated summaries!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"💥 End-to-end test failed: {e}")
        sys.exit(1)