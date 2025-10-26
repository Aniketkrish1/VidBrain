#!/usr/bin/env python3
"""
Test Kannada translation with fixed language codes
==================================================
This script tests the corrected Sarvam AI integration to ensure
Kannada translation works properly with region-specific language codes.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sarvam_ai import SarvamAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_kannada_translation():
    """Test Kannada translation with corrected API calls"""
    
    logger.info("🧪 Testing Kannada translation with fixed language codes...")
    
    try:
        # Initialize Sarvam AI
        sarvam = SarvamAI()
        
        # Test text (sample video summary)
        test_summary = """
        This video explains 10 different sorting algorithms in computer science. 
        The presenter covers bubble sort, selection sort, insertion sort, merge sort, 
        quick sort, heap sort, radix sort, counting sort, bucket sort, and shell sort. 
        Each algorithm is explained with examples and time complexity analysis.
        """
        
        logger.info(f"📝 Original summary: {test_summary[:100]}...")
        
        # Test translation to Kannada
        logger.info("🌐 Translating to Kannada...")
        translated_text = sarvam.translate_text(test_summary, "kn", "en")
        
        logger.info(f"✅ Translation successful!")
        logger.info(f"📝 Kannada translation: {translated_text[:200]}...")
        
        # Test TTS generation
        logger.info("🔊 Generating Kannada speech...")
        short_text = translated_text[:100]  # Use shorter text for TTS test
        audio_data = sarvam.generate_speech(short_text, "kn")
        
        logger.info(f"✅ TTS successful! Generated {len(audio_data)} bytes of audio")
        
        # Save test audio
        audio_path = "test_kannada_audio.wav"
        sarvam.save_audio(audio_data, audio_path)
        
        logger.info(f"💾 Test audio saved to: {audio_path}")
        
        logger.info("🎉 Kannada translation test completed successfully!")
        
        return {
            "original_text": test_summary,
            "translated_text": translated_text,
            "audio_path": audio_path,
            "audio_size": len(audio_data)
        }
        
    except Exception as e:
        logger.error(f"❌ Kannada translation test failed: {e}")
        logger.error(f"Error details: {str(e)}")
        raise

def test_language_code_mapping():
    """Test that all language codes are mapped correctly"""
    
    logger.info("🗺️ Testing language code mapping...")
    
    try:
        sarvam = SarvamAI()
        
        # Test that all our supported languages have proper Sarvam mapping
        supported = sarvam.get_supported_languages()
        mapping = sarvam.sarvam_language_map
        
        logger.info(f"📋 Testing {len(supported)} language codes...")
        
        for code, name in supported.items():
            if code in mapping:
                sarvam_code = mapping[code]
                logger.info(f"✅ {code} ({name}) → {sarvam_code}")
            else:
                logger.warning(f"⚠️ Missing mapping for {code} ({name})")
        
        # Test specific Kannada mapping
        if "kn" in mapping and mapping["kn"] == "kn-IN":
            logger.info("✅ Kannada mapping correct: kn → kn-IN")
        else:
            logger.error("❌ Kannada mapping incorrect!")
        
        logger.info("🎉 Language code mapping test completed!")
        
    except Exception as e:
        logger.error(f"❌ Language code mapping test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Starting Kannada translation fix verification...")
    
    try:
        # Test 1: Language code mapping
        test_language_code_mapping()
        
        # Test 2: Actual Kannada translation
        result = test_kannada_translation()
        
        logger.info("🎉 All tests passed! Kannada translation fix is working!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Original text length: {len(result['original_text'])} chars")
        logger.info(f"   - Translated text length: {len(result['translated_text'])} chars") 
        logger.info(f"   - Audio data size: {result['audio_size']} bytes")
        logger.info(f"   - Audio file: {result['audio_path']}")
        
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        sys.exit(1)