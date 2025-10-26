#!/usr/bin/env python3
"""
Quick test to verify bulbul:v2 model is working and summaries are being translated
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sarvam_ai import SarvamAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bulbul_v2_direct():
    """Test bulbul:v2 model directly with production data"""
    
    logger.info("🎯 Testing bulbul:v2 model configuration...")
    
    # Production text from logs
    text = "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ"
    language = "kn"
    
    try:
        sarvam = SarvamAI()
        logger.info(f"✅ SarvamAI initialized")
        
        # Test 1: Translation
        logger.info(f"🌐 Testing translation...")
        translated = sarvam.translate_text(text, "kn", "en")
        logger.info(f"📝 Translation result: {translated[:50]}...")
        
        # Test 2: TTS with bulbul:v2
        logger.info(f"🎵 Testing TTS with bulbul:v2...")
        audio_data = sarvam.generate_speech(translated, "kn", "anushka")
        logger.info(f"🎤 Audio generated: {len(audio_data)} bytes")
        
        # Test 3: Combined pipeline
        logger.info(f"🔄 Testing combined pipeline...")
        translated_text, audio_data = sarvam.translate_and_generate_speech(text, "kn", "en")
        logger.info(f"📋 Pipeline result:")
        logger.info(f"   📝 Translated: {translated_text[:50]}...")
        logger.info(f"   🎤 Audio: {len(audio_data)} bytes")
        
        # Save final result
        output_file = "test_bulbul_v2_output.wav"
        sarvam.save_audio(audio_data, output_file)
        logger.info(f"💾 Final audio saved to: {output_file}")
        
        logger.info("✅ All tests passed! bulbul:v2 is working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_frontend_response_format():
    """Test the format that would be sent to frontend"""
    
    logger.info("🌐 Testing frontend response format...")
    
    # Mock what would be sent to frontend
    summary_data = {
        "summary": "Bubble sort is a simple comparison-based sorting algorithm",
        "language": "en"
    }
    
    try:
        sarvam = SarvamAI()
        
        # Process like production
        original_text = summary_data["summary"]
        target_language = "kn"
        
        # Translate and generate TTS
        translated_text, audio_data = sarvam.translate_and_generate_speech(
            original_text, target_language, "en"
        )
        
        # Update like main.py does
        summary_data["summary"] = translated_text
        summary_data["language"] = target_language
        
        logger.info(f"📋 Frontend would receive:")
        logger.info(f"   📝 Summary: {summary_data['summary'][:50]}...")
        logger.info(f"   🌐 Language: {summary_data['language']}")
        logger.info(f"   🎤 Audio size: {len(audio_data)} bytes")
        
        # This is what frontend should see - Kannada text, not English
        if "ಬಬಲ್" in summary_data["summary"] or any(ord(c) > 127 for c in summary_data["summary"]):
            logger.info("✅ Frontend will receive translated Kannada content")
            return True
        else:
            logger.error("❌ Frontend will receive English content - translation failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Frontend test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING BULBUL:V2 MODEL AND FRONTEND INTEGRATION")
    logger.info("=" * 60)
    
    # Test 1: Direct bulbul:v2 functionality
    test1_success = test_bulbul_v2_direct()
    
    logger.info("\n" + "=" * 60)
    
    # Test 2: Frontend response format
    test2_success = test_frontend_response_format()
    
    logger.info("\n" + "=" * 60)
    overall_success = test1_success and test2_success
    logger.info(f"OVERALL RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
    logger.info("=" * 60)