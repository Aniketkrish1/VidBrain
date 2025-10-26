#!/usr/bin/env python3
"""
Debug the exact production call that's failing
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

def test_exact_production_call():
    """Test the exact call being made in production"""
    
    # Production data from logs
    summary_text = "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ"
    language = "kn"
    source_language = "en"
    
    logger.info(f"🧪 Testing exact production call...")
    logger.info(f"📝 Text: {summary_text[:50]}...")
    logger.info(f"🌐 Language: {language}")
    logger.info(f"🔤 Source: {source_language}")
    
    try:
        # Create SarvamAI instance exactly like production
        sarvam = SarvamAI()
        logger.info(f"✅ SarvamAI initialized")
        
        # Call exactly like production
        logger.info(f"🚀 Calling translate_and_generate_speech...")
        translated_text, audio_data = sarvam.translate_and_generate_speech(
            summary_text, language, source_language
        )
        
        logger.info(f"✅ Success!")
        logger.info(f"📝 Translated: {translated_text[:50]}...")
        logger.info(f"🎵 Audio size: {len(audio_data)} bytes")
        
        # Save to file like production
        audio_path = "debug_exact_production.wav"
        sarvam.save_audio(audio_data, audio_path)
        logger.info(f"💾 Audio saved to: {audio_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ EXACT PRODUCTION CALL FAILED: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # More detailed error info
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DEBUGGING EXACT PRODUCTION CALL")
    logger.info("=" * 60)
    
    success = test_exact_production_call()
    
    logger.info("=" * 60)
    logger.info(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    logger.info("=" * 60)