#!/usr/bin/env python3
"""
Test the exact production environment setup
"""
import os
import sys
import logging
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sarvam_ai import SarvamAI

# Setup logging exactly like main.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_production_environment():
    """Test with production-like environment setup"""
    
    # Create temp directory like main.py
    TEMP_DIR = os.getenv("TEMP_DIR", "temp_processing")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Mock data like production
    summaries = {
        0: {"summary": "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ"}
    }
    language = "kn"
    
    # Mock progress callback
    def update_progress(stage, percent, details=""):
        logger.info(f"Progress: {stage} {percent}% - {details}")
    
    logger.info("🧪 Testing production environment setup...")
    
    voiceover_paths = {}
    sarvam = SarvamAI()
    
    for cid, data in summaries.items():
        summary_text = data.get("summary", "")
        if not summary_text:
            logger.warning("Empty summary for cluster %s, skipping TTS", cid)
            continue
        
        try:
            # This is the exact line from production
            update_progress("tts", 80 + (cid * 10 // len(summaries)), f"Generating {language} voiceover for topic {cid + 1}...")
            
            logger.info(f"🎯 About to call translate_and_generate_speech")
            logger.info(f"📝 Summary: {summary_text[:50]}...")
            logger.info(f"🌐 Language: {language}")
            
            # Translate and generate speech using Sarvam AI (exact production call)
            translated_text, audio_data = sarvam.translate_and_generate_speech(
                summary_text, language, "en"
            )
            
            # Save audio file (exact production call)
            audio_path = os.path.join(TEMP_DIR, f"voiceover_{cid}.wav")
            sarvam.save_audio(audio_data, audio_path)
            voiceover_paths[cid] = audio_path
            
            # Update summary with translated text for frontend display
            summaries[cid]["summary"] = translated_text
            
            logger.info(f"✅ Multilingual voiceover generated for cluster {cid}")
            
        except Exception as e:
            logger.error(f"❌ Sarvam AI TTS failed for cluster {cid}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Log the full traceback
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            
            return False
    
    logger.info(f"✅ Production test completed successfully!")
    logger.info(f"Generated files: {voiceover_paths}")
    
    return True

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING PRODUCTION ENVIRONMENT")
    logger.info("=" * 60)
    
    success = test_production_environment()
    
    logger.info("=" * 60)
    logger.info(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    logger.info("=" * 60)