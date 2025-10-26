#!/usr/bin/env python3
"""
Test corrected TTS API with proper speaker and model
====================================================
Test the TTS API with the correct speaker names and model version.
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

def test_corrected_tts():
    """Test TTS with corrected speaker and model parameters"""
    
    logger.info("🧪 Testing corrected TTS API...")
    
    try:
        sarvam = SarvamAI()
        
        # Test with simple Kannada text
        test_text = "ನಮಸ್ಕಾರ"  # "Hello" in Kannada
        
        logger.info(f"🔊 Generating TTS for: {test_text}")
        
        # Test TTS with corrected parameters
        audio_data = sarvam.generate_speech(test_text, "kn", "anushka")
        
        logger.info(f"✅ TTS successful! Generated {len(audio_data)} bytes")
        
        # Save test audio
        audio_path = "test_corrected_tts.wav"
        sarvam.save_audio(audio_data, audio_path)
        
        logger.info(f"💾 Audio saved to: {audio_path}")
        
        return audio_path
        
    except Exception as e:
        logger.error(f"❌ TTS test failed: {e}")
        raise

def test_full_kannada_pipeline():
    """Test complete translation + TTS pipeline"""
    
    logger.info("🌐 Testing full Kannada pipeline...")
    
    try:
        sarvam = SarvamAI()
        
        # Sample video summary
        english_summary = """
        This video explains different sorting algorithms in computer science. 
        The presenter covers bubble sort, selection sort, and merge sort with examples.
        """
        
        logger.info("📝 Starting full pipeline...")
        logger.info(f"Original: {english_summary.strip()}")
        
        # Step 1: Translate
        logger.info("🌐 Translating to Kannada...")
        kannada_text = sarvam.translate_text(english_summary, "kn", "en")
        logger.info(f"✅ Translation: {kannada_text[:100]}...")
        
        # Step 2: Generate TTS (use shorter text for TTS)
        logger.info("🔊 Generating Kannada TTS...")
        short_text = kannada_text[:100]  # First 100 characters for TTS
        audio_data = sarvam.generate_speech(short_text, "kn", "anushka")
        
        logger.info(f"✅ TTS successful! Generated {len(audio_data)} bytes")
        
        # Save audio
        audio_path = "kannada_summary_tts.wav"
        sarvam.save_audio(audio_data, audio_path)
        
        logger.info(f"💾 Audio saved to: {audio_path}")
        
        logger.info("🎉 Full Kannada pipeline completed successfully!")
        
        return {
            "original": english_summary,
            "translated": kannada_text,
            "audio_path": audio_path,
            "audio_size": len(audio_data)
        }
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Testing corrected TTS API...")
    
    try:
        # Test 1: Simple TTS
        logger.info("=" * 50)
        logger.info("TEST 1: Simple TTS")
        logger.info("=" * 50)
        test_corrected_tts()
        
        # Test 2: Full pipeline
        logger.info("\n" + "=" * 50)
        logger.info("TEST 2: Full Pipeline")
        logger.info("=" * 50)
        result = test_full_kannada_pipeline()
        
        logger.info("\n" + "=" * 50)
        logger.info("🎯 FINAL RESULTS:")
        logger.info("✅ Translation: Working perfectly")
        logger.info("✅ TTS: Working with corrected parameters")
        logger.info(f"📊 Audio size: {result['audio_size']} bytes")
        logger.info(f"📁 Audio file: {result['audio_path']}")
        logger.info("🎉 Kannada integration is now fully functional!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        sys.exit(1)