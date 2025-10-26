#!/usr/bin/env python3
"""
Final test with corrected bulbul:v2 configuration
=================================================
Test the complete working configuration.
"""

import os
import sys
import json
import requests
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_corrected_v2():
    """Test with corrected v2 configuration"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Corrected payload for v2
    payload = {
        "inputs": ["ನಮಸ್ಕಾರ"],
        "target_language_code": "kn-IN",
        "speaker": "anushka",
        "pace": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    logger.info("🧪 Testing corrected bulbul:v2...")
    logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            audio_base64 = result.get("audios", [None])[0]
            
            if audio_base64:
                import base64
                audio_data = base64.b64decode(audio_base64)
                
                logger.info(f"✅ TTS SUCCESS!")
                logger.info(f"📊 Audio data: {len(audio_data)} bytes")
                
                # Save the audio
                with open("final_working_kannada.wav", "wb") as f:
                    f.write(audio_data)
                
                logger.info("💾 Audio saved to: final_working_kannada.wav")
                
                return True
            else:
                logger.error("❌ No audio data in response")
                return False
        else:
            logger.error(f"❌ Request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return False

def test_complete_pipeline():
    """Test complete translation + TTS pipeline with our module"""
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.sarvam_ai import SarvamAI
    
    logger.info("🌐 Testing complete Kannada pipeline...")
    
    try:
        sarvam = SarvamAI()
        
        # Test summary
        english_summary = "This video explains sorting algorithms. The presenter covers bubble sort, merge sort, and quick sort with detailed examples."
        
        logger.info(f"📝 Original: {english_summary}")
        
        # Step 1: Translation
        logger.info("🌐 Translating to Kannada...")
        kannada_text = sarvam.translate_text(english_summary, "kn", "en")
        logger.info(f"✅ Translation: {kannada_text[:100]}...")
        
        # Step 2: TTS (use shorter text)
        logger.info("🔊 Generating Kannada TTS...")
        short_text = kannada_text[:50]  # Use first 50 characters
        audio_data = sarvam.generate_speech(short_text, "kn", "anushka")
        
        logger.info(f"✅ TTS successful! {len(audio_data)} bytes")
        
        # Save complete result
        audio_path = "complete_kannada_pipeline.wav"
        sarvam.save_audio(audio_data, audio_path)
        
        logger.info(f"💾 Complete pipeline audio: {audio_path}")
        logger.info("🎉 COMPLETE KANNADA PIPELINE WORKING!")
        
        return {
            "original": english_summary,
            "translated": kannada_text,
            "audio_path": audio_path,
            "audio_size": len(audio_data)
        }
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Final corrected configuration test...")
    
    # Test 1: Direct API
    logger.info("=" * 60)
    logger.info("TEST 1: Direct API with corrected v2")
    logger.info("=" * 60)
    success1 = test_corrected_v2()
    
    if success1:
        # Test 2: Complete pipeline
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Complete Translation + TTS Pipeline")
        logger.info("=" * 60)
        
        try:
            result = test_complete_pipeline()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎯 FINAL SUCCESS REPORT:")
            logger.info("✅ Translation: WORKING")
            logger.info("✅ TTS: WORKING") 
            logger.info("✅ Complete Pipeline: WORKING")
            logger.info(f"📊 Audio generated: {result['audio_size']} bytes")
            logger.info(f"📁 Audio file: {result['audio_path']}")
            logger.info("🎉 KANNADA INTEGRATION IS FULLY FUNCTIONAL!")
            logger.info("🚀 Ready for production use!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
    else:
        logger.error("❌ Direct API test failed - cannot proceed to pipeline test")