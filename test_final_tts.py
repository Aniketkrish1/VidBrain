#!/usr/bin/env python3
"""
Final test of corrected TTS API
===============================
Test with all the correct parameters for bulbul:v3-beta model.
"""

import os
import json
import requests
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_final_tts():
    """Test TTS with all correct parameters"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Final corrected payload
    payload = {
        "inputs": ["ನಮಸ್ಕಾರ"],
        "target_language_code": "kn-IN",
        "speaker": "isha",
        "pace": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v3-beta"
    }
    
    logger.info("🧪 Testing final corrected TTS...")
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
                with open("final_kannada_test.wav", "wb") as f:
                    f.write(audio_data)
                
                logger.info("💾 Audio saved to: final_kannada_test.wav")
                
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

def test_with_sarvam_module():
    """Test using our updated sarvam_ai module"""
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.sarvam_ai import SarvamAI
    
    logger.info("🧪 Testing with updated sarvam_ai module...")
    
    try:
        sarvam = SarvamAI()
        
        # Test simple Kannada TTS
        audio_data = sarvam.generate_speech("ನಮಸ್ಕಾರ", "kn", "isha")
        
        logger.info(f"✅ Module TTS SUCCESS!")
        logger.info(f"📊 Audio data: {len(audio_data)} bytes")
        
        # Save audio
        sarvam.save_audio(audio_data, "module_kannada_test.wav")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Module test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Final TTS correction test...")
    
    # Test 1: Direct API call
    logger.info("=" * 50)
    logger.info("TEST 1: Direct API Call")
    logger.info("=" * 50)
    success1 = test_final_tts()
    
    # Test 2: Using our module
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: Using SarvamAI Module")
    logger.info("=" * 50)
    success2 = test_with_sarvam_module()
    
    # Results
    logger.info("\n" + "=" * 50)
    logger.info("🎯 FINAL RESULTS:")
    logger.info(f"Direct API: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    logger.info(f"Module API: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        logger.info("🎉 Kannada TTS is now fully working!")
        logger.info("🎉 Ready to test with the main application!")
    else:
        logger.error("❌ Still have issues to resolve")
    
    logger.info("=" * 50)