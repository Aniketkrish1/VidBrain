#!/usr/bin/env python3
"""
Test with bulbul:v2 model
=========================
The v3-beta requires special access, let's try v2 which should be available.
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

def test_v2_model():
    """Test TTS with bulbul:v2 model"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Test with v2 model (should be available)
    payload = {
        "inputs": ["ನಮಸ್ಕಾರ"],
        "target_language_code": "kn-IN",
        "speaker": "meera",  # Different speakers for v2
        "pace": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    logger.info("🧪 Testing bulbul:v2 model...")
    logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text[:500]}...")  # First 500 chars
        
        if response.status_code == 200:
            result = response.json()
            audio_base64 = result.get("audios", [None])[0]
            
            if audio_base64:
                import base64
                audio_data = base64.b64decode(audio_base64)
                
                logger.info(f"✅ TTS SUCCESS with v2 model!")
                logger.info(f"📊 Audio data: {len(audio_data)} bytes")
                
                # Save the audio
                with open("v2_kannada_test.wav", "wb") as f:
                    f.write(audio_data)
                
                logger.info("💾 Audio saved to: v2_kannada_test.wav")
                
                return True
            else:
                logger.error("❌ No audio data in response")
                return False
        else:
            logger.error(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return False

def test_minimal_v2():
    """Test with minimal v2 payload"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Absolutely minimal payload for v2
    payload = {
        "inputs": ["Hello"],
        "target_language_code": "en-IN",
        "speaker": "meera",
        "model": "bulbul:v2"
    }
    
    logger.info("🧪 Testing minimal v2 payload...")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Minimal v2 works!")
            return True
        else:
            logger.error(f"❌ Minimal v2 failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Testing bulbul:v2 model...")
    
    # Test 1: Minimal v2 with English
    logger.info("=" * 50)
    logger.info("TEST 1: Minimal v2 with English")
    logger.info("=" * 50)
    success1 = test_minimal_v2()
    
    # Test 2: v2 with Kannada
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: v2 with Kannada")
    logger.info("=" * 50)
    success2 = test_v2_model()
    
    # Results
    logger.info("\n" + "=" * 50)
    logger.info("🎯 RESULTS:")
    logger.info(f"English v2: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    logger.info(f"Kannada v2: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1 or success2:
        logger.info("✅ v2 model works! Let's update our code to use v2")
    else:
        logger.warning("⚠️ TTS might not be available for this API key")
    
    logger.info("=" * 50)