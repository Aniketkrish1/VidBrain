#!/usr/bin/env python3
"""
Get detailed TTS error response
===============================
Debug exactly what's wrong with the TTS API call.
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

def test_tts_with_details():
    """Test TTS with detailed error logging"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Test with corrected parameters
    payload = {
        "inputs": ["ನಮಸ್ಕಾರ"],
        "target_language_code": "kn-IN",
        "speaker": "anushka",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v3-beta"
    }
    
    logger.info("🧪 Testing TTS with corrected parameters...")
    logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        logger.info(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            audio_base64 = result.get("audios", [None])[0]
            if audio_base64:
                logger.info(f"✅ Success! Audio data length: {len(audio_base64)}")
            else:
                logger.error("❌ No audio data in response")
        else:
            logger.error(f"❌ Request failed with status {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")

def test_minimal_payload():
    """Test with absolutely minimal payload"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Minimal payload
    payload = {
        "inputs": ["Hello"],
        "target_language_code": "en-IN",
        "speaker": "anushka",
        "model": "bulbul:v3-beta"
    }
    
    logger.info("\n🧪 Testing minimal payload...")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Minimal payload works!")
        else:
            logger.error(f"❌ Minimal payload failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")

if __name__ == "__main__":
    logger.info("🔍 Getting detailed TTS error information...")
    
    # Test 1: Corrected parameters
    test_tts_with_details()
    
    # Test 2: Minimal payload
    test_minimal_payload()