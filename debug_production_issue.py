#!/usr/bin/env python3
"""
Debug the exact TTS error in production
======================================
Check what's causing the 400 error in the actual production scenario.
"""

import os
import sys
import json
import requests
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sarvam_ai import SarvamAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_production_tts_error():
    """Debug the exact TTS error happening in production"""
    
    logger.info("🔍 Debugging production TTS error...")
    
    # Use the exact text that failed in production
    production_text = "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ. ಇದು ವಿಂಗಡಿಸಬೇಕಾದ ಪಟ್ಟಿಯ ಮೂಲಕ ಪದೇ ಪದೇ ಹೆಜ್"
    
    logger.info(f"📝 Production text: {production_text}")
    logger.info(f"📏 Text length: {len(production_text)} characters")
    
    # Test with direct API call first
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Exact payload structure
    payload = {
        "inputs": [production_text],
        "target_language_code": "kn-IN",
        "speaker": "anushka",
        "pace": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    logger.info("🧪 Testing direct API call...")
    logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code != 200:
            logger.error(f"❌ API Error: {response.status_code}")
            logger.error(f"Error details: {response.text}")
            
            # Try with shorter text
            short_text = production_text[:50]
            logger.info(f"🔄 Trying with shorter text: {short_text}")
            
            payload["inputs"] = [short_text]
            response2 = requests.post(f"{base_url}/text-to-speech", json=payload, headers=headers, timeout=30)
            logger.info(f"Short text status: {response2.status_code}")
            logger.info(f"Short text response: {response2.text}")
        
    except Exception as e:
        logger.error(f"❌ Direct API failed: {e}")

def test_with_sarvam_module():
    """Test with our SarvamAI module"""
    
    logger.info("🧪 Testing with SarvamAI module...")
    
    try:
        sarvam = SarvamAI()
        
        # Use the exact text from production
        production_text = "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ"
        
        logger.info(f"📝 Testing text: {production_text}")
        
        # Try TTS
        audio_data = sarvam.generate_speech(production_text, "kn", "anushka")
        
        logger.info(f"✅ Module TTS successful: {len(audio_data)} bytes")
        
        # Save audio
        sarvam.save_audio(audio_data, "debug_production_tts.wav")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Module TTS failed: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        return False

def test_character_limits():
    """Test if there are character limits causing the issue"""
    
    logger.info("📏 Testing character limits...")
    
    sarvam = SarvamAI()
    
    test_texts = [
        "ನಮಸ್ಕಾರ",  # Very short
        "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿ",  # Medium
        "ಬಬಲ್ ಸಾರ್ಟ್ ಒಂದು ಸರಳ ಹೋಲಿಕೆ-ಆಧಾರಿತ ವಿಂಗಡಣಾ ಕ್ರಮಾವಳಿಯಾಗಿದೆ. ಇದು ವಿಂಗಡಿಸಬೇಕಾದ ಪಟ್ಟಿಯ ಮೂಲಕ ಪದೇ ಪದೇ ಹೆಜ್ಜೆ ಹಾಕುವ ಮೂಲಕ ಕೆಲಸ ಮಾಡುತ್ತದೆ"  # Long
    ]
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"🧪 Test {i}: {len(text)} chars - {text[:30]}...")
        
        try:
            audio_data = sarvam.generate_speech(text, "kn", "anushka")
            logger.info(f"✅ Test {i} successful: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"❌ Test {i} failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Debugging production TTS error...")
    
    # Test 1: Direct API with production text
    logger.info("=" * 60)
    logger.info("TEST 1: Direct API with Production Text")
    logger.info("=" * 60)
    debug_production_tts_error()
    
    # Test 2: Module test
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: SarvamAI Module Test")
    logger.info("=" * 60)
    module_success = test_with_sarvam_module()
    
    # Test 3: Character limits
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Character Limit Testing")
    logger.info("=" * 60)
    test_character_limits()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 DEBUGGING COMPLETE")
    logger.info("=" * 60)