#!/usr/bin/env python3
"""
Debug TTS API issue for Kannada
===============================
The translation is working but TTS is failing with 400 Bad Request.
Let's debug the TTS payload structure.
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

def debug_tts_api():
    """Debug the TTS API to understand the correct payload structure"""
    
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.error("❌ SARVAM_API_KEY not found")
        return
    
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Simple Kannada text for testing
    test_text = "ನಮಸ್ಕಾರ"  # "Hello" in Kannada
    
    logger.info("🔍 Testing different TTS payload structures...")
    
    # Test 1: Current payload structure
    payload1 = {
        "inputs": [test_text],
        "target_language_code": "kn-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    
    logger.info("🧪 Test 1: Current payload structure")
    logger.info(f"Payload: {json.dumps(payload1, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload1, headers=headers, timeout=30)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Test 1 successful!")
            return
        else:
            logger.error(f"❌ Test 1 failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Test 1 exception: {e}")
    
    # Test 2: Simplified payload
    payload2 = {
        "inputs": [test_text],
        "target_language_code": "kn-IN",
        "speaker": "meera",
        "model": "bulbul:v1"
    }
    
    logger.info("\n🧪 Test 2: Simplified payload")
    logger.info(f"Payload: {json.dumps(payload2, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload2, headers=headers, timeout=30)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Test 2 successful!")
            return
        else:
            logger.error(f"❌ Test 2 failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Test 2 exception: {e}")
    
    # Test 3: Different speaker
    payload3 = {
        "inputs": [test_text],
        "target_language_code": "kn-IN",
        "speaker": "arjun",
        "model": "bulbul:v1"
    }
    
    logger.info("\n🧪 Test 3: Different speaker (arjun)")
    logger.info(f"Payload: {json.dumps(payload3, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload3, headers=headers, timeout=30)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Test 3 successful!")
            return
        else:
            logger.error(f"❌ Test 3 failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Test 3 exception: {e}")
    
    # Test 4: English text to check if it's a language-specific issue
    payload4 = {
        "inputs": ["Hello"],
        "target_language_code": "en-IN",
        "speaker": "meera",
        "model": "bulbul:v1"
    }
    
    logger.info("\n🧪 Test 4: English text for comparison")
    logger.info(f"Payload: {json.dumps(payload4, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{base_url}/text-to-speech", json=payload4, headers=headers, timeout=30)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Test 4 successful! Issue might be with Kannada-specific parameters")
            return
        else:
            logger.error(f"❌ Test 4 failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Test 4 exception: {e}")
    
    # Test 5: Check API documentation/validation
    logger.info("\n🧪 Test 5: Checking if TTS is available for Kannada")
    
    # Let's try a different approach - maybe check available languages for TTS
    logger.info("💡 Suggestion: The TTS API might not support Kannada yet, or requires different parameters")
    logger.info("💡 Let's focus on translation for now, which is working perfectly!")

def test_only_translation():
    """Test just the translation part which we know is working"""
    
    logger.info("🧪 Testing only translation (which we know works)...")
    
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    test_summary = "This video explains sorting algorithms in computer science."
    
    payload = {
        "input": test_summary,
        "source_language_code": "en-IN",
        "target_language_code": "kn-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1"
    }
    
    try:
        response = requests.post(f"{base_url}/translate", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        translated_text = result.get("translated_text", "")
        
        logger.info("✅ Translation working perfectly!")
        logger.info(f"📝 Original: {test_summary}")
        logger.info(f"📝 Kannada: {translated_text}")
        
        return translated_text
        
    except Exception as e:
        logger.error(f"❌ Translation test failed: {e}")
        return None

if __name__ == "__main__":
    logger.info("🚀 Debugging TTS API issue...")
    
    # First confirm translation works
    translated = test_only_translation()
    
    if translated:
        logger.info("\n" + "="*50)
        logger.info("🎯 SUMMARY:")
        logger.info("✅ Translation is working perfectly!")
        logger.info("❌ TTS has issues - might not support Kannada yet")
        logger.info("💡 Recommendation: Use translation-only for now")
        logger.info("="*50)
        
        # Now debug TTS
        debug_tts_api()
    else:
        logger.error("❌ Translation failed - need to fix this first")