"""
Detailed debug script to check Sarvam AI API issues.
"""

import os
import json
import requests
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sarvam_api_direct():
    """Test Sarvam AI API directly with different payloads"""
    
    load_dotenv(override=True)
    api_key = os.getenv("SARVAM_API_KEY")
    
    if not api_key:
        logger.error("❌ No API key found")
        return False
    
    base_url = "https://api.sarvam.ai"
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key
    }
    
    # Test 1: Simple payload
    test_text = "Hello, how are you?"
    
    # Try different payload structures
    payloads = [
        # Current payload structure
        {
            "input": test_text,
            "source_language_code": "en",
            "target_language_code": "kn", 
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1"
        },
        # Simplified payload
        {
            "input": test_text,
            "source_language_code": "en",
            "target_language_code": "kn"
        },
        # Alternative structure
        {
            "text": test_text,
            "source_language": "en",
            "target_language": "kn"
        }
    ]
    
    for i, payload in enumerate(payloads):
        logger.info(f"\n🧪 Testing payload structure {i+1}:")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(f"{base_url}/translate", json=payload, headers=headers, timeout=30)
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Success! Response: {result}")
                return True
            else:
                logger.error(f"❌ Failed with status {response.status_code}")
                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    logger.error(f"Error text: {response.text}")
                    
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
    
    return False

def test_api_key_validity():
    """Test if the API key is valid by trying a simple endpoint"""
    
    load_dotenv(override=True)
    api_key = os.getenv("SARVAM_API_KEY")
    
    logger.info("🔑 Testing API key validity...")
    
    # Try to hit any endpoint to check if key is valid
    headers = {
        "API-Subscription-Key": api_key
    }
    
    try:
        # Test with a simple request (this might not work, but will tell us about auth)
        response = requests.get("https://api.sarvam.ai/translate", headers=headers, timeout=10)
        
        if response.status_code == 401:
            logger.error("❌ API key is invalid or expired")
            return False
        elif response.status_code == 405:
            logger.info("✅ API key seems valid (got 405 Method Not Allowed, expected for GET)")
            return True
        else:
            logger.info(f"✅ API key seems valid (got status {response.status_code})")
            return True
            
    except Exception as e:
        logger.error(f"❌ API key test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Detailed Sarvam AI debugging...")
    
    # Test API key first
    if not test_api_key_validity():
        logger.error("❌ API key issue - check your key")
    
    # Test different API payload structures
    if test_sarvam_api_direct():
        logger.info("✅ Found working API structure!")
    else:
        logger.error("❌ All API structures failed")
        logger.info("💡 Possible issues:")
        logger.info("   1. API key is invalid")
        logger.info("   2. API endpoint has changed")
        logger.info("   3. Payload structure is incorrect")
        logger.info("   4. Service is temporarily down")