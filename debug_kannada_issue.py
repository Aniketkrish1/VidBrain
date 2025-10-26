"""
Quick test to verify Sarvam AI translation is working with the API key.
"""

import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sarvam_translation():
    """Test Sarvam AI translation functionality"""
    
    load_dotenv(override=True)
    
    try:
        from utils.sarvam_ai import SarvamAI
        
        # Test text (similar to what would be generated in summary)
        test_text = """Bubble sort is a simple comparison-based sorting algorithm that iterates over a list multiple times, comparing adjacent elements and swapping them if they are in the wrong order. This process is repeated until no more swaps are needed, indicating that the list is sorted."""
        
        logger.info("🧪 Testing Sarvam AI translation to Kannada...")
        logger.info(f"Original text: {test_text[:100]}...")
        
        sarvam = SarvamAI()
        
        # Test translation to Kannada (what user selected)
        translated = sarvam.translate_text(test_text, "kn", "en")
        
        logger.info(f"✅ Translation successful!")
        logger.info(f"Translated to Kannada: {translated}")
        
        # Test if it actually changed
        if translated != test_text:
            logger.info("✅ Translation appears to be working - text was changed")
            return True
        else:
            logger.warning("⚠️ Translation returned same text - might be an issue")
            return False
            
    except Exception as e:
        logger.error(f"❌ Translation test failed: {e}")
        return False

def test_api_key():
    """Test if API key is properly loaded"""
    load_dotenv(override=True)
    
    api_key = os.getenv("SARVAM_API_KEY")
    if api_key:
        logger.info(f"✅ Sarvam API key found: {api_key[:10]}...")
        return True
    else:
        logger.error("❌ Sarvam API key not found")
        return False

if __name__ == "__main__":
    logger.info("🔍 Debugging Kannada translation issue...")
    
    # Test 1: API key
    if not test_api_key():
        logger.error("❌ API key issue - fix this first")
        exit(1)
    
    # Test 2: Translation
    if test_sarvam_translation():
        logger.info("✅ Sarvam AI translation is working")
        logger.info("💡 The issue might be in the pipeline logic")
    else:
        logger.error("❌ Sarvam AI translation is not working")
        logger.info("💡 Check API key validity or network connection")