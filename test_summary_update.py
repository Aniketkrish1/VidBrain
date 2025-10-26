#!/usr/bin/env python3
"""
Test the exact summary update logic from main.py
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sarvam_ai import SarvamAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_summary_update_logic():
    """Test the exact summary update logic from main.py"""
    
    logger.info("🧪 Testing summary update logic...")
    
    # Mock summaries dict like main.py
    summaries = {
        0: {
            "summary": "Bubble sort is a simple comparison-based sorting algorithm that repeatedly steps through the list.",
            "start": 0.0,
            "end": 30.0
        }
    }
    
    language = "kn"
    
    logger.info(f"📝 Original summary: {summaries[0]['summary'][:50]}...")
    logger.info(f"🌐 Target language: {language}")
    
    try:
        # Simulate exact main.py logic
        sarvam = SarvamAI()
        
        for cid, data in summaries.items():
            summary_text = data.get("summary", "")
            
            logger.info(f"🔄 Processing cluster {cid}...")
            logger.info(f"📝 Original text: {summary_text[:50]}...")
            
            # Translate and generate speech using Sarvam AI (exact main.py call)
            translated_text, audio_data = sarvam.translate_and_generate_speech(
                summary_text, language, "en"
            )
            
            logger.info(f"✅ Translation completed")
            logger.info(f"📝 Translated text: {translated_text[:50]}...")
            
            # Update summary with translated text for frontend display (exact main.py logic)
            summaries[cid]["summary"] = translated_text
            
            logger.info(f"🔄 Updated summaries dict")
        
        # Check final result
        logger.info(f"📋 Final summaries dict:")
        for cid, data in summaries.items():
            logger.info(f"   Cluster {cid}: {data['summary'][:50]}...")
        
        # Verify translation worked
        final_summary = summaries[0]["summary"]
        if "ಬಬಲ್" in final_summary or any(ord(c) > 127 for c in final_summary):
            logger.info("✅ Summary successfully updated with Kannada translation!")
            logger.info("✅ Frontend would receive translated content")
            return True
        else:
            logger.error("❌ Summary was not translated - still in English!")
            logger.error(f"❌ Final summary: {final_summary}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING SUMMARY UPDATE LOGIC")
    logger.info("=" * 60)
    
    success = test_summary_update_logic()
    
    logger.info("=" * 60)
    logger.info(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    logger.info("=" * 60)