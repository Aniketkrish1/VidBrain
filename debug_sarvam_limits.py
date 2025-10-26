#!/usr/bin/env python3
"""
Debug script to check text length limits in Sarvam AI translation and TTS
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

def debug_text_lengths():
    """Debug actual text lengths being processed"""
    
    # Sample long text (similar to what you might get from video processing)
    long_text = """
    Quick sort is a highly efficient, comparison-based sorting algorithm that employs the divide and conquer strategy. 
    It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, 
    according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively. 
    This can be done in-place, requiring small additional amounts of memory to perform the sorting. Quick sort is 
    often faster in practice than other O(n log n) algorithms such as merge sort or heap sort. However, it has a 
    worst-case time complexity of O(n²), which occurs when the pivot is consistently the smallest or largest element. 
    Despite this, its average-case performance is excellent, making it a popular choice for sorting large datasets. 
    The algorithm's efficiency comes from its ability to reduce the problem size significantly with each recursive call, 
    and its cache-friendly access patterns. Modern implementations often use techniques like median-of-three pivot 
    selection and hybrid approaches that switch to insertion sort for small arrays to optimize performance further.
    """.strip()
    
    logger.info("🔍 DEBUGGING TEXT LENGTH PROCESSING")
    logger.info("=" * 60)
    logger.info(f"📝 Original text length: {len(long_text)} characters")
    logger.info(f"📝 Original text word count: {len(long_text.split())} words")
    logger.info("=" * 60)
    
    try:
        sarvam = SarvamAI()
        
        # Test translation
        logger.info("🌐 Testing FULL TEXT translation...")
        translated_text = sarvam.translate_text(long_text, "kn", "en")
        
        logger.info(f"✅ Translation completed!")
        logger.info(f"📝 Translated text length: {len(translated_text)} characters")
        logger.info(f"📝 Translated text word count: {len(translated_text.split())} words")
        logger.info(f"📝 First 200 chars: {translated_text[:200]}...")
        logger.info(f"📝 Last 200 chars: ...{translated_text[-200:]}")
        
        # Check if translation is complete
        if len(translated_text) > len(long_text) * 0.5:  # Reasonable heuristic
            logger.info("✅ Translation appears to be COMPLETE (good length ratio)")
        else:
            logger.warning("⚠️ Translation might be TRUNCATED (short length)")
        
        logger.info("=" * 60)
        
        # Test TTS with full translated text
        logger.info("🎤 Testing TTS with FULL translated text...")
        try:
            audio_data = sarvam.generate_speech(translated_text, "kn")
            logger.info(f"✅ TTS successful with full text: {len(audio_data)} bytes")
        except Exception as tts_e:
            logger.error(f"❌ TTS failed with full text: {tts_e}")
            
            # Try with shorter text
            logger.info("🎤 Testing TTS with SHORTER text...")
            short_text = translated_text[:500]  # First 500 chars
            try:
                audio_data = sarvam.generate_speech(short_text, "kn")
                logger.info(f"✅ TTS successful with short text: {len(audio_data)} bytes")
                logger.warning(f"⚠️ TTS LIMIT FOUND: Full text ({len(translated_text)} chars) fails, but short text ({len(short_text)} chars) works")
            except Exception as short_tts_e:
                logger.error(f"❌ TTS failed even with short text: {short_tts_e}")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def test_different_text_lengths():
    """Test TTS with different text lengths to find the limit"""
    
    base_text = "ಇದು ಒಂದು ಪರೀಕ್ಷೆಯ ವಾಕ್ಯ. "  # "This is a test sentence. "
    sarvam = SarvamAI()
    
    # Test different lengths
    lengths_to_test = [50, 100, 200, 500, 1000, 2000, 3000]
    
    logger.info("🧪 TESTING TTS LENGTH LIMITS")
    logger.info("=" * 60)
    
    for target_length in lengths_to_test:
        # Create text of target length
        repeat_count = (target_length // len(base_text)) + 1
        test_text = (base_text * repeat_count)[:target_length]
        
        logger.info(f"📝 Testing {len(test_text)} characters...")
        
        try:
            audio_data = sarvam.generate_speech(test_text, "kn")
            logger.info(f"   ✅ SUCCESS: {len(audio_data)} bytes audio generated")
        except Exception as e:
            logger.error(f"   ❌ FAILED: {str(e)[:100]}...")
            logger.info(f"   🚨 LIMIT FOUND: TTS fails at ~{len(test_text)} characters")
            break

if __name__ == "__main__":
    logger.info("🔬 SARVAM AI TEXT LENGTH DEBUGGING")
    logger.info("=" * 80)
    
    # Test 1: Debug actual text processing
    debug_text_lengths()
    
    logger.info("\n" + "=" * 80)
    
    # Test 2: Find TTS length limits
    test_different_text_lengths()
    
    logger.info("=" * 80)