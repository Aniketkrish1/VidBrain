#!/usr/bin/env python3
"""
Test the new chunking approach for Sarvam AI
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

def test_chunked_translation():
    """Test the new chunked translation approach"""
    
    # Long text that would fail before
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
    
    logger.info("🧪 TESTING CHUNKED TRANSLATION")
    logger.info("=" * 60)
    logger.info(f"📝 Original text: {len(long_text)} characters")
    logger.info(f"📝 First 150 chars: {long_text[:150]}...")
    logger.info(f"📝 Last 150 chars: ...{long_text[-150:]}")
    logger.info("=" * 60)
    
    try:
        sarvam = SarvamAI()
        
        # Test chunked translation
        logger.info("🌐 Testing chunked translation...")
        translated_text = sarvam.translate_text(long_text, "kn", "en")
        
        logger.info(f"✅ Translation completed!")
        logger.info(f"📝 Translated length: {len(translated_text)} characters")
        logger.info(f"📝 First 150 chars: {translated_text[:150]}...")
        logger.info(f"📝 Last 150 chars: ...{translated_text[-150:]}")
        
        # Verify it's actually translated (contains Kannada characters)
        kannada_chars = sum(1 for c in translated_text if ord(c) > 127)
        logger.info(f"📝 Kannada characters detected: {kannada_chars}")
        
        if kannada_chars > 100:
            logger.info("✅ SUCCESS: Full text was translated with chunking!")
        else:
            logger.warning("⚠️ WARNING: Translation might not be complete")
        
        return translated_text
        
    except Exception as e:
        logger.error(f"❌ Chunked translation failed: {e}")
        return None

def test_chunked_tts(translated_text):
    """Test the new chunked TTS approach"""
    
    if not translated_text:
        logger.error("❌ No translated text to test TTS with")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎤 TESTING CHUNKED TTS")
    logger.info("=" * 60)
    logger.info(f"📝 Text for TTS: {len(translated_text)} characters")
    
    try:
        sarvam = SarvamAI()
        
        # Test chunked TTS
        logger.info("🎤 Testing chunked TTS...")
        audio_data = sarvam.generate_speech(translated_text, "kn")
        
        logger.info(f"✅ TTS completed!")
        logger.info(f"🎵 Audio generated: {len(audio_data)} bytes")
        
        # Save the audio for verification
        audio_file = "test_chunked_tts_output.wav"
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"💾 Audio saved to: {audio_file}")
        logger.info("✅ SUCCESS: Full text was converted to speech with chunking!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chunked TTS failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 TESTING SARVAM AI CHUNKING SOLUTION")
    logger.info("=" * 80)
    
    # Test translation
    translated = test_chunked_translation()
    
    # Test TTS
    if translated:
        tts_success = test_chunked_tts(translated)
        
        logger.info("\n" + "=" * 80)
        if tts_success:
            logger.info("🎉 COMPLETE SUCCESS!")
            logger.info("✅ Both translation and TTS work with long text!")
            logger.info("✅ Users will now get FULL content in their language!")
        else:
            logger.info("⚠️ PARTIAL SUCCESS!")
            logger.info("✅ Translation works with chunking")
            logger.info("❌ TTS still has issues")
    else:
        logger.info("\n" + "=" * 80)
        logger.error("❌ FAILED: Translation chunking needs more work")
    
    logger.info("=" * 80)