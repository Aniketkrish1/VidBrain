#!/usr/bin/env python3
"""
Test the actual main.py process_video function with Kannada
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import process_video

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_main_process():
    """Test main.py process_video with Kannada language"""
    
    # Use the uploaded video file
    video_path = "uploads/6a6e3fab-2277-48ec-920d-f620ec0b016c_10 Sorting Algorithms Easily Explained - Coding with Lewis (144p, h264).mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    logger.info(f"🎬 Testing with video: {video_path}")
    logger.info(f"🌐 Language: kn (Kannada)")
    
    try:
        # Process with Kannada language
        result = process_video(
            video_path=video_path,
            youtube_url=None,
            query="sorting algorithms",
            output_path="outputs/test_kannada_output.mp4",
            language="kn"
        )
        
        logger.info(f"✅ Processing completed!")
        logger.info(f"📋 Result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main processing failed: {e}")
        
        # More detailed error info
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING MAIN.PY PROCESS_VIDEO WITH KANNADA")
    logger.info("=" * 60)
    
    success = test_main_process()
    
    logger.info("=" * 60)
    logger.info(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    logger.info("=" * 60)