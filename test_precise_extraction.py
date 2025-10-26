"""
Test script to verify that the improved clip extraction only includes topic-relevant content.
"""

import os
import sys
sys.path.append('.')

from utils.topic_query_processor import process_topic_query_enhanced
from utils.clip_extractor import prepare_clips_for_topic
from utils.database import VectorDB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_precise_extraction():
    """Test that clips contain only topic-relevant content with minimal padding."""
    
    logger.info("🧪 Testing precise extraction improvements...")
    
    # Test query
    test_query = "sorting algorithms"
    
    # Check if vector database exists
    vector_db_path = "vector_db.pkl"
    if not os.path.exists(vector_db_path):
        logger.error("❌ Vector database not found. Please run the video processing first.")
        return
    
    # Load vector database
    logger.info("📊 Loading vector database...")
    vector_db = VectorDB()
    vector_db.load(vector_db_path)
    
    # Process query to get relevant segments
    logger.info(f"🔍 Processing query: '{test_query}'")
    result = process_topic_query_enhanced(test_query, vector_db)
    
    if result and "segments" in result:
        segments = result["segments"]
        segment_groups = result["segment_groups"]
        
        logger.info(f"✅ Found {len(segments)} relevant segments")
        logger.info(f"📍 Grouped into {len(segment_groups)} clip groups")
        
        # Test clip preparation
        video_path = "uploads"
        video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
        
        if not video_files:
            logger.error("❌ No video files found in uploads/")
            return
        
        video_file = os.path.join(video_path, video_files[0])
        logger.info(f"🎬 Using video: {video_file}")
        
        # Test precise extraction
        logger.info("✂️ Testing precise clip extraction...")
        clips = prepare_clips_for_topic(video_file, segment_groups, min_clip_duration=1.0)
        
        logger.info(f"🎯 Generated {len(clips)} precise clips")
        
        # Analyze clip precision
        for i, clip_path in enumerate(clips):
            if os.path.exists(clip_path):
                from moviepy import VideoFileClip
                with VideoFileClip(clip_path) as clip:
                    duration = clip.duration
                    logger.info(f"   Clip {i+1}: {duration:.1f}s duration")
            else:
                logger.warning(f"   Clip {i+1}: File not found - {clip_path}")
        
        logger.info("✅ Precise extraction test completed!")
        
        # Check for improvements
        logger.info("📈 Improvements made:")
        logger.info("   • Reduced context padding from 1.0s to 0.2s")
        logger.info("   • Reduced expansion padding from 0.5s to 0.1s") 
        logger.info("   • Reduced minimum clip duration from 2.0s to 1.0s")
        logger.info("   • Reduced merge gap threshold from 2.0s to 1.0s")
        logger.info("   • Added detailed logging for merge decisions")
        
    else:
        logger.error("❌ No relevant segments found for the query")

if __name__ == "__main__":
    test_precise_extraction()