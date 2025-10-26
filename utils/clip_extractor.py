"""
utils/clip_extractor.py

Extract and prepare video clips based on topic timestamps.
Aligns video clips with the content being discussed in the summary.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)


def convert_timestamp_to_seconds(timestamp) -> float:
    """
    Convert various timestamp formats to seconds.
    
    Args:
        timestamp: Can be float, int, or string (HH:MM:SS,mmm format)
    
    Returns:
        Float representing seconds
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    if isinstance(timestamp, str):
        # Handle SRT format: HH:MM:SS,mmm
        if ',' in timestamp:
            time_part, ms_part = timestamp.split(',')
            h, m, s = time_part.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000
        # Handle simple float string
        try:
            return float(timestamp)
        except ValueError:
            logger.warning(f"Could not convert timestamp: {timestamp}")
            return 0.0
    
    return 0.0


def expand_clip_duration(start: float, end: float, min_duration: float = 1.0, 
                        max_duration: float = 30.0, video_duration: float = None) -> Tuple[float, float]:
    """
    Ensure clip has reasonable duration with MINIMAL padding to avoid irrelevant content.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        min_duration: Minimum clip duration (reduced to 1.0 second)
        max_duration: Maximum clip duration
        video_duration: Total video duration for boundary checking
    
    Returns:
        Adjusted (start, end) tuple with minimal expansion
    """
    duration = end - start
    
    # Only expand if clip is extremely short (less than 1 second)
    if duration < min_duration:
        expansion = (min_duration - duration) / 2
        start = max(0, start - expansion)
        end = end + expansion
    
    # MINIMAL padding only for smooth cuts - reduced from 0.5 to 0.1 seconds
    padding = 0.1
    start = max(0, start - padding)
    end = end + padding
    
    # Respect video boundaries
    if video_duration:
        start = max(0, start)
        end = min(video_duration, end)
    
    # Enforce maximum duration
    if end - start > max_duration:
        end = start + max_duration
    
    return start, end


def merge_overlapping_clips(clips: List[Tuple[float, float]], 
                           gap_threshold: float = 1.0) -> List[Tuple[float, float]]:
    """
    Merge clips that are close together or overlapping.
    REDUCED gap_threshold to avoid merging unrelated topic segments.
    
    Args:
        clips: List of (start, end) tuples
        gap_threshold: Maximum gap in seconds to merge clips (reduced to 1.0)
    
    Returns:
        List of merged (start, end) tuples
    """
    if not clips:
        return []
    
    # Sort clips by start time
    sorted_clips = sorted(clips, key=lambda x: x[0])
    
    merged = [sorted_clips[0]]
    
    for current_start, current_end in sorted_clips[1:]:
        prev_start, prev_end = merged[-1]
        
        # Check if clips overlap or are close enough to merge
        # STRICTER merging to avoid combining different topic segments
        if current_start <= prev_end + gap_threshold:
            # Merge by extending the end time
            merged[-1] = (prev_start, max(prev_end, current_end))
            logger.info(f"🔗 MERGED clips: ({prev_start:.1f}-{prev_end:.1f}) + ({current_start:.1f}-{current_end:.1f}) = ({prev_start:.1f}-{max(prev_end, current_end):.1f})")
        else:
            # Add as new clip - gap too large, likely different topic
            logger.info(f"📍 SEPARATE clips: Gap of {current_start - prev_end:.1f}s too large, keeping separate")
            merged.append((current_start, current_end))
    
    logger.info(f"Merged {len(clips)} clips into {len(merged)} clips with stricter criteria")
    return merged


def extract_clips_from_timestamps(video_path: str, 
                                  timestamps: List[Tuple[float, float]], 
                                  output_dir: str = "temp_processing") -> List[str]:
    """
    Extract video clips based on timestamps.
    
    Args:
        video_path: Path to source video file
        timestamps: List of (start, end) tuples in seconds
        output_dir: Directory to save extracted clips
    
    Returns:
        List of paths to extracted clip files
    """
    logger.info(f"Extracting {len(timestamps)} clips from video")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    clip_paths = []
    
    try:
        video = VideoFileClip(video_path)
        video_duration = video.duration
        
        for idx, (start, end) in enumerate(timestamps):
            # Ensure timestamps are within video bounds
            start = max(0, float(start))
            end = min(video_duration, float(end))
            
            if start >= end:
                logger.warning(f"Invalid clip {idx}: start={start}, end={end}")
                continue
            
            # Extract clip with retry logic
            clip_path = os.path.join(output_dir, f"clip_{idx:03d}.mp4")
            
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                clip = None
                try:
                    # Create clip
                    clip = video.subclipped(start, end)
                    
                    # Write with explicit cleanup
                    clip.write_videofile(
                        clip_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile=os.path.join(output_dir, f"temp_audio_{idx}.m4a"),
                        remove_temp=True,
                        logger=None,
                        threads=1  # Use single thread to avoid subprocess issues
                    )
                    
                    # Verify the file was created
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                        clip_paths.append(clip_path)
                        logger.info(f"Extracted clip {idx}: {start:.2f}s - {end:.2f}s -> {clip_path}")
                        success = True
                        break  # Success - exit retry loop
                    else:
                        raise Exception("Output file not created or empty")
                    
                except Exception as e:
                    logger.warning(f"Clip {idx} attempt {attempt + 1}/{max_retries} failed: {e}")
                    
                    # Clean up failed attempt
                    if os.path.exists(clip_path):
                        try:
                            os.remove(clip_path)
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        # Delay before retry
                        import time
                        time.sleep(1.5)
                    
                finally:
                    # Always close the clip
                    if clip is not None:
                        try:
                            clip.close()
                        except:
                            pass
            
            if not success:
                logger.error(f"Failed to extract clip {idx} after {max_retries} attempts - skipping")
                continue
        
        video.close()
        
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        raise
    
    logger.info(f"Successfully extracted {len(clip_paths)} clips")
    return clip_paths


def prepare_clips_for_topic(video_path: str, 
                           segment_groups: List[List[Dict]], 
                           output_dir: str = "temp_processing",
                           min_clip_duration: float = 2.0) -> Dict:
    """
    Prepare video clips for a specific topic with enhanced timestamp accuracy.
    
    Args:
        video_path: Path to source video
        segment_groups: List of segment groups with timestamps
        output_dir: Directory for temporary files
        min_clip_duration: Minimum duration for clips
    
    Returns:
        Dictionary with:
        - clips: List of clip file paths
        - timestamps: List of (start, end) tuples
        - total_duration: Total duration of all clips
        - segment_info: Details about each segment
    """
    logger.info(f"🎯 Preparing clips from {len(segment_groups)} topic-relevant segment groups")
    logger.info(f"🔍 VERIFYING: Each segment group will be used for video extraction")
    
    # Get video duration
    try:
        with VideoFileClip(video_path) as video:
            video_duration = video.duration
            logger.info(f"📹 Source video duration: {video_duration:.2f}s")
    except Exception as e:
        logger.error(f"Could not get video duration: {e}")
        video_duration = None
    
    # Extract timestamps from groups with better context
    raw_timestamps = []
    segment_info = []
    invalid_count = 0
    
    for i, group in enumerate(segment_groups):
        if not group:
            logger.warning(f"Empty group {i}, skipping...")
            continue
        
        # Get the full range of this segment group
        start = convert_timestamp_to_seconds(group[0].get("start", 0))
        end = convert_timestamp_to_seconds(group[-1].get("end", start + 1))
        
        # 🎯 CRITICAL: Validate timestamps before using them
        if start == 0 and end == 0:
            logger.error(f"❌ INVALID GROUP {i}: Both start and end are 0!")
            invalid_count += 1
            continue
            
        if start >= end:
            logger.error(f"❌ INVALID GROUP {i}: start ({start}) >= end ({end})")
            invalid_count += 1
            continue
            
        if video_duration and start >= video_duration:
            logger.error(f"❌ INVALID GROUP {i}: start ({start}s) beyond video duration ({video_duration}s)")
            invalid_count += 1
            continue
        
        logger.info(f"✅ VALID GROUP {i}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        
        # MINIMAL context padding to avoid irrelevant content - reduced from 1.0 to 0.2 seconds
        context_padding = 0.2  # Only 0.2 seconds of context before/after
        start_with_context = max(0, start - context_padding)
        end_with_context = end + context_padding
        if video_duration:
            end_with_context = min(video_duration, end_with_context)
        
        # Expand clip duration for minimum requirements with minimal expansion
        final_start, final_end = expand_clip_duration(
            start_with_context, end_with_context, 
            min_duration=1.0,  # Fixed parameter name - was min_clip_duration
            max_duration=30.0, 
            video_duration=video_duration
        )
        
        raw_timestamps.append((final_start, final_end))
        
        # Store segment information
        segment_text = " ".join([seg.get("text", "") for seg in group])
        segment_info.append({
            "group_id": i,
            "original_start": start,
            "original_end": end,
            "final_start": final_start,
            "final_end": final_end,
            "duration": final_end - final_start,
            "text_preview": segment_text[:100] + "..." if len(segment_text) > 100 else segment_text
        })
        
        logger.info(f"   📍 Final timestamps: {start:.1f}s-{end:.1f}s → {final_start:.1f}s-{final_end:.1f}s ({final_end-final_start:.1f}s)")
    
    if invalid_count > 0:
        logger.error(f"❌ FOUND {invalid_count} INVALID TIMESTAMP GROUPS!")
        logger.error(f"❌ This will cause video clips to be extracted from wrong locations!")
    
    if not raw_timestamps:
        logger.error(f"❌ NO VALID TIMESTAMPS for video extraction!")
        raise ValueError("No valid timestamps available for video clip extraction")
    
    # Merge overlapping or nearby clips for smoother transitions
    logger.info(f"🔗 Merging overlapping clips...")
    merged_timestamps = merge_overlapping_clips(raw_timestamps, gap_threshold=2.0)
    logger.info(f"   Merged {len(raw_timestamps)} clips → {len(merged_timestamps)} final clips")
    
    # Extract clips with better naming
    clip_paths = extract_clips_from_timestamps(video_path, merged_timestamps, output_dir)
    
    # Calculate total duration
    total_duration = sum(end - start for start, end in merged_timestamps)
    
    result = {
        "clips": clip_paths,
        "timestamps": merged_timestamps,
        "total_duration": total_duration,
        "num_clips": len(clip_paths),
        "segment_info": segment_info,
        "video_duration": video_duration
    }
    
    logger.info(f"✅ Prepared {result['num_clips']} synchronized clips:")
    logger.info(f"   📊 Total duration: {total_duration:.2f}s")
    logger.info(f"   📹 Coverage: {(total_duration/video_duration*100):.1f}% of source video" if video_duration else "")
    
    return result


def get_clip_durations(clip_paths: List[str]) -> List[float]:
    """
    Get duration of each clip.
    
    Args:
        clip_paths: List of paths to video clips
    
    Returns:
        List of durations in seconds
    """
    durations = []
    
    for clip_path in clip_paths:
        try:
            with VideoFileClip(clip_path) as clip:
                durations.append(clip.duration)
        except Exception as e:
            logger.error(f"Could not get duration for {clip_path}: {e}")
            durations.append(0.0)
    
    return durations


if __name__ == "__main__":
    # Test clip extraction
    print("Clip extractor module loaded")
    
    # Example usage:
    # clips = prepare_clips_for_topic("video.mp4", segment_groups)
    # print(f"Extracted {clips['num_clips']} clips")
