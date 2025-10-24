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


def expand_clip_duration(start: float, end: float, min_duration: float = 2.0, 
                        max_duration: float = 30.0, video_duration: float = None) -> Tuple[float, float]:
    """
    Ensure clip has reasonable duration and add padding.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        min_duration: Minimum clip duration
        max_duration: Maximum clip duration
        video_duration: Total video duration for boundary checking
    
    Returns:
        Adjusted (start, end) tuple
    """
    duration = end - start
    
    # If clip is too short, expand it
    if duration < min_duration:
        expansion = (min_duration - duration) / 2
        start = max(0, start - expansion)
        end = end + expansion
    
    # Add small padding for context
    padding = 0.5
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
                           gap_threshold: float = 2.0) -> List[Tuple[float, float]]:
    """
    Merge clips that are close together or overlapping.
    
    Args:
        clips: List of (start, end) tuples
        gap_threshold: Maximum gap in seconds to merge clips
    
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
        if current_start <= prev_end + gap_threshold:
            # Merge by extending the end time
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            # Add as new clip
            merged.append((current_start, current_end))
    
    logger.info(f"Merged {len(clips)} clips into {len(merged)} clips")
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
    Prepare video clips for a specific topic.
    
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
    """
    logger.info(f"Preparing clips from {len(segment_groups)} segment groups")
    
    # Get video duration
    try:
        with VideoFileClip(video_path) as video:
            video_duration = video.duration
    except Exception as e:
        logger.error(f"Could not get video duration: {e}")
        video_duration = None
    
    # Extract timestamps from groups
    raw_timestamps = []
    for group in segment_groups:
        if not group:
            continue
        
        start = convert_timestamp_to_seconds(group[0].get("start", 0))
        end = convert_timestamp_to_seconds(group[-1].get("end", start + 1))
        
        # Expand clip duration
        start, end = expand_clip_duration(start, end, min_clip_duration, 
                                         max_duration=30.0, video_duration=video_duration)
        
        raw_timestamps.append((start, end))
    
    # Merge overlapping or nearby clips
    merged_timestamps = merge_overlapping_clips(raw_timestamps, gap_threshold=3.0)
    
    # Extract clips
    clip_paths = extract_clips_from_timestamps(video_path, merged_timestamps, output_dir)
    
    # Calculate total duration
    total_duration = sum(end - start for start, end in merged_timestamps)
    
    result = {
        "clips": clip_paths,
        "timestamps": merged_timestamps,
        "total_duration": total_duration,
        "num_clips": len(clip_paths)
    }
    
    logger.info(f"Prepared {result['num_clips']} clips with total duration {total_duration:.2f}s")
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
