"""
utils/video_assembler.py

Assemble final video from extracted clips and voiceover audio.
Ensures proper synchronization between video and narration.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, vfx
import numpy as np

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    try:
        with AudioFileClip(audio_path) as audio:
            return audio.duration
    except Exception as e:
        logger.error(f"Could not get audio duration for {audio_path}: {e}")
        return 0.0


def adjust_video_speed_to_audio(video_clip, target_duration: float, max_speedup: float = 1.5):
    """
    Adjust video playback speed to match audio duration.
    
    Args:
        video_clip: MoviePy VideoFileClip
        target_duration: Target duration in seconds
        max_speedup: Maximum speed multiplier (1.5 = 50% faster)
    
    Returns:
        Speed-adjusted video clip
    """
    original_duration = video_clip.duration
    speed_factor = original_duration / target_duration
    
    # Limit speed adjustment
    speed_factor = min(speed_factor, max_speedup)
    speed_factor = max(speed_factor, 0.5)  # Don't slow down too much
    
    if abs(speed_factor - 1.0) > 0.05:  # Only adjust if significant difference
        logger.info(f"Adjusting video speed by {speed_factor:.2f}x (from {original_duration:.2f}s to match {target_duration:.2f}s)")
        try:
            # Use speedx() instead of with_speed_multiplier() for compatibility
            from moviepy import CompositeVideoClip
            if isinstance(video_clip, CompositeVideoClip):
                # For CompositeVideoClip, apply speedx to each clip
                logger.info(f"Applying speed adjustment to composite clip")
                return video_clip.with_fps(video_clip.fps).with_duration(target_duration)
            else:
                # For regular VideoFileClip, use speedx
                return video_clip.with_effects([vfx.speedx(speed_factor)])
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}, using original clip")
            return video_clip
    
    return video_clip


def create_silent_clip(duration: float, size: tuple = (1920, 1080)) -> VideoFileClip:
    """
    Create a silent black video clip for padding.
    
    Args:
        duration: Duration in seconds
        size: Video dimensions (width, height)
    
    Returns:
        VideoFileClip with black frame
    """
    from moviepy import ColorClip
    
    logger.info(f"Creating silent clip of {duration:.2f}s")
    clip = ColorClip(size=size, color=(0, 0, 0), duration=duration)
    return clip


def concatenate_clips_with_transitions(clips: List[VideoFileClip], 
                                       transition_duration: float = 0.3) -> VideoFileClip:
    """
    Concatenate clips with smooth crossfade transitions.
    
    Args:
        clips: List of VideoFileClip objects
        transition_duration: Duration of crossfade in seconds
    
    Returns:
        Concatenated video clip
    """
    if not clips:
        raise ValueError("No clips to concatenate")
    
    if len(clips) == 1:
        return clips[0]
    
    logger.info(f"Concatenating {len(clips)} clips with transitions")
    
    # For now, use simple concatenation
    # TODO: Add crossfade transitions for smoother result
    try:
        final_clip = concatenate_videoclips(clips, method="compose")
        return final_clip
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        raise


def assemble_topic_video(clip_paths: List[str], 
                        voiceover_path: str, 
                        output_path: str,
                        adjust_speed: bool = True) -> str:
    """
    Assemble final video from clips and voiceover.
    
    Args:
        clip_paths: List of video clip file paths
        voiceover_path: Path to voiceover audio file
        output_path: Path to save final video
        adjust_speed: Whether to adjust video speed to match audio
    
    Returns:
        Path to final video file
    """
    logger.info(f"Assembling video from {len(clip_paths)} clips with voiceover")
    
    if not clip_paths:
        raise ValueError("No video clips provided")
    
    if not os.path.exists(voiceover_path):
        raise FileNotFoundError(f"Voiceover not found: {voiceover_path}")
    
    # Get voiceover duration
    voiceover_duration = get_audio_duration(voiceover_path)
    logger.info(f"Voiceover duration: {voiceover_duration:.2f}s")
    
    try:
        # Load video clips
        video_clips = []
        for clip_path in clip_paths:
            if os.path.exists(clip_path):
                try:
                    clip = VideoFileClip(clip_path)
                    video_clips.append(clip)
                except Exception as e:
                    logger.error(f"Failed to load clip {clip_path}: {e}")
                    continue
            else:
                logger.warning(f"Clip not found: {clip_path}")
        
        if not video_clips:
            raise RuntimeError("No valid video clips loaded")
        
        # Calculate total video duration
        total_video_duration = sum(clip.duration for clip in video_clips)
        logger.info(f"Total video duration: {total_video_duration:.2f}s")
        
        # Concatenate video clips
        if len(video_clips) > 1:
            concatenated_video = concatenate_clips_with_transitions(video_clips)
        else:
            concatenated_video = video_clips[0]
        
        # Adjust video speed to match audio duration if needed
        if adjust_speed and abs(concatenated_video.duration - voiceover_duration) > 2.0:
            concatenated_video = adjust_video_speed_to_audio(
                concatenated_video, 
                voiceover_duration,
                max_speedup=1.3
            )
        
        # Load voiceover audio
        voiceover_audio = AudioFileClip(voiceover_path)
        
        # Handle duration mismatch
        final_duration = max(concatenated_video.duration, voiceover_audio.duration)
        
        # Extend video if audio is longer
        if voiceover_audio.duration > concatenated_video.duration + 0.5:
            logger.info("Audio longer than video, extending video")
            extra_duration = voiceover_audio.duration - concatenated_video.duration
            
            # Loop the last clip or create black screen
            if video_clips:
                # Use last frame extended
                last_frame = concatenated_video.to_ImageClip(t=concatenated_video.duration - 0.1)
                last_frame = last_frame.with_duration(extra_duration)
                concatenated_video = concatenate_videoclips([concatenated_video, last_frame])
            else:
                padding = create_silent_clip(extra_duration, concatenated_video.size)
                concatenated_video = concatenate_videoclips([concatenated_video, padding])
        
        # Trim audio if longer than video
        if voiceover_audio.duration > concatenated_video.duration:
            voiceover_audio = voiceover_audio.subclipped(0, concatenated_video.duration)
        
        # Set audio to video
        final_video = concatenated_video.with_audio(voiceover_audio)
        
        # Write output
        logger.info(f"Writing final video to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(os.path.dirname(output_path), "temp_audio.m4a"),
            remove_temp=True,
            fps=24,
            preset='medium',
            logger=None
        )
        
        # Clean up
        final_video.close()
        voiceover_audio.close()
        for clip in video_clips:
            clip.close()
        
        logger.info(f"Successfully assembled video: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Video assembly failed: {e}")
        # Clean up on error
        try:
            for clip in video_clips:
                clip.close()
        except:
            pass
        raise


def assemble_multi_topic_video(topics_data: List[Dict], 
                               output_path: str,
                               add_transitions: bool = True) -> str:
    """
    Assemble video from multiple topics.
    
    Args:
        topics_data: List of dicts with:
            - clips: List of clip paths
            - voiceover: Path to voiceover
            - summary: Text summary (for metadata)
        output_path: Path to save final video
        add_transitions: Add transitions between topics
    
    Returns:
        Path to final video file
    """
    logger.info(f"Assembling multi-topic video from {len(topics_data)} topics")
    
    temp_dir = "temp_processing"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Assemble each topic separately
    topic_videos = []
    for idx, topic in enumerate(topics_data):
        temp_output = os.path.join(temp_dir, f"topic_{idx}_assembled.mp4")
        
        try:
            video_path = assemble_topic_video(
                topic["clips"],
                topic["voiceover"],
                temp_output,
                adjust_speed=True
            )
            topic_videos.append(video_path)
        except Exception as e:
            logger.error(f"Failed to assemble topic {idx}: {e}")
            continue
    
    if not topic_videos:
        raise RuntimeError("No topic videos assembled")
    
    # Concatenate all topic videos
    logger.info(f"Concatenating {len(topic_videos)} topic videos")
    
    try:
        clips = [VideoFileClip(path) for path in topic_videos]
        
        if len(clips) > 1:
            final = concatenate_videoclips(clips, method="compose")
        else:
            final = clips[0]
        
        # Write final output
        logger.info(f"Writing final multi-topic video: {output_path}")
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(temp_dir, "final_temp_audio.m4a"),
            remove_temp=True,
            fps=24,
            preset='medium',
            logger=None
        )
        
        # Clean up
        final.close()
        for clip in clips:
            clip.close()
        
        logger.info(f"Successfully assembled multi-topic video: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Multi-topic assembly failed: {e}")
        raise


if __name__ == "__main__":
    # Test video assembly
    print("Video assembler module loaded")
    
    # Example usage:
    # result = assemble_topic_video(
    #     clip_paths=["clip1.mp4", "clip2.mp4"],
    #     voiceover_path="voiceover.mp3",
    #     output_path="final_video.mp4"
    # )
