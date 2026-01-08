import os
import gc
import time
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def assemble_video(
    video_path: str,
    topic_clusters: Dict[int, Dict],
    voiceover_paths: Dict[int, str],
    output_path: str
) -> str:
    """
    Assemble final summary video with topic clips and voiceovers.
    More robust: validates input, logs details, ensures resources are closed.
    """
    logger.info("Assembling final summary video")
    original_video = None
    video_segments = []
    audio_clips = []  # Track audio clips for cleanup
    final_video = None

    try:
        # Load original video
        logger.debug(f"Loading original video: {video_path}")
        original_video = VideoFileClip(video_path)

        if not topic_clusters:
            raise ValueError("topic_clusters is empty. Nothing to assemble.")

        logger.info(f"Found {len(topic_clusters)} clusters: {list(topic_clusters.keys())}")
        sorted_clusters = sorted(topic_clusters.items(), key=lambda item: item[1]['start'])
        
        for cluster_id,data in sorted_clusters:
            start_time = data.get("start")
            end_time = data.get("end")
            if start_time is None or end_time is None:
                logger.warning(f"Cluster {cluster_id} missing start/end timestamps. Skipping.")
                continue

            # Ensure voiceover exists
            voiceover_path = voiceover_paths.get(cluster_id)
            if not voiceover_path or not os.path.exists(voiceover_path):
                raise FileNotFoundError(f"Missing voiceover for cluster {cluster_id}: {voiceover_path}")

            # Load voiceover audio first to get its duration
            voiceover_audio = AudioFileClip(voiceover_path)
            audio_clips.append(voiceover_audio)  # Track for cleanup later
            
            voiceover_duration = voiceover_audio.duration
            original_segment_duration = end_time - start_time
            
            logger.info(f"Processing cluster {cluster_id} -> Original: {start_time:.2f}-{end_time:.2f} ({original_segment_duration:.2f}s), Voiceover: {voiceover_duration:.2f}s")

            # Extract video segment ONLY for the voiceover duration (condense the output!)
            # This ensures the final video is shorter than the original
            video_extract_duration = min(voiceover_duration, original_segment_duration)
            video_segment = original_video.subclip(start_time, start_time + video_extract_duration)

            # Only loop if voiceover is longer than available video (rare case)
            if voiceover_duration > video_extract_duration:
                video_segment = video_segment.loop(duration=voiceover_duration)

            # Replace audio with voiceover (apply short fades)
            audio_clip = voiceover_audio.audio_fadein(0.02).audio_fadeout(0.02)
            video_segment = video_segment.set_audio(audio_clip)

            # Keep segment for concatenation
            video_segments.append(video_segment)

        # Defensive check
        if not video_segments:
            raise ValueError("No video segments were created. Check topic_clusters and voiceover_paths.")

        # Concatenate and write output
        final_video = concatenate_videoclips(video_segments, method="compose")
        logger.info(f"Writing final video to: {output_path}")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=False,
            verbose=False,
            logger=None
        )

        logger.info(f"Final video saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error during video assembly: {e}")
        raise

    finally:
        # Ensure all resources are closed to release file handles
        try:
            if final_video is not None:
                final_video.close()
        except Exception:
            pass

        for seg in video_segments:
            try:
                seg.close()
            except Exception:
                pass
        
        # Close audio clips
        for audio in audio_clips:
            try:
                audio.close()
            except Exception:
                pass

        try:
            if original_video is not None:
                original_video.close()
        except Exception:
            pass

        # give OS a moment to release handles on Windows
        gc.collect()
        time.sleep(0.2)
