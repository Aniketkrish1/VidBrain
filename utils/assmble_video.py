import os
import gc
import time
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)




def assemble_video(
    video_path: str,
    topic_clusters: Dict[int, List[Dict]],
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
    final_video = None

    try:
        # Load original video
        logger.debug(f"Loading original video: {video_path}")
        original_video = VideoFileClip(video_path)

        if not topic_clusters:
            raise ValueError("topic_clusters is empty. Nothing to assemble.")

        logger.info(f"Found {len(topic_clusters)} clusters: {list(topic_clusters.keys())}")

        for cluster_id in sorted(topic_clusters.keys()):
            sentences = topic_clusters[cluster_id]
            if not sentences:
                logger.warning(f"Cluster {cluster_id} is empty. Skipping.")
                continue

            # Ensure timestamps exist
            start_time = sentences[0].get('start')
            end_time = sentences[-1].get('end')
            if start_time is None or end_time is None:
                logger.warning(f"Cluster {cluster_id} missing start/end timestamps. Skipping.")
                continue

            # Ensure voiceover exists
            voiceover_path = voiceover_paths.get(cluster_id)
            if not voiceover_path or not os.path.exists(voiceover_path):
                raise FileNotFoundError(f"Missing voiceover for cluster {cluster_id}: {voiceover_path}")

            logger.info(f"Processing cluster {cluster_id} -> {start_time:.2f}-{end_time:.2f}")

            # Extract video segment
            video_segment = original_video.subclip(start_time, end_time)

            # Load voiceover audio
            voiceover_audio = AudioFileClip(voiceover_path)

            # Adjust video segment duration to match voiceover
            if voiceover_audio.duration < video_segment.duration:
                video_segment = video_segment.subclip(0, voiceover_audio.duration)
            elif voiceover_audio.duration > video_segment.duration:
                # extend video to match voiceover duration
                video_segment = video_segment.loop(duration=voiceover_audio.duration)

            # Replace audio with voiceover (apply short fades)
            audio_clip = voiceover_audio.audio_fadein(0.02).audio_fadeout(0.02)
            video_segment = video_segment.set_audio(audio_clip)

            # Keep segment for concatenation
            video_segments.append(video_segment)

            # close the original audio file object if possible (audio resources are referenced by clip)
            try:
                voiceover_audio.close()
            except Exception:
                pass

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

        try:
            if original_video is not None:
                original_video.close()
        except Exception:
            pass

        # give OS a moment to release handles on Windows
        gc.collect()
        time.sleep(0.2)
