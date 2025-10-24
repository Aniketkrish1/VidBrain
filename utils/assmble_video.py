# utils/assmble_video.py
# Purpose: Assemble the final summary video by stitching topic-based subclips
# from the original video and overlaying corresponding TTS voiceovers. Handles
# missing voiceovers gracefully, ensures resources are closed, and writes the
# output using safe defaults compatible with MoviePy 2.x.
import os
import gc
import time
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
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

            # Ensure voiceover exists (skip instead of failing hard)
            voiceover_path = voiceover_paths.get(cluster_id)
            if not voiceover_path or not os.path.exists(voiceover_path):
                logger.warning(f"Missing voiceover for cluster {cluster_id}: {voiceover_path} — skipping this cluster")
                continue

            logger.info(f"Processing cluster {cluster_id} -> {start_time:.2f}-{end_time:.2f}")

            try:
                # Extract video segment (MoviePy 2.x uses subclipped)
                video_segment = original_video.subclipped(start_time, end_time)

                # Load voiceover audio
                voiceover_audio = AudioFileClip(voiceover_path)

                # Adjust video segment duration to match voiceover
                if voiceover_audio.duration < video_segment.duration:
                    video_segment = video_segment.subclipped(0, voiceover_audio.duration)
                # If voiceover is longer, just use the original segment length

                # Store voiceover info for later use
                video_segment._voiceover_path = voiceover_path
                video_segment._voiceover_audio = voiceover_audio

                # Create video segment WITHOUT original audio (important for voiceover replacement)
                video_only_segment = video_segment.without_audio()
                video_segments.append(video_only_segment)

                logger.info(f"✅ Prepared segment for cluster {cluster_id} (original audio removed, voiceover ready)")

                # Don't close audio yet - we'll need it later

            except Exception as e:
                logger.error(f"Failed to process cluster {cluster_id}: {e}")
                # Try to continue with original video segment but without audio
                try:
                    video_segment = original_video.subclipped(start_time, end_time)
                    video_only_segment = video_segment.without_audio()
                    video_only_segment._voiceover_path = voiceover_path
                    video_segments.append(video_only_segment)
                    logger.warning(f"Added cluster {cluster_id} without original audio due to processing error")
                except Exception as e2:
                    logger.error(f"Completely failed to process cluster {cluster_id}: {e2}")
                    continue

        # Defensive check
        if not video_segments:
            raise ValueError("No video segments were created. Check topic_clusters and voiceover_paths.")

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Concatenate video segments (without audio to avoid MoviePy issues)
        logger.info(f"Concatenating {len(video_segments)} video segments...")
        final_video = concatenate_videoclips(video_segments, method="chain")
        logger.info("✅ Video concatenation successful")

        # Check if we have voiceover audio to add
        has_voiceovers = any(hasattr(seg, '_voiceover_path') for seg in video_segments)
        if has_voiceovers:
            logger.info("Adding voiceover audio using ffmpeg...")

            # Use ffmpeg to combine video and voiceover (more reliable than MoviePy audio)
            logger.info("Combining video and voiceover using ffmpeg...")

            # Write video temporarily without audio
            temp_video = os.path.join(out_dir or ".", "temp_video_no_audio.mp4")
            final_video.write_videofile(temp_video, codec='libx264', fps=24)

            # Collect and concatenate all voiceover audio files
            voiceover_paths = []
            for seg in video_segments:
                if hasattr(seg, '_voiceover_path'):
                    voiceover_paths.append(getattr(seg, '_voiceover_path'))

            if voiceover_paths:
                try:
                    import subprocess

                    if len(voiceover_paths) == 1:
                        # Single voiceover file - direct mux
                        logger.info("Single voiceover file - direct mux with ffmpeg...")
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", temp_video,
                            "-i", voiceover_paths[0],
                            "-c:v", "copy", "-c:a", "aac",
                            "-shortest",  # End when shortest stream ends
                            output_path
                        ], check=True, capture_output=True)
                    else:
                        # Multiple voiceover files - concatenate first
                        logger.info(f"Multiple voiceover files ({len(voiceover_paths)}) - concatenating...")
                        audio_concat_file = os.path.join(out_dir or ".", "audio_concat.txt")
                        with open(audio_concat_file, 'w') as f:
                            for audio_file in voiceover_paths:
                                f.write(f"file '{audio_file}'\n")

                        temp_combined_audio = os.path.join(out_dir or ".", "temp_combined_audio.m4a")

                        # Concatenate audio files
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", audio_concat_file, "-c:a", "aac", "-b:a", "128k", temp_combined_audio
                        ], check=True, capture_output=True)

                        # Mux video and combined audio
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", temp_video,
                            "-i", temp_combined_audio,
                            "-c:v", "copy", "-c:a", "aac",
                            "-shortest",
                            output_path
                        ], check=True, capture_output=True)

                        # Clean up audio concat file
                        try:
                            if os.path.exists(temp_combined_audio):
                                os.remove(temp_combined_audio)
                        except:
                            pass

                        try:
                            if os.path.exists(audio_concat_file):
                                os.remove(audio_concat_file)
                        except:
                            pass

                    logger.info("✅ Video and voiceover combined successfully")

                    # Clean up temp video
                    try:
                        if os.path.exists(temp_video):
                            os.remove(temp_video)
                    except:
                        pass

                    return output_path

                except subprocess.CalledProcessError as e:
                    logger.warning(f"FFmpeg processing failed: {e}")
                    logger.info("Falling back to video without audio...")
                    # Just copy the video without audio
                    import shutil
                    shutil.copy2(temp_video, output_path)
                    logger.info("Video created with fallback method")

                    # Clean up temp files
                    try:
                        if os.path.exists(temp_video):
                            os.remove(temp_video)
                    except:
                        pass

                finally:
                    # Clean up video segments
                    for seg in video_segments:
                        try:
                            seg.close()
                        except:
                            pass

            else:
                logger.warning("No voiceover audio files found, writing video without custom audio")

        # Write video without custom audio (fallback)
        logger.info("Writing video without custom audio...")
        final_video.write_videofile(output_path, codec='libx264', fps=24)
        logger.info("✅ Video written successfully")
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
