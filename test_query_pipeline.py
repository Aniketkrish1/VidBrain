#!/usr/bin/env python3
# Test the full pipeline with a specific query
import os
import sys
sys.path.append('.')

from main import process_video

# Test with a specific query using existing video
# Use existing video file
video_path = 'outputs/summary_output.mp4'  # Use existing video
query = 'quick sort'  # Specific topic
output_path = 'outputs/test_quick_sort_summary.mp4'

print(f'Testing full pipeline with query: {query}')

try:
    def progress_callback(stage, percent, details):
        print(f'Progress: {stage} - {percent}% - {details}')

    summaries = []
    def summaries_callback(summary_list):
        summaries.extend(summary_list)
        print(f'Received {len(summary_list)} summaries')
        for summary in summary_list:
            print(f'  Summary: {summary.get("summary", "")[:200]}...')

    result = process_video(
        video_path=video_path,
        youtube_url=None,
        query=query,
        output_path=output_path,
        progress_callback=progress_callback,
        summaries_callback=summaries_callback
    )

    print(f'Pipeline completed: {result}')

    if os.path.exists(result):
        size = os.path.getsize(result) / (1024*1024)
        print(f'File size: {size:.1f} MB')

        # Test if it has audio
        from moviepy import VideoFileClip
        clip = VideoFileClip(result)
        has_audio = clip.audio is not None
        print(f'Has audio: {has_audio}')
        if has_audio:
            print(f'Audio duration: {clip.audio.duration:.1f}s')
            print(f'Video duration: {clip.duration:.1f}s')
        clip.close()

        # Show summaries
        print(f'Generated summaries: {len(summaries)}')
        for summary in summaries:
            summary_text = summary.get('summary', '')[:100]
            cluster_id = summary.get('cluster_id', 'unknown')
            print(f'  Cluster {cluster_id}: {summary_text}...')

except Exception as e:
    print(f'Pipeline failed: {e}')
    import traceback
    traceback.print_exc()
