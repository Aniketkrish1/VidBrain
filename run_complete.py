#!/usr/bin/env python3
"""
Complete standalone video processing script - no server interruptions
"""
import os
import sys
from pathlib import Path
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

def main():
    print("🎬 VidBrain Complete Video Processing")
    print("="*50)

    # Check if we have all components
    temp_dir = 'temp_processing'
    video_path = os.path.join(temp_dir, '10_Sorting_Algorithms_Easily_Explained', 'video.mp4')
    voiceovers = {}

    if not os.path.exists(video_path):
        print("❌ Original video not found")
        return 1

    # Collect voiceovers
    for filename in os.listdir(temp_dir):
        if filename.startswith('voiceover_') and filename.endswith('.wav'):
            cluster_id = int(filename.split('_')[1].split('.')[0])
            voiceovers[cluster_id] = os.path.join(temp_dir, filename)

    if not voiceovers:
        print("❌ No voiceovers found")
        return 1

    print(f"✅ Found original video: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    print(f"✅ Found {len(voiceovers)} voiceovers")

    # Load vector DB for segments
    if not os.path.exists('vector_db.pkl'):
        print("❌ Vector DB not found")
        return 1

    with open('vector_db.pkl', 'rb') as f:
        import pickle
        db_data = pickle.load(f)

    segments = db_data['metadata']
    print(f"📊 Vector DB contains {len(segments)} segments")

    # Create clusters based on voiceovers
    cluster_size = len(segments) // len(voiceovers)
    topic_clusters = {}

    for cid in sorted(voiceovers.keys()):
        start_idx = cid * cluster_size
        end_idx = min((cid + 1) * cluster_size, len(segments))
        cluster_segments = segments[start_idx:end_idx]
        topic_clusters[cid] = cluster_segments
        print(f"📝 Cluster {cid}: {len(cluster_segments)} segments")

    # Import and run video assembly
    try:
        from utils.assmble_video import assemble_video

        output_path = os.path.join('outputs', 'summary_output.mp4')
        os.makedirs('outputs', exist_ok=True)

        print("🎬 Assembling final video...")
        final_path = assemble_video(video_path, topic_clusters, voiceovers, output_path)

        print("\n✅ SUCCESS!")
        print(f"📹 Final video generated: {final_path}")
        print(f"📂 File size: {os.path.getsize(final_path) / (1024*1024):.1f} MB")
        print(f"📁 Location: {os.path.abspath(final_path)}")

        # Show download info
        print("\n🌐 For web access:")
        print(f"   File: {os.path.basename(final_path)}")
        print(f"   Path: {final_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Video assembly failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
