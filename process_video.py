#!/usr/bin/env python3
"""
Standalone video processing script - runs without server interruptions
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_video

def main():
    print("🎥 VidBrain Standalone Video Processor")
    print("="*50)

    # Get YouTube URL
    youtube_url = input("Enter YouTube URL: ").strip()
    if not youtube_url:
        print("❌ No URL provided")
        return

    # Get query (optional)
    query = input("Enter topic query (optional, press Enter to skip): ").strip()
    if not query:
        query = None

    # Get output filename
    output_name = input("Enter output filename (default: summary_output.mp4): ").strip()
    if not output_name:
        output_name = "summary_output.mp4"

    # Create outputs directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_name

    print(f"📹 Processing: {youtube_url}")
    print(f"🎯 Query: {query or 'Auto-detect topics'}")
    print(f"📁 Output: {output_path}")
    print("\n⏳ Starting video processing... (this may take several minutes)\n")

    try:
        result_path = process_video(
            video_path=None,
            youtube_url=youtube_url,
            query=query,
            output_path=str(output_path)
        )

        print("\n✅ SUCCESS!")
        print(f"📹 Final video saved to: {result_path}")
        print(f"📂 File location: {os.path.abspath(result_path)}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
