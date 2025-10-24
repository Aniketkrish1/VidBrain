#!/usr/bin/env python3
"""
Server startup script that excludes temp_processing from file watching
to prevent interruptions during video processing
"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    print("🎥 Starting VidBrain Server")
    print("📁 Temp processing directory excluded from file watching")
    print("⚠️  Server will NOT restart during video processing")
    print("="*60)

    # Run server with comprehensive exclusions to prevent interruptions
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=[
            "temp_processing",
            "temp_processing/*",
            "outputs",
            "outputs/*",
            "*.wav",
            "*.mp4",
            "*.mp3",
            "*.pyc",
            "__pycache__",
            "*.pkl",
            "vector_db.pkl",
            "transcript.srt",
            "temp-audio.m4a"
        ],
        reload_dirs=[".", "static", "utils"]  # Only watch essential directories
    )
