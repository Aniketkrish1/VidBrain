import yt_dlp
import os
from typing import Tuple
from pathlib import Path
from moviepy import VideoFileClip
from dotenv import load_dotenv
load_dotenv()

temp_dir = os.getenv("TEMP_DIR", "temp_processing")

def download_and_extract_audio(youtube_url: str) -> Tuple[str, str]:
    """
    Download YouTube video and extract audio using yt-dlp.
    """
    print("Downloading from url:", youtube_url)

    try:
        # Extract video info first to get title
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            title = info['title'].replace("/", "_").replace("\\", "_").replace(" ", "_")

        # Configure yt-dlp options for download
        video_dir = os.path.join(temp_dir, title)
        os.makedirs(video_dir, exist_ok=True)

        ydl_opts = {
            'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
            'format': 'best[height<=720]',  # Best single format up to 720p (more compatible)
            'noplaylist': True,
            'quiet': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
        }

        # Download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Find the downloaded video file
        video_files = [f for f in os.listdir(video_dir) if f.startswith('video.') and f.endswith(('.mp4', '.webm', '.mkv'))]
        if not video_files:
            raise RuntimeError("Video download failed - no video file found")

        video_path = os.path.join(video_dir, video_files[0])

        # Extract audio from video using moviepy
        audio_path = os.path.join(video_dir, "audio.mp3")
        if not os.path.exists(audio_path):
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path)
            video_clip.close()

        print("Video & audio downloaded:", video_path, audio_path)
        return video_path, audio_path

    except Exception as e:
        print("Error downloading/extracting:", e)
        raise
