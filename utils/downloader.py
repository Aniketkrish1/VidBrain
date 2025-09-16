from pytubefix import YouTube
import os
from typing import Tuple
from pathlib import Path
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
load_dotenv()

temp_dir = os.getenv("TEMP_DIR")

# absolute path to cookies.txt next to this script
COOKIES_FILE = Path(__file__).parent / "cookies.txt"
if not COOKIES_FILE.exists():
    COOKIES_FILE = None  # fallback: no cookies

def download_and_extract_audio(youtube_url: str) -> Tuple[str, str]:
    """
    Download YouTube video and extract audio.
    """
    print("Downloading from url:", youtube_url)

    try:
        yt = YouTube(youtube_url, token_file=str(COOKIES_FILE) if COOKIES_FILE else None)

        title = yt.title.replace("/", "_").replace("\\", "_").replace(" ", "_")

        video_stream = yt.streams.filter(res='1080p', file_extension='mp4').first()
        if not video_stream:
            video_stream = yt.streams.filter(only_video=True, file_extension='mp4') \
                                     .order_by('resolution').desc().first()
        if not video_stream:
            raise RuntimeError("Could not find a video stream.")

        video_dir = os.path.join(temp_dir, title)
        os.makedirs(video_dir, exist_ok=True)
        video_stream.download(output_path=video_dir, filename="video.mp4")

        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4') \
                                 .order_by('abr').desc().first()
        if not audio_stream:
            raise RuntimeError("Could not find an audio stream.")

        audio_dir = os.path.join(temp_dir, title)
        os.makedirs(audio_dir, exist_ok=True)
        audio_stream.download(output_path=audio_dir, filename="audio.mp3")

        video_path = os.path.join(video_dir, "video.mp4")
        audio_path = os.path.join(audio_dir, "audio.mp3")

        print("Video & audio downloaded:", video_path, audio_path)
        return video_path, audio_path

    except Exception as e:
        print("Error downloading/extracting:", e)
        raise
