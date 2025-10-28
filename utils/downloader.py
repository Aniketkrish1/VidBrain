from pytubefix import YouTube
import yt_dlp
import os
from typing import Tuple
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# default temp dir: use env var if set, otherwise create ./temp_processing
env_temp = os.getenv("TEMP_DIR")
temp_dir = env_temp if env_temp else os.path.join(os.getcwd(), "temp_processing")

# absolute path to cookies.txt next to this script (optional)
COOKIES_FILE ="./utils/cookies.txt"
if not Path(COOKIES_FILE).exists():
    COOKIES_FILE = None  # fallback: no cookies


def _yt_dlp_fallback(youtube_url: str, out_dir: str) -> Tuple[str, str]:
    """Fallback downloader using yt_dlp (supports cookies.txt). Returns (video_path,audio_path)."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(out_dir, 'video.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
        'quiet': True,
    }
    if COOKIES_FILE:
        ydl_opts['cookiefile'] = str(COOKIES_FILE)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # find downloaded files
    files = os.listdir(out_dir)
    video_files = [f for f in files if f.startswith('video.') and not f.endswith('.temp')]
    if not video_files:
        raise RuntimeError("Video file not found after yt-dlp download")
    video_filename = video_files[0]
    video_path = os.path.join(out_dir, video_filename)

    # find mp3 output (yt-dlp FFmpegExtractAudio outputs video.mp3)
    audio_files = [f for f in files if f.endswith('.mp3')]
    audio_path = os.path.join(out_dir, audio_files[0]) if audio_files else os.path.join(out_dir, 'audio.mp3')
    return video_path, audio_path


def download_and_extract_audio(youtube_url: str) -> Tuple[str, str]:
    """
    Download YouTube video and extract audio. Tries pytubefix first (with `use_po_token` and optional cookies).
    If YouTube blocks as bot, falls back to yt_dlp (which can use cookies.txt).
    Returns: (video_path, audio_path)
    """
    print("Downloading from url:", youtube_url)
    os.makedirs(temp_dir, exist_ok=True)

    # create a sanitized sub-directory per-video
    safe_title = None

    # 1) Try pytubefix with proof-of-token method
    try:
        kwargs = {'use_po_token': True}
        if COOKIES_FILE:
            kwargs['token_file'] = str(COOKIES_FILE)

        yt = YouTube(youtube_url, **kwargs)
        title = yt.title or youtube_url.split('=')[-1]
        safe_title = title.replace('/', '_').replace('\\', '_').replace(' ', '_')

        video_dir = os.path.join(temp_dir, safe_title)
        os.makedirs(video_dir, exist_ok=True)

        # choose best video stream and download
        video_stream = yt.streams.filter(res='1080p', file_extension='mp4').first()
        if not video_stream:
            video_stream = yt.streams.filter(only_video=True, file_extension='mp4').order_by('resolution').desc().first()
        if video_stream:
            video_stream.download(output_path=video_dir, filename='video.mp4')
        else:
            # if no separate video stream, try progressive
            prog = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not prog:
                raise RuntimeError('No downloadable mp4 stream found (pytubefix)')
            prog.download(output_path=video_dir, filename='video.mp4')

        # audio
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if audio_stream:
            audio_stream.download(output_path=video_dir, filename='audio.mp3')
        else:
            # leave extraction to yt-dlp fallback if needed
            pass

        video_path = os.path.join(video_dir, 'video.mp4')
        audio_path = os.path.join(video_dir, 'audio.mp3')

        if not os.path.exists(video_path):
            # fallback to yt-dlp if pytubefix didn't produce the file
            raise RuntimeError('pytubefix download did not produce expected files')

        print('Downloaded via pytubefix:', video_path, audio_path)
        return video_path, audio_path

    except Exception as e:
        msg = str(e).lower()
        print('pytubefix failed:', e)
        # If it's a bot-detection issue or missing files, fallback to yt-dlp
        try:
            # use fallback directory
            fallback_dir = os.path.join(temp_dir, safe_title or 'download')
            os.makedirs(fallback_dir, exist_ok=True)
            vp, ap = _yt_dlp_fallback(youtube_url, fallback_dir)
            print('Downloaded via yt-dlp fallback:', vp, ap)
            return vp, ap
        except Exception as e2:
            print('yt-dlp fallback failed:', e2)
            raise

if __name__ == "__main__":
    # simple test
    test_url = "https://www.youtube.com/watch?v=kp3fCihUXEg"
    video_path, audio_path = download_and_extract_audio(
        test_url
    )
    print("Video saved to:", video_path)
    print("Audio saved to:", audio_path)