# import yt_dlp
from pytubefix import YouTube
import os
from typing import Tuple
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
load_dotenv()

# log_level = os.getenv("LOG_LEVEL")
temp_dir = os.getenv("TEMP_DIR")
# logging.basicConfig(
#     level=log_level,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

def download_and_extract_audio(youtube_url: str) -> Tuple[str, str]:
    """
    Download YouTube video and extract audio.
    
    Args:
        youtube_url: URL of the YouTube video
        
    Returns:
        Tuple of (video_path, audio_path)
    """
    # logger.info(f"Downloading video from: {youtube_url}")
    print("Downloading from url :",youtube_url)
    
    
    try:

        yt = YouTube(youtube_url)

        title = yt.title.replace("/", "_").replace("\\", "_").replace(" ","_")

        video_stream = yt.streams.filter(res='1080p', file_extension='mp4').first()
        if not video_stream:
            # Fallback to the highest available video-only stream if 1080p is not an option
            video_stream = yt.streams.filter(only_video=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not video_stream:
            print("Could not find a video-only stream.")
            exit()
        video_path = os.path.join(temp_dir,title)
        video_stream.download(output_path=video_path,filename="video.mp4")

        print("Video Downloaded successfully")
        

        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()

        if not audio_stream:
            print("Could not find an audio-only stream.")
            exit()
        audio_path=os.path.join(temp_dir,title)
        audio_stream.download(output_path=audio_path,filename="audio.mp3")

        audio_path=os.path.join(audio_path,"audio.mp3")
        video_path=os.path.join(video_path,"video.mp4")

        print("Audio Extracted successfully")
        print(video_path,audio_path)
        return video_path, audio_path
        
    except Exception as e:
        # logger.error(f"Error downloading/extracting: {e}")
        print(e)
        raise


# download_and_extract_audio("https://www.youtube.com/watch?v=EFg3u_E6eHU")
