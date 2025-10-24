import yt_dlp
import os
import sys
import platform
from typing import Tuple, Optional
from pathlib import Path
from moviepy import VideoFileClip
from dotenv import load_dotenv
import logging

load_dotenv()

temp_dir = os.getenv("TEMP_DIR", "temp_processing")
logger = logging.getLogger(__name__)


def get_available_browsers():
    """
    Get list of browsers available on the current platform.
    
    Returns:
        List of browser names that are supported on this OS
    """
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        browsers = ['safari', 'chrome', 'firefox', 'edge', 'brave', 'opera']
    elif system == 'windows':
        browsers = ['chrome', 'firefox', 'edge', 'brave', 'opera']
    elif system == 'linux':
        browsers = ['chrome', 'chromium', 'firefox', 'brave', 'opera']
    else:
        # Fallback for unknown systems
        browsers = ['chrome', 'firefox']
    
    logger.info(f"Platform: {system}, Available browsers: {browsers}")
    return browsers


def get_ydl_options(video_dir: str, use_cookies: bool = True) -> dict:
    """
    Get yt-dlp options with enhanced bot evasion.
    
    Args:
        video_dir: Directory to save video
        use_cookies: Whether to use browser cookies
    
    Returns:
        Dictionary of yt-dlp options
    """
    ydl_opts = {
        'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
        'format': 'best[height<=720]',  # Best single format up to 720p
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        # Enhanced headers to avoid bot detection
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        # Additional options for stability
        'retries': 10,
        'fragment_retries': 10,
        'skip_unavailable_fragments': True,
        'ignoreerrors': False,
        'nocheckcertificate': True,
    }
    
    # Try to use browser cookies if available
    if use_cookies:
        # Get platform-specific browsers
        browsers = get_available_browsers()
        
        # Try each browser until one works
        for browser in browsers:
            try:
                ydl_opts['cookiesfrombrowser'] = (browser,)
                logger.info(f"Will attempt to use {browser} cookies")
                return ydl_opts
            except Exception as e:
                logger.debug(f"Could not configure {browser} cookies: {e}")
                continue
        
        # If no browser cookies could be configured, remove the option
        if 'cookiesfrombrowser' in ydl_opts:
            del ydl_opts['cookiesfrombrowser']
            logger.warning("No browser cookies available, will try without cookies")
            logger.warning("For better results: close all browsers, then try again")
    
    return ydl_opts


def download_and_extract_audio(youtube_url: str, use_cookies: bool = True) -> Tuple[str, str]:
    """
    Download YouTube video and extract audio using yt-dlp.
    
    Args:
        youtube_url: YouTube video URL
        use_cookies: Whether to use browser cookies for authentication
    
    Returns:
        Tuple of (video_path, audio_path)
    """
    logger.info(f"Downloading from URL: {youtube_url}")

    try:
        # Extract video info first to get title
        info_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        # Try to add cookies for info extraction
        if use_cookies:
            browsers = get_available_browsers()
            cookie_success = False
            
            for browser in browsers:
                try:
                    info_opts['cookiesfrombrowser'] = (browser,)
                    # Test if we can actually use this browser's cookies
                    with yt_dlp.YoutubeDL(info_opts) as test_ydl:
                        info = test_ydl.extract_info(youtube_url, download=False)
                    logger.info(f"Successfully using {browser} cookies")
                    cookie_success = True
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'cookie' in error_msg or 'locked' in error_msg or 'could not copy' in error_msg:
                        logger.warning(f"{browser} cookies locked or unavailable: {str(e)[:100]}")
                        # Remove cookies option and try next browser
                        if 'cookiesfrombrowser' in info_opts:
                            del info_opts['cookiesfrombrowser']
                        continue
                    else:
                        # Some other error, still try next browser
                        logger.debug(f"Error with {browser}: {str(e)[:100]}")
                        continue
            
            if not cookie_success:
                logger.warning("Could not use any browser cookies")
                logger.warning("💡 TIP: Close Chrome/Firefox/Edge and try again")
                logger.warning("Attempting download without cookies...")
                # Remove cookies option
                if 'cookiesfrombrowser' in info_opts:
                    del info_opts['cookiesfrombrowser']
                # Try to get info without cookies
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
        else:
            # No cookies requested
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
        
        title = info.get('title', 'video').replace("/", "_").replace("\\", "_").replace(" ", "_")
        title = "".join(c for c in title if c.isalnum() or c in "_-")[:100]  # Sanitize filename

        # Configure yt-dlp options for download
        video_dir = os.path.join(temp_dir, title)
        os.makedirs(video_dir, exist_ok=True)

        ydl_opts = get_ydl_options(video_dir, use_cookies=use_cookies)

        # Download video with retry logic for cookie issues
        logger.info("Starting video download...")
        download_success = False
        last_error = None
        
        # If cookies are enabled, try each browser
        if use_cookies and 'cookiesfrombrowser' in ydl_opts:
            browsers = get_available_browsers()
            
            for browser in browsers:
                try:
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                    logger.info(f"Attempting download with {browser} cookies...")
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                    
                    download_success = True
                    logger.info(f"Download successful with {browser} cookies!")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'cookie' in error_msg or 'locked' in error_msg or 'could not copy' in error_msg:
                        logger.warning(f"{browser} cookies unavailable: {str(e)[:80]}")
                        last_error = e
                        continue
                    else:
                        # Different error, re-raise
                        raise
            
            # If all browsers failed, try without cookies
            if not download_success:
                logger.warning("All browser cookies failed, trying without cookies...")
                if 'cookiesfrombrowser' in ydl_opts:
                    del ydl_opts['cookiesfrombrowser']
                
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                    download_success = True
                    logger.info("Download successful without cookies!")
                except Exception as e:
                    logger.error(f"Download failed even without cookies: {e}")
                    raise
        else:
            # No cookies or cookies disabled
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            download_success = True

        # Find the downloaded video file
        video_files = [f for f in os.listdir(video_dir) if f.startswith('video.') and f.endswith(('.mp4', '.webm', '.mkv'))]
        if not video_files:
            raise RuntimeError("Video download failed - no video file found")

        video_path = os.path.join(video_dir, video_files[0])
        logger.info(f"Video downloaded: {video_path}")

        # Extract audio from video using moviepy
        audio_path = os.path.join(video_dir, "audio.mp3")
        if not os.path.exists(audio_path):
            logger.info("Extracting audio from video...")
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()

        logger.info(f"Video & audio ready: {video_path}, {audio_path}")
        return video_path, audio_path

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
            logger.error("YouTube bot detection triggered.")
            logger.error("💡 Solutions:")
            logger.error("1. Close ALL browser windows (Chrome, Firefox, Edge)")
            logger.error("2. Open Chrome, sign into YouTube")
            logger.error("3. Try download again")
            logger.error("4. If still fails, try a different video")
        elif "cookie" in error_msg.lower() or "locked" in error_msg.lower():
            logger.error("Browser cookies locked or unavailable.")
            logger.error("💡 Solution: Close all browser windows and try again")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
