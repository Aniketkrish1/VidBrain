"""
test_youtube_download.py

Quick test to verify YouTube download is working with cookies.
"""

import sys
import os
import platform


def get_available_browsers():
    """Get browsers available on current platform."""
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        return ['safari', 'chrome', 'firefox', 'edge', 'brave']
    elif system == 'windows':
        return ['chrome', 'firefox', 'edge', 'brave']
    elif system == 'linux':
        return ['chrome', 'chromium', 'firefox', 'brave']
    else:
        return ['chrome', 'firefox']


def test_cookie_access():
    """Test if browser cookies are accessible."""
    print("Testing browser cookie access...\n")
    print(f"Platform: {platform.system()}\n")
    
    import yt_dlp
    
    browsers = get_available_browsers()
    working_browsers = []
    
    for browser in browsers:
        try:
            print(f"Testing {browser}... ", end="")
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'cookiesfrombrowser': (browser,),
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Try to extract info (without downloading)
                info = ydl.extract_info('https://www.youtube.com/watch?v=dQw4w9WgXcQ', download=False)
                if info:
                    print(f"✓ Working! (Found: {info.get('title', 'video')})")
                    working_browsers.append(browser)
                else:
                    print("✗ Failed")
        except Exception as e:
            print(f"✗ Not available ({str(e)[:50]}...)")
    
    print(f"\n{'='*60}")
    if working_browsers:
        print(f"✓ SUCCESS! Working browsers: {', '.join(working_browsers)}")
        print(f"VidBrain will automatically use: {working_browsers[0]}")
        return True
    else:
        print("✗ No working browsers found")
        print("Please make sure you're logged into YouTube in Chrome or Firefox")
        return False


def test_download():
    """Test actual video download."""
    print("\n" + "="*60)
    print("Testing actual video download...")
    print("="*60 + "\n")
    
    try:
        from utils.downloader import download_and_extract_audio
        
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"Downloading: {test_url}")
        print("This will take a moment...\n")
        
        video_path, audio_path = download_and_extract_audio(test_url, use_cookies=True)
        
        if os.path.exists(video_path) and os.path.exists(audio_path):
            print(f"\n{'='*60}")
            print("✓ DOWNLOAD SUCCESSFUL!")
            print(f"Video: {video_path}")
            print(f"Audio: {audio_path}")
            print(f"Video size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
            print(f"Audio size: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
            
            # Clean up test files
            try:
                import shutil
                import os.path
                test_dir = os.path.dirname(video_path)
                shutil.rmtree(test_dir)
                print("\nTest files cleaned up")
            except:
                pass
            
            return True
        else:
            print("\n✗ Download failed - files not found")
            return False
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    print("="*60)
    print("YouTube Download Fix - Test Suite")
    print("="*60)
    print()
    
    # Test 1: Cookie access
    cookie_test = test_cookie_access()
    
    if not cookie_test:
        print("\n" + "="*60)
        print("⚠️  RECOMMENDATION")
        print("="*60)
        print("\n1. Open Chrome or Firefox")
        print("2. Go to youtube.com")
        print("3. Sign in to your account")
        print("4. Run this test again")
        print("\nFor more help, see: YOUTUBE_FIX_GUIDE.md")
        return False
    
    # Test 2: Actual download
    print("\n" + "="*60)
    print("Ready to test actual download?")
    print("="*60)
    response = input("\nThis will download a short video (~5MB). Continue? (y/n): ").lower()
    
    if response == 'y':
        download_test = test_download()
        
        if download_test:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED!")
            print("="*60)
            print("\nYour YouTube downloads should work now!")
            print("You can start the server with: python start_server.py")
            return True
        else:
            print("\n" + "="*60)
            print("⚠️  DOWNLOAD TEST FAILED")
            print("="*60)
            print("\nCheck YOUTUBE_FIX_GUIDE.md for troubleshooting")
            return False
    else:
        print("\nSkipping download test.")
        print("Cookie test passed - downloads should work!")
        return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
