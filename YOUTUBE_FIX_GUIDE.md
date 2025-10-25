# YouTube Download Fix Guide

## Problem
YouTube is blocking yt-dlp downloads with "Sign in to confirm you're not a bot" error.

## Solution
The system now automatically uses your browser cookies to authenticate with YouTube.

---

## ✅ Quick Fix (Recommended)

### Option 1: Use Browser Cookies (Automatic)

1. **Make sure you're logged into YouTube** in one of these browsers:
   - Safari (prioritized)
   - Chrome
   - Firefox
   - Edge
   - Brave
   - Opera

2. **That's it!** The system will automatically use your browser's YouTube cookies.

3. **Restart the server** if it was already running:
   ```bash
   # Stop the server (Ctrl+C)
   # Start it again
   python start_server.py
   ```

4. **Try downloading again** - it should work now!

---

## 🔧 If Still Having Issues

### Option 2: Update yt-dlp

YouTube frequently changes their API. Update yt-dlp to the latest version:

```bash
pip install --upgrade yt-dlp
```

### Option 3: Use a Different Browser

If Safari doesn't work, try:
1. Open Chrome/Firefox/Edge
2. Sign into YouTube
3. Restart the server
4. Try again

### Option 4: Clear Browser Cache & Re-login

Sometimes stale cookies cause issues:
1. Clear your browser cache and cookies
2. Sign into YouTube again
3. Restart the server
4. Try downloading

### Option 5: Wait and Retry

YouTube's bot detection can be temporary:
- Wait 5-10 minutes
- Try a different video
- Try during off-peak hours

---

## 🎯 How It Works

The updated `downloader.py` now:

1. **Automatically detects** your installed browsers
2. **Uses your browser cookies** for authentication
3. **Tries multiple browsers** if one fails
4. **Enhanced headers** to avoid detection
5. **Better error messages** to help troubleshooting

### Code Changes

The system now includes:
- `cookiesfrombrowser` option in yt-dlp
- Enhanced user-agent headers
- Automatic browser detection
- Better error handling

---

## 📋 Manual Cookie Export (Advanced)

If automatic method doesn't work, you can manually export cookies:

### Using Browser Extension

1. **Install Extension:**
   - Chrome: "Get cookies.txt LOCALLY"
   - Firefox: "cookies.txt"

2. **Export Cookies:**
   - Go to youtube.com
   - Click extension icon
   - Export cookies to `cookies.txt`

3. **Place Cookie File:**
   ```
   vidbrain/
   └── cookies.txt  # Put it here
   ```

4. **Update Code:**
   In `utils/downloader.py`, add this to `ydl_opts`:
   ```python
   'cookiefile': 'cookies.txt'
   ```

---

## 🚫 Common Errors & Solutions

### Error: "No browser cookies available"
**Solution:** Make sure you're logged into YouTube in Safari/Chrome/Firefox

### Error: "HTTP Error 403: Forbidden"
**Solution:** 
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Clear browser cache and re-login to YouTube

### Error: "Video unavailable"
**Solution:** 
- Video might be region-locked or private
- Try a different video

### Error: "Unable to extract video data"
**Solution:**
- Update yt-dlp to latest version
- Wait a few minutes and retry

---

## 🔍 Testing

Test if your browser cookies are accessible:

```python
# test_cookies.py
import yt_dlp

ydl_opts = {
    'quiet': True,
    'cookiesfrombrowser': ('chrome',),  # or 'firefox', 'edge'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info('https://www.youtube.com/watch?v=dQw4w9WgXcQ', download=False)
    print(f"✓ Success! Video: {info['title']}")
```

---

## 📱 Alternative: Use Public Videos

If you're still having issues:
- Try videos from smaller channels
- Use unlisted/public videos
- Avoid age-restricted content

---

## 🆘 Still Not Working?

### Check Browser Support

Make sure your browser version is supported:
- Chrome 80+
- Firefox 70+
- Edge 80+

### Check Permissions

On Windows, make sure Python has permission to access browser data:
- Run terminal as Administrator (if needed)
- Check antivirus isn't blocking access

### Use Alternative Method

Instead of YouTube URL, you can:
1. Download video manually from YouTube
2. Use local file path in VidBrain
3. Process the local video file

---

## 🎉 Success Indicators

You'll know it's working when you see:

```
INFO: Attempting to use safari cookies
INFO: Starting video download...
INFO: Video downloaded: temp_processing/video_title/video.mp4
INFO: Extracting audio from video...
INFO: Video & audio ready
```

---

## 💡 Pro Tips

1. **Keep browser open** while downloading (especially Safari)
2. **Stay logged into YouTube** in your browser
3. **Update yt-dlp regularly**: `pip install --upgrade yt-dlp`
4. **Use popular videos** - less likely to be restricted
5. **Avoid age-restricted content** - requires additional authentication

---

## 📝 Summary

The bot detection issue is now handled automatically by:
- ✅ Using your browser's YouTube cookies
- ✅ Enhanced request headers
- ✅ Multiple browser fallback
- ✅ Better error messages
- ✅ Automatic retry logic

Just make sure you're **logged into YouTube in Chrome or Firefox**, and the system will handle the rest!
