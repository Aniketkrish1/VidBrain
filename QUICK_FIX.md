# 🔧 Quick Fix for YouTube Bot Detection Error

## The Problem
```
ERROR: [youtube] Sign in to confirm you're not a bot
```

## ✅ The Solution (30 seconds)

### Step 1: Make sure you're logged into YouTube
**On macOS:** Open Safari, Chrome, or Firefox and sign into YouTube  
**On Windows:** Open Chrome, Firefox, or Edge and sign into YouTube  
**On Linux:** Open Chrome/Chromium or Firefox and sign into YouTube

### Step 2: Update yt-dlp (optional but recommended)
```bash
pip install --upgrade yt-dlp
```

### Step 3: Test the fix
```bash
python test_youtube_download.py
```

### Step 4: Restart your server
```bash
python start_server.py
```

That's it! The system will automatically use your browser cookies.

---

## 📋 What Changed?

The `downloader.py` file has been updated to:
- ✅ Automatically use your browser's YouTube cookies
- ✅ Try multiple browsers (Chrome, Firefox, Edge, Brave)
- ✅ Better error messages
- ✅ Enhanced bot evasion

---

## 🆘 Still Having Issues?

### Quick Checks:
1. ✓ Are you logged into YouTube in your browser?
   - **macOS:** Safari, Chrome, or Firefox
   - **Windows:** Chrome, Firefox, or Edge
   - **Linux:** Chrome/Chromium or Firefox
2. ✓ Is your browser up to date?
3. ✓ Did you update yt-dlp?

### Try This:
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Clear temp files
rm -rf temp_processing/*  # Linux/Mac
rmdir /s temp_processing   # Windows

# Test again
python test_youtube_download.py
```

### More Help:
See **YOUTUBE_FIX_GUIDE.md** for detailed troubleshooting

---

## 💡 How It Works

When you download a video, VidBrain now:
1. **Detects your operating system** (Windows/macOS/Linux)
2. **Uses appropriate browsers** for your platform
3. **Automatically tries available browsers** in order
4. YouTube sees the request as coming from your logged-in browser
5. Download works! 🎉

### Browser Priority by Platform

**macOS:**
1. Safari (native to macOS)
2. Chrome
3. Firefox
4. Edge
5. Brave

**Windows:**
1. Chrome
2. Firefox
3. Edge
4. Brave

**Linux:**
1. Chrome/Chromium
2. Firefox
3. Brave

No manual cookie export needed - it's all automatic!

---

## ✨ Benefits

- **No manual setup** - works automatically
- **No cookie files** - uses your browser directly  
- **Always up-to-date** - uses your current login session
- **Secure** - doesn't store or expose your credentials

---

## 📝 For Developers

The key changes in `utils/downloader.py`:

```python
# New function to get yt-dlp options with cookies
def get_ydl_options(video_dir: str, use_cookies: bool = True) -> dict:
    ydl_opts = {
        # ... other options ...
        'cookiesfrombrowser': ('chrome',),  # Auto-use browser cookies
    }
    return ydl_opts
```

The system tries browsers in this order:

**On macOS:**
1. Safari (native)
2. Chrome
3. Firefox
4. Edge
5. Brave
6. Opera

**On Windows:**
1. Chrome
2. Firefox
3. Edge
4. Brave
5. Opera

**On Linux:**
1. Chrome/Chromium
2. Firefox
3. Brave
4. Opera

---

## 🎯 Success!

You'll know it's working when you see:

**On macOS:**
```
Platform: darwin, Available browsers: ['safari', 'chrome', 'firefox', 'edge', 'brave', 'opera']
INFO: Will attempt to use safari cookies
INFO: Starting video download...
```

**On Windows:**
```
Platform: windows, Available browsers: ['chrome', 'firefox', 'edge', 'brave', 'opera']
INFO: Will attempt to use chrome cookies
INFO: Starting video download...
```

**On Linux:**
```
Platform: linux, Available browsers: ['chrome', 'chromium', 'firefox', 'brave', 'opera']
INFO: Will attempt to use chrome cookies
INFO: Starting video download...
```

Happy video processing! 🎬
