# Video Upload Feature - User Guide

## ✨ New Feature: Upload Your Own Videos!

You can now upload your own video files directly to VidBrain instead of only using YouTube URLs.

---

## 🎯 How to Use

### Option 1: YouTube URL (Original Method)
1. Open VidBrain at http://localhost:8000
2. Click the **"📺 YouTube URL"** tab (default)
3. Paste a YouTube URL
4. Enter your query (e.g., "explain quick sort")
5. Click **"Create Summary from YouTube"**

### Option 2: Upload Video (NEW!)
1. Open VidBrain at http://localhost:8000
2. Click the **"📁 Upload Video"** tab
3. Click **"Choose File"** and select your video
4. Enter your query (e.g., "explain recursion")
5. Click **"Create Summary from Upload"**

---

## 📁 Supported Video Formats

The following video formats are supported:
- ✅ **MP4** (.mp4) - Recommended
- ✅ **AVI** (.avi)
- ✅ **MOV** (.mov)
- ✅ **MKV** (.mkv)
- ✅ **WEBM** (.webm)
- ✅ **FLV** (.flv)
- ✅ **WMV** (.wmv)

---

## 📊 File Size Recommendations

| File Size | Processing Time | Recommendation |
|-----------|----------------|----------------|
| < 50 MB   | Fast (2-5 min) | ✅ Ideal |
| 50-200 MB | Medium (5-15 min) | ✅ Good |
| 200-500 MB| Slow (15-30 min) | ⚠️ Warning shown |
| > 500 MB  | Very slow (30+ min) | ⚠️ Confirmation required |

**Tips for faster processing:**
- Use shorter videos (< 20 minutes)
- Lower resolution (720p is fine)
- Compress before uploading
- Use MP4 format

---

## 🚀 Upload Process

### Step-by-Step

1. **Select Video File**
   - Click "Choose File" button
   - Browse to your video
   - Select the file

2. **Configure Settings** (Optional)
   - **Query**: What topic to summarize (required)
   - **Whisper Model**: base/small/medium/large (optional)
   - **Output Filename**: Custom name (optional)

3. **Upload & Process**
   - Click "Create Summary from Upload"
   - File uploads to server
   - Processing begins automatically

4. **Monitor Progress**
   - Real-time progress bar
   - Stage updates (uploading → transcribing → summarizing)
   - Estimated time remaining

5. **View Results**
   - Text summary appears first
   - Final video available for download
   - Embedded video player

---

## 📋 What Happens Behind the Scenes

### Upload Process
1. **Upload**: File is uploaded to server (`uploads/` directory)
2. **Validation**: File type and size checked
3. **Storage**: Saved with unique ID: `{job_id}_{filename}`

### Processing Process
1. **Audio Extraction**: Audio extracted from video
2. **Transcription**: Whisper transcribes audio to text
3. **Vector DB**: Transcript indexed for semantic search
4. **Query Search**: Finds segments matching your query
5. **AI Summary**: OpenRouter generates focused summary
6. **Clip Extraction**: Extracts relevant video segments
7. **Voiceover**: TTS generates narration from summary
8. **Assembly**: Combines clips with voiceover
9. **Output**: Final video saved to `outputs/` directory

### Cleanup
- ✅ Uploaded file deleted after successful processing
- ✅ Temporary files cleaned up
- ✅ Only final video kept

---

## 🔒 Security & Privacy

### File Handling
- Files saved temporarily during processing
- Automatically deleted after completion
- Stored in `uploads/` directory (not public)
- Unique filenames prevent conflicts

### Privacy
- Videos processed locally on your server
- No files sent to external services (except OpenRouter API for summaries)
- Transcription done locally with Whisper
- You control all data

### Storage
```
vidbrain/
├── uploads/          # Temporary uploaded files (auto-deleted)
│   └── {job_id}_video.mp4
├── temp_processing/  # Processing workspace (auto-cleaned)
│   └── extracted_clips/
└── outputs/          # Final videos (kept)
    └── summary_output.mp4
```

---

## ⚙️ Configuration

### Maximum File Size
Edit `app.py` to change max file size:

```python
# Add to app.py (optional)
app = FastAPI()
app.add_middleware(
    ...,
    max_upload_size=1024 * 1024 * 1000  # 1GB limit
)
```

### Allowed Formats
Edit `app.py` to add/remove formats:

```python
allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
# Add more: '.m4v', '.mpeg', '.mpg', etc.
```

---

## 🆘 Troubleshooting

### Upload Failed

**Error: "Invalid file type"**
- Solution: Use supported formats (MP4, AVI, MOV, etc.)
- Check file extension

**Error: "Upload failed: disk space"**
- Solution: Free up disk space
- Delete old videos from `outputs/` and `uploads/`

**Error: "File too large"**
- Solution: Compress video before uploading
- Use tools like Handbrake or FFmpeg

### Processing Failed

**Error: "Could not extract audio"**
- Solution: Video may be corrupted
- Try re-encoding: `ffmpeg -i input.mp4 -c copy output.mp4`

**Error: "Transcription failed"**
- Solution: Audio track may be missing
- Check video has audio

**Error: "Out of memory"**
- Solution: Video too large
- Use smaller file or add more RAM

---

## 💡 Use Cases

### Educational Videos
- Upload lecture recordings
- Extract explanations of specific concepts
- Create study summaries

### Meeting Recordings
- Upload Zoom/Teams recordings
- Get summaries of specific topics discussed
- Share key points with team

### Training Videos
- Upload internal training content
- Extract specific procedures
- Create quick reference guides

### Personal Videos
- Upload your own screen recordings
- Tutorial videos
- Presentation recordings

---

## 🎨 Frontend Features

### Tab Interface
- **YouTube Tab**: For YouTube URLs
- **Upload Tab**: For local files
- Smooth tab switching
- Clean, intuitive design

### Upload Progress
- Real-time upload progress
- File size display
- Processing stages
- Time estimates

### File Validation
- Client-side validation
- File type checking
- Size warnings
- User confirmations

---

## 📊 API Endpoints

### POST /api/upload
Upload and process a video file

**Parameters:**
- `video`: Video file (multipart/form-data)
- `query`: Search query (string)
- `whisper_model`: Whisper model name (optional)
- `output_name`: Output filename (optional)

**Response:**
```json
{
  "job_id": "uuid-string"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "video=@my_video.mp4" \
  -F "query=explain quick sort" \
  -F "whisper_model=base"
```

---

## 🔍 Comparison: YouTube vs Upload

| Feature | YouTube URL | Upload Video |
|---------|------------|--------------|
| **Source** | Online video | Local file |
| **Speed** | Depends on download | Upload + process |
| **Restrictions** | YouTube bot detection | None |
| **Privacy** | Public URL | Fully private |
| **File size** | Unlimited | Limited by server |
| **Formats** | YouTube only | Multiple formats |
| **Use case** | Public content | Private/custom videos |

---

## ✨ Benefits of Upload Feature

1. **Privacy**: Process private videos without uploading to YouTube
2. **No restrictions**: No YouTube bot detection issues
3. **Flexibility**: Use any video source
4. **Offline**: Works with local recordings
5. **Control**: Full control over content
6. **Speed**: No download needed, direct processing

---

## 🎉 Example Workflow

### Scenario: Lecture Recording

1. **Record**: Record your lecture with OBS/Zoom
2. **Save**: Save as MP4 file
3. **Upload**: Open VidBrain, upload the file
4. **Query**: Enter "explain binary search trees"
5. **Process**: Wait for AI to generate summary
6. **Share**: Download summary video, share with students

**Result**: 2-hour lecture → 3-minute focused explanation!

---

## 📝 Best Practices

### Before Upload
- ✅ Check video has clear audio
- ✅ Ensure video is in supported format
- ✅ Compress if > 200MB
- ✅ Test with shorter clip first

### During Processing
- ✅ Keep browser tab open
- ✅ Don't close server
- ✅ Monitor progress
- ✅ Check for errors

### After Processing
- ✅ Download summary immediately
- ✅ Old videos auto-cleaned
- ✅ Close unused tabs
- ✅ Free up disk space

---

## 🚀 Quick Start

### 1. Start Server
```bash
python start_server.py
```

### 2. Open Browser
```
http://localhost:8000
```

### 3. Upload Video
- Click "📁 Upload Video" tab
- Select your video file
- Enter query: "explain [topic]"
- Click "Create Summary from Upload"

### 4. Wait for Results
- Upload progress shown
- Processing stages displayed
- Summary appears first
- Video ready for download

That's it! 🎊

---

## 📞 Need Help?

If you have issues:
1. Check file format and size
2. Ensure enough disk space
3. Try smaller video first
4. Check server logs for errors
5. See troubleshooting section above

Happy video processing! 🎬✨
