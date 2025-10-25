# VidBrain Project Flow - Complete Explanation

## 🎯 Project Purpose
**VidBrain** takes a long video and a user query (like "explain quick sort"), then creates a short summary video that:
- Extracts only the relevant parts from the original video
- Generates an AI summary of that topic
- Creates a voiceover narration
- Combines clips with voiceover into a final summary video

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (index.html)                    │
│  - YouTube URL input or Upload video                            │
│  - Query input: "explain quick sort"                            │
│  - Progress display, Summary display, Video player              │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP POST
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (app.py - FastAPI)                  │
│  - Receives request, creates job ID                             │
│  - Runs async processing in background                          │
│  - Returns progress updates via polling                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE (main.py)                       │
│  - Orchestrates entire video processing workflow                │
│  - Calls all utility modules in sequence                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    UTILITY MODULES (utils/)                      │
│  - downloader.py: Download YouTube videos                       │
│  - transcriber.py: Convert audio to text                        │
│  - database.py: Vector database for semantic search             │
│  - topic_query_processor.py: AI analysis & summary              │
│  - clip_extractor.py: Extract video clips                       │
│  - voiceover_generator.py: Text-to-speech                       │
│  - video_assembler.py: Combine clips + audio                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT (outputs/)                            │
│  - summary_output.mp4: Final summary video                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Workflow - Step by Step

### 📍 **STEP 1: User Input (Frontend)**

**File**: `index.html` + `static/js/app.js`

```
User Actions:
1. Opens: http://localhost:8000
2. Chooses tab: "YouTube URL" or "Upload Video"
3. Enters:
   - URL: https://www.youtube.com/watch?v=kPRA0W1kECg
   - Query: "explain quick sort"
   - (Optional) Whisper model, output filename
4. Clicks: "Create Summary"
```

**What Happens**:
```javascript
// app.js
form.addEventListener('submit', async (e) => {
  // 1. Collect form data
  const formData = new FormData(form);
  
  // 2. Send POST request to backend
  const response = await fetch('/api/start', {
    method: 'POST',
    body: formData
  });
  
  // 3. Get job ID
  const data = await response.json();
  const jobId = data.job_id;
  
  // 4. Start polling for progress
  poll(jobId);
});
```

---

### 📍 **STEP 2: Backend Receives Request**

**File**: `app.py` (FastAPI server)

```python
@app.post("/api/start")
async def start_processing(
    youtube_url: str = Form(""),
    query: str = Form(""),
    whisper_model: str = Form("base"),
    output_name: str = Form("summary")
):
    # 1. Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # 2. Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "stage": "starting",
        "message": "Initializing...",
        "summaries": []
    }
    
    # 3. Start background processing
    asyncio.create_task(process_video_task(
        job_id, youtube_url, query, whisper_model, output_name
    ))
    
    # 4. Return job ID to frontend
    return {"job_id": job_id}
```

**Job Status Updates**:
```python
def update_progress(stage, percent, message):
    jobs[job_id]["stage"] = stage
    jobs[job_id]["progress"] = percent
    jobs[job_id]["message"] = message
    # Frontend polls /api/status/{job_id} to get these updates
```

---

### 📍 **STEP 3: Main Pipeline Orchestration**

**File**: `main.py` → `process_video()` function

This is the **brain** of the system. It calls all other modules in sequence.

```python
def process_video(
    youtube_url="",
    uploaded_video_path=None,
    query="",
    whisper_model="base",
    output_filename="summary",
    progress_callback=None,
    summaries_callback=None
):
    """
    Main orchestration function.
    Coordinates all processing steps.
    """
```

**Flow**:
```
1. Download/Load Video    →  update_progress("download", 5%)
2. Extract Audio          →  update_progress("audio", 15%)
3. Transcribe Audio       →  update_progress("transcribe", 25%)
4. Build Vector DB        →  update_progress("vectordb", 45%)
5. Topic Query Processing →  update_progress("query", 52%)
6. Extract Clips          →  update_progress("clips", 60%)
7. Generate Voiceover     →  update_progress("voiceover", 80%)
8. Assemble Final Video   →  update_progress("assembly", 90%)
9. Cleanup & Complete     →  update_progress("completed", 100%)
```

---

### 📍 **STEP 4: Download Video**

**File**: `utils/downloader.py`

```python
def download_and_extract_audio(youtube_url, output_dir):
    """
    Downloads YouTube video and extracts audio.
    Uses yt-dlp with browser cookies to avoid bot detection.
    """
    
    # 1. Get available browsers for cookie extraction
    browsers = get_available_browsers()  # ['chrome', 'firefox', 'edge']
    
    # 2. Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': video_path,
        'cookiesfrombrowser': (browsers[0], None, None, None)  # Use browser cookies
    }
    
    # 3. Download video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # 4. Extract audio using MoviePy
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    
    return video_path, audio_path
```

**Output**:
```
temp_processing/
├── downloaded_video.mp4      # Full video
└── extracted_audio.mp3       # Audio only
```

---

### 📍 **STEP 5: Transcribe Audio**

**File**: `utils/transcriber.py`

```python
def transcribe_video(audio_path, model_name="base"):
    """
    Converts audio to text using OpenAI Whisper.
    Returns timestamped transcript.
    """
    
    # 1. Load Whisper model
    model = whisper.load_model(model_name)
    
    # 2. Transcribe audio
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en"
    )
    
    # 3. Parse into sentences with timestamps
    sentences = []
    for segment in result['segments']:
        sentences.append({
            "text": segment['text'],
            "start": segment['start'],  # seconds
            "end": segment['end']       # seconds
        })
    
    # 4. Save as SRT file
    save_as_srt(sentences, "transcript.srt")
    
    return sentences
```

**Output**:
```python
sentences = [
    {"text": "Today I'll explain sorting algorithms.", "start": 0.0, "end": 3.5},
    {"text": "Let's start with bubble sort.", "start": 3.5, "end": 6.2},
    {"text": "Quick sort is a divide and conquer algorithm.", "start": 45.2, "end": 49.8},
    {"text": "It works by selecting a pivot element.", "start": 49.8, "end": 53.1},
    ...
]
```

---

### 📍 **STEP 6: Build Vector Database**

**File**: `utils/database.py`

```python
class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Creates semantic search database using sentence embeddings.
        """
        self.encoder = SentenceTransformer(model_name)
        self.segments = []
        self.embeddings = None
    
    def build(self, segments):
        """
        Converts text segments to vector embeddings.
        """
        # 1. Extract text from segments
        texts = [seg["text"] for seg in segments]
        
        # 2. Generate embeddings (768-dimensional vectors)
        self.embeddings = self.encoder.encode(texts)
        
        # 3. Store segments with embeddings
        self.segments = segments
        
        # 4. Build FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(768)
        self.index.add(self.embeddings)
    
    def search(self, query, top_k=10):
        """
        Finds segments most similar to query.
        """
        # 1. Encode query to vector
        query_embedding = self.encoder.encode([query])
        
        # 2. Search for nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 3. Return matching segments with scores
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                **self.segments[idx],
                "score": 1 / (1 + dist)  # Convert distance to similarity
            })
        
        return results
```

**What It Does**:
```
"Quick sort" (query) → [0.12, 0.45, 0.89, ...] (768-dim vector)
                              ↓ Compare with
Transcript segments → Each converted to 768-dim vector
                              ↓ Find similar
Returns top 15 most similar segments about "quick sort"
```

---

### 📍 **STEP 7: Topic Query Processing** ⭐ **KEY STEP**

**File**: `utils/topic_query_processor.py`

This is where the **magic** happens! 🎩✨

```python
def process_topic_query(db, query, sentences):
    """
    1. Searches for relevant segments
    2. Sends FULL transcript to AI
    3. Gets focused summary about the query
    4. Returns timestamps and summary
    """
    
    # STEP 7A: Search for relevant segments
    relevant_segments = search_topic_in_transcript(db, query, top_k=15)
    # Returns: 15 segments about "quick sort" with timestamps
    
    # STEP 7B: Build full transcript
    full_transcript = " ".join([sent["text"] for sent in sentences])
    # All 134 sentences combined into one text (10,814 chars)
    
    # STEP 7C: Send to AI for analysis
    summary_result = generate_summary_from_full_transcript(
        full_transcript,      # Complete video content
        query,               # "explain quick sort"
        relevant_segments    # For timestamp extraction
    )
    
    # STEP 7D: Merge adjacent segments
    segment_groups = merge_adjacent_segments(relevant_segments)
    # Groups nearby segments: [seg1, seg2] become one clip
    
    return {
        "summary": summary_result["summary"],  # AI-generated text
        "segments": relevant_segments,          # 15 segments
        "groups": segment_groups,               # 12 merged groups
        "query": query
    }
```

#### **Sub-step: AI Summary Generation** 🤖

```python
def generate_summary_from_full_transcript(full_transcript, query, segments):
    """
    Sends entire transcript to OpenRouter API.
    AI reads everything and generates focused summary.
    """
    
    # Create smart prompt
    prompt = f"""
    You are analyzing a video transcript to create a focused summary about "{query}".
    
    TASK:
    1. Read the ENTIRE transcript carefully
    2. Identify ALL parts where "{query}" is explained or discussed
    3. Create a clear, comprehensive summary explaining "{query}"
    4. Write in natural spoken language (for voiceover)
    
    TRANSCRIPT:
    {full_transcript}  # ← All 10,814 characters!
    
    Create a 4-8 sentence summary focusing ONLY on "{query}".
    """
    
    # Call OpenRouter API
    response = openrouter_client.chat.completions.create(
        model="nvidia/nemotron-nano-9b-v2:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    summary = response.choices[0].message.content
    
    return {
        "summary": summary,
        "timestamps": [(seg["start"], seg["end"]) for seg in segments],
        "confidence": 0.8
    }
```

**AI Output Example**:
```
"Quick sort is a highly efficient divide-and-conquer sorting algorithm. 
It works by selecting a pivot element and partitioning the array into 
elements less than and greater than the pivot. These sub-arrays are 
recursively sorted, achieving O(n log n) average time complexity. 
However, in worst cases with poor pivot selection, it can degrade to 
O(n²). Its in-place sorting capability makes it memory efficient."
```

---

### 📍 **STEP 8: Extract Video Clips**

**File**: `utils/clip_extractor.py`

```python
def prepare_clips_for_topic(video_path, segment_groups, output_dir):
    """
    Extracts video clips based on timestamps.
    """
    
    # STEP 8A: Extract timestamps from groups
    timestamps = []
    for group in segment_groups:
        start = group[0]["start"]  # 45.2s
        end = group[-1]["end"]      # 53.1s
        
        # Expand clip duration (add padding)
        start = max(0, start - 0.5)
        end = end + 0.5
        
        timestamps.append((start, end))
    
    # Result: [(44.7, 53.6), (98.0, 105.2), ...]
    
    # STEP 8B: Merge overlapping clips
    merged = merge_overlapping_clips(timestamps, gap_threshold=3.0)
    # If clips are < 3 seconds apart, merge them
    
    # STEP 8C: Extract clips from video
    clip_paths = extract_clips_from_timestamps(video_path, merged, output_dir)
    
    return {
        "clips": clip_paths,
        "timestamps": merged,
        "total_duration": sum(end - start for start, end in merged)
    }
```

**Clip Extraction**:
```python
def extract_clips_from_timestamps(video_path, timestamps, output_dir):
    """
    Uses MoviePy to cut video clips.
    """
    video = VideoFileClip(video_path)
    clip_paths = []
    
    for idx, (start, end) in enumerate(timestamps):
        # Extract clip
        clip = video.subclipped(start, end)
        
        # Save to file
        clip_path = f"temp_processing/clip_{idx:03d}.mp4"
        clip.write_videofile(
            clip_path,
            codec='libx264',
            audio_codec='aac'
        )
        
        clip_paths.append(clip_path)
        clip.close()
    
    video.close()
    return clip_paths
```

**Output**:
```
temp_processing/
├── clip_000.mp4   (44.7s - 53.6s)
├── clip_001.mp4   (98.0s - 105.2s)
├── clip_002.mp4   (180.5s - 192.8s)
...
├── clip_011.mp4   (450.2s - 458.9s)
```

---

### 📍 **STEP 9: Generate Voiceover**

**File**: `utils/voiceover_generator.py`

```python
class VoiceoverGenerator:
    def generate(self, text, output_path):
        """
        Converts text summary to speech audio.
        Uses edge-tts (online) or pyttsx3 (offline).
        """
        
        # Try edge-tts first (better quality)
        try:
            communicate = edge_tts.Communicate(
                text, 
                voice="en-US-GuyNeural",  # Male voice
                rate="+0%",                # Normal speed
                pitch="+0Hz"               # Normal pitch
            )
            
            await communicate.save(output_path)
            return output_path
            
        except:
            # Fallback to pyttsx3 (offline)
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('voice', voices[0].id)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            return output_path
```

**Input** (AI Summary):
```
"Quick sort is a highly efficient divide-and-conquer sorting algorithm..."
```

**Output**:
```
temp_processing/voiceover.mp3
Duration: 28.5 seconds
```

---

### 📍 **STEP 10: Assemble Final Video**

**File**: `utils/video_assembler.py`

```python
def assemble_topic_video(clips, voiceover_audio, output_path):
    """
    Combines video clips with voiceover narration.
    """
    
    # STEP 10A: Load all clips
    video_clips = [VideoFileClip(clip) for clip in clips]
    
    # STEP 10B: Concatenate clips
    concatenated = concatenate_videoclips(video_clips, method="compose")
    # Total video duration: 77.9 seconds
    
    # STEP 10C: Load voiceover audio
    voiceover = AudioFileClip(voiceover_audio)
    # Voiceover duration: 28.5 seconds
    
    # STEP 10D: Adjust video speed to match audio
    if concatenated.duration > voiceover.duration:
        # Speed up video to match audio length
        speed_factor = concatenated.duration / voiceover.duration
        concatenated = concatenated.speedx(speed_factor)
    
    # STEP 10E: Set audio track
    final = concatenated.set_audio(voiceover)
    
    # STEP 10F: Write final video
    final.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=24
    )
    
    # Cleanup
    final.close()
    voiceover.close()
    for clip in video_clips:
        clip.close()
    
    return output_path
```

**Output**:
```
outputs/summary_output.mp4
Duration: 28.5 seconds
Contains: 12 clips about quick sort + AI voiceover narration
```

---

### 📍 **STEP 11: Return Results**

**File**: `app.py`

```python
# Update job status
jobs[job_id] = {
    "status": "completed",
    "progress": 100,
    "stage": "completed",
    "message": "Topic video created successfully: quick sort",
    "summaries": [{
        "cluster_id": 0,
        "summary": "Quick sort is a highly efficient...",
        "query": "quick sort"
    }],
    "video_url": "/outputs/summary_output.mp4"
}
```

**Frontend polls `/api/status/{job_id}`** and receives:
```json
{
  "status": "completed",
  "progress": 100,
  "summaries": [
    {
      "summary": "Quick sort is a highly efficient divide-and-conquer..."
    }
  ],
  "video_url": "/outputs/summary_output.mp4"
}
```

---

### 📍 **STEP 12: Display Results**

**File**: `static/js/app.js`

```javascript
function displayResults(data) {
    // Show summary text
    document.getElementById('summaries').innerHTML = `
        <div class="summary-item">
            <h3>Generated Summary</h3>
            <p>${data.summaries[0].summary}</p>
        </div>
    `;
    
    // Show video player
    document.getElementById('videoContainer').innerHTML = `
        <video controls>
            <source src="${data.video_url}" type="video/mp4">
        </video>
        <a href="${data.video_url}" download>Download Video</a>
    `;
}
```

**User sees**:
```
✅ Status: completed

📝 Generated Summary:
"Quick sort is a highly efficient divide-and-conquer sorting 
algorithm. It works by selecting a pivot element and partitioning 
the array into elements less than and greater than the pivot..."

🎬 Video Player:
[▶️ Play] [Download] summary_output.mp4
```

---

## 🗂️ File Structure & Responsibilities

```
vidbrain/
│
├── app.py                          # FastAPI server, API endpoints
├── main.py                         # Main orchestration pipeline
├── start_server.py                 # Server startup script
│
├── index.html                      # Frontend UI
├── static/
│   ├── js/app.js                  # Frontend JavaScript logic
│   └── css/styles.css             # Styling
│
├── utils/                          # Core processing modules
│   ├── downloader.py              # YouTube download + audio extraction
│   ├── transcriber.py             # Whisper transcription
│   ├── database.py                # Vector DB for semantic search
│   ├── topic_query_processor.py   # AI analysis & summary
│   ├── clip_extractor.py          # Video clip extraction
│   ├── voiceover_generator.py     # Text-to-speech
│   └── video_assembler.py         # Combine clips + audio
│
├── outputs/                        # Final videos
│   └── summary_output.mp4
│
├── temp_processing/                # Temporary files (cleaned up)
│   ├── downloaded_video.mp4
│   ├── extracted_audio.mp3
│   ├── clip_000.mp4 ... clip_011.mp4
│   └── voiceover.mp3
│
└── uploads/                        # User-uploaded videos
    └── {job_id}_video.mp4
```

---

## 🔑 Key Technologies

### Backend
- **FastAPI**: Web server & API
- **yt-dlp**: YouTube video download
- **Whisper**: Audio transcription
- **SentenceTransformers**: Text embeddings
- **FAISS**: Vector similarity search
- **HDBSCAN**: Topic clustering
- **MoviePy**: Video processing
- **edge-tts / pyttsx3**: Text-to-speech
- **OpenRouter API**: AI summarization

### Frontend
- **HTML5**: Structure
- **JavaScript (Vanilla)**: Logic
- **CSS3**: Styling
- **Fetch API**: HTTP requests

---

## 📊 Data Flow Example

### Input:
```
URL: https://www.youtube.com/watch?v=kPRA0W1kECg
Query: "quick sort"
```

### Processing:
```
1. Download → 15-minute video (720p, 150MB)
2. Transcribe → 134 sentences, 10,814 characters
3. Vector DB → 134 embeddings (768-dim each)
4. Search → Find 15 segments about "quick sort"
5. AI Analysis → Full transcript → Focused summary (450 chars)
6. Extract → 12 clips (total 77.9 seconds)
7. Voiceover → 28.5 seconds of speech
8. Assemble → Speed up clips to 28.5s, add voiceover
```

### Output:
```
summary_output.mp4
- Duration: 28.5 seconds
- Shows: Only "quick sort" sections
- Audio: AI-generated explanation
- Size: ~8MB
```

---

## ⚡ Performance Metrics

### Typical Processing Time:
```
Download:        30-60 seconds
Transcription:   60-120 seconds (depends on video length)
Vector DB:       5-10 seconds
AI Summary:      3-5 seconds
Clip Extraction: 20-30 seconds
Voiceover:       2-5 seconds
Assembly:        10-20 seconds
─────────────────────────────────
Total:           ~3-5 minutes for 15-minute video
```

---

## 🎯 Summary

**VidBrain transforms long videos into focused summaries by**:

1. 📥 **Downloading** YouTube videos
2. 🎤 **Transcribing** audio to text with timestamps
3. 🔍 **Searching** transcript for your query using AI embeddings
4. 🤖 **Analyzing** full transcript with AI to generate focused summary
5. ✂️ **Extracting** relevant video clips from timestamps
6. 🗣️ **Creating** voiceover narration from AI summary
7. 🎬 **Assembling** clips + voiceover into final video
8. 📺 **Displaying** summary text and video in browser

**Result**: 15-minute video → 30-second focused summary about your topic!

---

## 💡 Key Innovation

**The full transcript analysis** (Step 7) is what makes VidBrain special:
- Instead of just extracting clips, it **understands** the content
- AI reads the **entire** video to generate accurate summaries
- Combines semantic search (for timestamps) + AI analysis (for understanding)
- Creates natural, comprehensive explanations

**This is what makes your summaries accurate and focused!** 🎯✨
