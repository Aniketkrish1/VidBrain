# VidBrain - Topic-Based Video Summarization

## New Feature: Topic Query System

VidBrain now supports **intelligent topic-based video summarization**! You can ask for summaries of specific topics from videos, and the system will:

1. 🔍 **Search the transcript** for relevant content about your topic
2. 📝 **Generate an AI summary** explaining the concept clearly
3. 🎬 **Extract video clips** showing exactly where that topic is discussed
4. 🎙️ **Create a voiceover** narrating the summary
5. 🎥 **Assemble a final video** with synchronized clips and narration

---

## How It Works

### 1. User Query Input
Enter a topic you want to learn about (e.g., "quick sort", "recursion", "machine learning"):

```
Query: "explain quick sort algorithm"
```

### 2. Transcript Analysis
- Uses **VectorDB** with semantic search to find all relevant segments
- Filters results for high relevance using similarity scores
- Groups nearby segments together for context

### 3. AI Summarization
- Sends relevant transcript segments to **OpenRouter API**
- Generates a clear, concise educational summary (3-5 sentences)
- Focuses specifically on the queried topic

### 4. Video Clip Extraction
- Identifies timestamps where the topic is discussed
- Extracts video clips from those time ranges
- Adds padding for context
- Merges overlapping/adjacent clips for smooth flow

### 5. Voiceover Generation
- Converts the AI summary to natural speech using:
  - **edge-tts** (online, high quality) - preferred
  - **pyttsx3** (offline, fallback)
- Creates clear narration suitable for educational content

### 6. Final Assembly
- Combines extracted video clips
- Synchronizes clips with the voiceover
- Adjusts video speed if needed to match audio duration
- Produces a polished final video

---

## New Files & Components

### `utils/topic_query_processor.py`
**Purpose**: Process user queries and find relevant content

**Key Functions**:
- `search_topic_in_transcript(db, query, top_k)` - Search for topic in transcript
- `merge_adjacent_segments(segments, gap_threshold)` - Group related segments
- `generate_summary_from_segments(segments, query)` - Create AI summary
- `process_topic_query(db, query, sentences)` - Main pipeline function

**Example**:
```python
from utils.topic_query_processor import process_topic_query

result = process_topic_query(db, "quick sort", sentences)
# Returns: {
#   "summary": "Quick sort is a divide and conquer algorithm...",
#   "segments": [...],
#   "groups": [...],
#   "query": "quick sort"
# }
```

### `utils/clip_extractor.py`
**Purpose**: Extract relevant video clips based on timestamps

**Key Functions**:
- `prepare_clips_for_topic(video_path, segment_groups)` - Main extraction pipeline
- `extract_clips_from_timestamps(video_path, timestamps)` - Extract clips
- `merge_overlapping_clips(clips, gap_threshold)` - Merge nearby clips
- `expand_clip_duration(start, end, min_duration)` - Ensure reasonable clip length

**Example**:
```python
from utils.clip_extractor import prepare_clips_for_topic

clips_data = prepare_clips_for_topic(
    "video.mp4", 
    segment_groups,
    min_clip_duration=2.0
)
# Returns: {
#   "clips": ["clip_001.mp4", "clip_002.mp4"],
#   "timestamps": [(10.5, 25.3), (45.2, 60.8)],
#   "total_duration": 30.4
# }
```

### `utils/voiceover_generator.py`
**Purpose**: Generate high-quality voiceovers from text

**Key Functions**:
- `VoiceoverGenerator` - Main TTS class supporting multiple engines
- `generate_voiceover(text, output_path, engine)` - Simple generation function
- `generate_multiple_voiceovers(texts, output_dir)` - Batch generation

**Supported Engines**:
1. **edge-tts** (recommended) - Microsoft Edge TTS, high quality, requires internet
2. **pyttsx3** - Offline TTS, works without internet but lower quality

**Example**:
```python
from utils.voiceover_generator import generate_voiceover

audio_path = generate_voiceover(
    "Quick sort is a divide and conquer algorithm...",
    "voiceover.mp3",
    engine="edge-tts"  # or "pyttsx3" or "auto"
)
```

### `utils/video_assembler.py`
**Purpose**: Combine video clips with voiceover audio

**Key Functions**:
- `assemble_topic_video(clip_paths, voiceover_path, output_path)` - Main assembly
- `adjust_video_speed_to_audio(video_clip, target_duration)` - Sync video to audio
- `concatenate_clips_with_transitions(clips)` - Smooth clip joining

**Example**:
```python
from utils.video_assembler import assemble_topic_video

final_video = assemble_topic_video(
    clip_paths=["clip_001.mp4", "clip_002.mp4"],
    voiceover_path="voiceover.mp3",
    output_path="final_summary.mp4",
    adjust_speed=True
)
```

---

## Updated Workflow in `main.py`

### Old Workflow (Clustering)
```
Download → Transcribe → Cluster All Topics → Summarize → Assemble
```

### New Workflow (Topic Query)
```
Download → Transcribe → VectorDB Search → 
Find Relevant Segments → AI Summary → 
Extract Clips → Generate Voiceover → 
Assemble Final Video
```

### Key Changes:
1. **Conditional branching**: Detects if user provided a query
2. **Topic-specific processing**: Uses new utilities for targeted extraction
3. **Early summary callback**: Sends summary to frontend during processing
4. **Direct video assembly**: Bypasses old clustering for query mode

---

## Frontend Updates

### HTML (`index.html`)
Added new summary display section:
```html
<div id="summary-section" class="hidden">
  <h4>📝 Generated Summary</h4>
  <div id="summary-container">
    <div id="summary-query"></div>
    <div id="summary-text"></div>
  </div>
</div>
```

### JavaScript (`static/js/app.js`)
- Displays summary text as it's generated (during processing)
- Shows query topic and timestamps
- Embeds video player in results
- Improved progress tracking

### API (`app.py`)
- Stores query with job data
- Includes summaries in status responses
- Supports real-time summary updates

---

## Usage Examples

### Example 1: Learn About Quick Sort
```
YouTube URL: https://youtube.com/watch?v=xyz123
Query: "explain quick sort algorithm"
```

**Output**: 
- 2-minute video showing only quick sort explanation
- Voiceover: "Quick sort is a divide and conquer sorting algorithm that works by selecting a pivot element..."
- Video clips from original video showing quick sort demonstration

### Example 2: Understand Recursion
```
YouTube URL: https://youtube.com/watch?v=abc456
Query: "what is recursion and how does it work"
```

**Output**:
- 3-minute focused video on recursion
- Clear explanation with examples from the original lecture
- Synchronized narration and visuals

### Example 3: Full Video Summary (No Query)
```
YouTube URL: https://youtube.com/watch?v=def789
Query: (leave blank)
```

**Output**:
- Uses original clustering approach
- Condenses entire video
- Multiple topic summaries

---

## Configuration

### Environment Variables (.env)
```bash
# OpenRouter API for AI summaries
OPENROUTER_API_KEY=your_key_here
OPENROUTER_SUMMARIZE_MODEL=nvidia/nemotron-nano-9b-v2:free

# Vector Database
VECTOR_DB_PATH=vector_db.pkl

# TTS Configuration
# Use "edge-tts" for best quality (requires internet)
# Use "pyttsx3" for offline mode
TTS_ENGINE=auto

# Processing
TEMP_DIR=temp_processing
```

### Install Additional Dependencies
```bash
pip install edge-tts  # For high-quality voiceovers (optional)
pip install pyttsx3   # For offline voiceovers (already included)
```

---

## API Endpoints

### POST `/api/start`
**Parameters**:
- `youtube_url`: Video URL
- `query`: Topic to summarize (NEW!)
- `output_name`: Output filename (optional)
- `whisper_model`: Transcription model (optional)

**Response**:
```json
{
  "job_id": "uuid-here"
}
```

### GET `/api/status/{job_id}`
**Response**:
```json
{
  "status": "processing",
  "progress": 75,
  "stage": "Generating voiceover",
  "details": "Creating TTS audio...",
  "summaries": [
    {
      "cluster_id": 0,
      "summary": "Quick sort is a divide and conquer...",
      "start": 120.5,
      "end": 180.3,
      "query": "quick sort"
    }
  ]
}
```

### GET `/api/download/{job_id}`
Downloads the final video file.

---

## Performance Tips

1. **Query Specificity**: More specific queries yield better results
   - ✅ "explain bubble sort algorithm"
   - ❌ "sorting"

2. **TTS Engine**: Use `edge-tts` for best voice quality
   ```python
   # In voiceover_generator.py
   generator = VoiceoverGenerator(engine="edge-tts")
   ```

3. **Clip Duration**: Adjust minimum clip duration for better context
   ```python
   clips = prepare_clips_for_topic(
       video_path, 
       segments,
       min_clip_duration=3.0  # Longer clips = more context
   )
   ```

4. **Vector Search**: Increase `top_k` for comprehensive coverage
   ```python
   segments = search_topic_in_transcript(db, query, top_k=20)
   ```

---

## Troubleshooting

### Issue: "No content found for topic"
**Solutions**:
- Make query more specific
- Check if topic is actually in video
- Try related keywords

### Issue: "Voiceover generation failed"
**Solutions**:
- Install edge-tts: `pip install edge-tts`
- Check internet connection (for edge-tts)
- Fallback to pyttsx3 (offline mode)

### Issue: "Video clips not synchronized with audio"
**Solutions**:
- Enable speed adjustment: `adjust_speed=True`
- Check if clips cover the right timestamps
- Increase `gap_threshold` to merge more clips

### Issue: "Summary not displaying in frontend"
**Solutions**:
- Check browser console for errors
- Verify `summaries_callback` is being called
- Ensure API returns summaries in status

---

## Future Enhancements

- [ ] Multi-topic queries in one video
- [ ] Custom voice selection
- [ ] Subtitle generation for summaries
- [ ] Export summaries as PDF
- [ ] Support for multiple languages
- [ ] Advanced video transitions
- [ ] Real-time streaming of generated content

---

## Credits

**New Components Developed**:
- Topic Query Processor
- Intelligent Clip Extractor  
- Voiceover Generator
- Video Assembler

**Technologies Used**:
- OpenRouter API (AI Summaries)
- Sentence Transformers (Vector Search)
- MoviePy (Video Processing)
- edge-tts / pyttsx3 (Text-to-Speech)
- FastAPI (Backend)

---

## License

MIT License - See main README for details
