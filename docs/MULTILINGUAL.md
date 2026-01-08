# Multilingual Support

This project now supports multilingual video summarization using Sarvam AI for translation and text-to-speech.

## Setup

1. Get your Sarvam AI API key from [https://www.sarvam.ai/](https://www.sarvam.ai/)

2. Add to your `.env` file:
```
SARVAM_API_KEY=your_api_key_here
```

## Supported Languages

- English (en-IN)
- Hindi (hi-IN)
- Bengali (bn-IN)
- Kannada (kn-IN)
- Malayalam (ml-IN)
- Marathi (mr-IN)
- Odia (od-IN)
- Punjabi (pa-IN)
- Tamil (ta-IN)
- Telugu (te-IN)
- Gujarati (gu-IN)

## Usage

### Command Line

When running `main.py`, you'll be prompted for a target language:

```bash
python main.py
```

Follow the prompts:
1. Enter video source (YouTube URL or local file)
2. Enter topic query (optional)
3. **Enter target language** (e.g., 'hindi', 'tamil', or leave blank for English)
4. Enter output filename

### Programmatic Usage

```python
from main import process_video

process_video(
    video_path="path/to/video.mp4",
    youtube_url=None,
    query="sorting algorithms",
    output_path="output.mp4",
    target_language="hindi"  # or 'tamil', 'kannada', etc.
)
```

## How It Works

1. **Transcription**: Audio is transcribed in the original language (English)
2. **Summarization**: Summaries are generated in English using the LLM
3. **Translation**: Summaries are translated to the target language using Sarvam AI
4. **TTS Generation**: 
   - For non-English: Uses Sarvam AI TTS with native voice
   - For English: Falls back to Coqui TTS
5. **Video Assembly**: Creates final video with translated voiceovers

## Error Handling

The pipeline includes robust error handling:

- If translation fails, original English text is used
- If Sarvam TTS fails, falls back to Coqui TTS
- Missing API key gracefully skips translation
- Network errors don't break the pipeline

## Testing

Run the test suite to verify your setup:

```bash
python tests/test_multilingual.py
```

This will test:
- Translation API connectivity
- TTS generation
- Graceful fallback behavior

## Architecture

New files added:
- `utils/translator.py` - Sarvam AI translation and TTS integration
- `tests/test_multilingual.py` - Test suite for multilingual features

Modified files:
- `utils/summarizer.py` - Added translation step after summarization
- `main.py` - Added language parameter throughout pipeline
- `requirements.txt` - Added `requests` dependency
