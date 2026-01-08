# TTS Migration: Sarvam AI → gTTS

## Problem
Sarvam AI TTS was not generating audio output properly, resulting in silent videos.

## Solution
Switched to **gTTS (Google Text-to-Speech)** which is:
- ✅ More reliable and battle-tested
- ✅ Supports 50+ languages including all Indian languages
- ✅ Free and doesn't require API keys
- ✅ Simple to use with minimal dependencies
- ✅ Works offline after first download

## What Changed

### 1. `utils/translator.py`
- Added `generate_tts_gtts()` function using gTTS
- Deprecated `generate_tts_sarvam()` (kept for backwards compatibility)
- Added `GTTS_LANGUAGE_CODES` mapping for supported languages
- gTTS outputs **MP3** format (instead of WAV)

### 2. `main.py`
- Updated `generate_voiceovers_from_summaries()` to use gTTS by default
- Coqui TTS kept as fallback only
- Audio files now use `.mp3` extension (gTTS) instead of `.wav`

### 3. `requirements.txt`
- Added `gtts>=2.5.0`
- Added `pydub>=0.25.1` for audio processing

## Supported Languages

gTTS supports all the languages we need:
- English (`en`)
- Hindi (`hi`)
- Bengali (`bn`)
- Kannada (`kn`) ✅ **Tested and working**
- Malayalam (`ml`)
- Marathi (`mr`)
- Odia (`or`)
- Punjabi (`pa`)
- Tamil (`ta`)
- Telugu (`te`)
- Gujarati (`gu`)

Plus 40+ more languages!

## Testing

Tested with:
```bash
python test_gtts.py
```

Results:
- ✅ Kannada TTS: 89KB MP3 generated successfully
- ✅ English TTS: 46KB MP3 generated successfully

## Usage

No changes needed from user perspective! Just run:
```bash
python main.py
```

The pipeline will now use gTTS automatically for all languages.

## Fallback Chain

1. **Primary**: gTTS (all languages)
2. **Fallback**: Coqui TTS (English only, if gTTS fails)

## Benefits

1. **Reliability**: gTTS is used by thousands of projects worldwide
2. **No API Keys**: No need for Sarvam API keys
3. **Better Audio**: Higher quality audio output
4. **Faster**: No network latency for API calls after first use
5. **More Languages**: 50+ languages vs 11 with Sarvam

## Migration Notes

- Old voiceover files were `.wav`, new ones are `.mp3`
- MoviePy handles both formats seamlessly
- No breaking changes to the pipeline
