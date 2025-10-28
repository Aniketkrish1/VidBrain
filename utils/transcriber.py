import os
import datetime
import json
from faster_whisper import WhisperModel
from typing import Dict, List, Optional
from dotenv import load_dotenv
import re
# --- Load Environment ---
load_dotenv()

# --- Import the OpenRouter caller from your *other* file ---
# This assumes your file is in a package, e.g., 'utils/summarizer.py'
# If it's in the same flat directory, you might use:
# from summarizer import _safe_openrouter_call
try:
    from utils.summarizer import _safe_openrouter_call
except ImportError:
    import summarizer as summarizer
    _safe_openrouter_call = summarizer._safe_openrouter_call
# --- Model Definitions ---
# We need a good model for JSON-based sentence splitting.
# We'll default to the 'SUMMARIZE_MODEL' from your other file if not set.
FALLBACK_MODEL = os.getenv("OPENROUTER_SUMMARIZE_MODEL", "openai/gpt-oss-20b:free")
SEGMENT_MODEL = os.getenv("OPENROUTER_SEGMENT_MODEL", FALLBACK_MODEL)


# --- Whisper Model (Globally defined, not loaded) ---
_WHISPER_MODEL: Optional[WhisperModel] = None


def load_models(whisper_model_name: Optional[str] = None):
    """
    Explicitly loads the Whisper model into memory.
    (SpaCy is no longer needed)
    """
    global _WHISPER_MODEL

    if _WHISPER_MODEL is None:
        model_name = whisper_model_name or os.getenv("WHISPER_MODEL", "base")
        try:
            print(f"Loading Whisper model: {model_name}")
            _WHISPER_MODEL = WhisperModel(model_name, device="cuda", compute_type="float16")
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

# --- Helper Functions (no changes) ---

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def build_srt(sentences: List[Dict]) -> str:
    """Convert sentences to SRT string"""
    srt_lines = []
    for idx, sent in enumerate(sentences, start=1):
        start_time = seconds_to_srt_time(sent["start"])
        end_time = seconds_to_srt_time(sent["end"])
        srt_lines.append(f"{idx}\n{start_time} --> {end_time}\n{sent['text']}\n")
    return "\n".join(srt_lines)


# --- NEW LLM-based Sentence Segmenter ---

def _segment_text_with_llm(full_text: str) -> List[str]:
    """
    Uses an LLM to segment raw text into a list of clean sentences.
    """
    if not _safe_openrouter_call:
        raise Exception("OpenRouter call function is not available.")

    # This prompt is critical. It asks for JSON and forbids word changes.
    prompt = (
        "You are an expert text editor. Your task is to take the following "
        "raw, messy block of transcript text and format it.\n"
        "1. Break the text into complete, logical sentences.\n"
        "2. Add correct punctuation (periods, question marks, exclamation marks) to the end of each sentence.\n"
        "3. CRITICAL: DO NOT change any words or fix spelling mistakes. Keep ALL words EXACTLY as they appear in the transcript, even if they seem misspelled.\n"
        "4. Only add punctuation and sentence breaks. Do not modify, add, or remove any words.\n"
        "5. Return your response *only* as a valid JSON object with a single key "
        "'sentences', which holds a list of the sentence strings.\n\n"
        "EXAMPLE INPUT:\n"
        "hello how are you it's a great day I hope you're doing well\n"
        "EXAMPLE OUTPUT:\n"
        '{"sentences": ["hello how are you.", "it\'s a great day.", "I hope you\'re doing well."]}\n\n'
        "The output sentences must not be limited to any particular number.\n\n"
        "TRANSCRIPT:\n"
        f"{full_text}"
    )

    messages = [{"role": "user", "content": prompt}]
    
    print(f"Requesting sentence segmentation from LLM: {SEGMENT_MODEL}")
    content = _safe_openrouter_call(SEGMENT_MODEL, messages)
    with open("D:\\major-project\\aaa\\llm_segment_response.txt", "w", encoding="utf-8") as f:
        f.write(content or "No response")
    
    if not content:
        print("Warning: LLM call failed. Falling back to simple sentence heuristics.")
        # Try a naive punctuation-based split first
        naive = re.split(r"(?<=[.!?])\s+", full_text.strip())
        naive = [s.strip() for s in naive if s and s.strip()]
        if len(naive) > 1:
            return naive
        # As a last resort, split by newlines (may still be one sentence if no newlines)
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        return lines if lines else [full_text.strip()]

    try:
        # Try to parse the JSON response
        # It might be inside a code block, so we clean it first
        clean_content = content.strip().lstrip("```json").lstrip("```").rstrip("```")
        parsed = json.loads(clean_content)
        
        if "sentences" in parsed and isinstance(parsed["sentences"], list):
            print(f"LLM successfully segmented text into {len(parsed['sentences'])} sentences.")
            return parsed["sentences"]
        else:
            raise Exception("JSON missing 'sentences' key or value is not a list.")
            
    except Exception as e:
        print(f"Warning: Could not parse LLM JSON response: {e}")
        print(f"Raw response: {content}")
        # Fallback: naive punctuation-based split
        naive = re.split(r"(?<=[.!?])\s+", full_text.strip())
        naive = [s.strip() for s in naive if s and s.strip()]
        if len(naive) > 0:
            return naive
        # Last resort: split by newlines
        return [line.strip() for line in full_text.split("\n") if line.strip()]

def _segment_text_fallback_by_pauses(words_data: List[Dict], max_pause: float = 0.8, max_words: int = 30) -> List[str]:
    """
    Heuristic segmentation when LLM segmentation fails or returns a single giant sentence.
    - Starts a new sentence when the pause between consecutive words exceeds `max_pause` seconds,
      or when the current sentence reaches `max_words` words.
    - Returns a list of sentence texts composed of the original words (no spelling changes).
    """
    if not words_data:
        return []

    sentences: List[str] = []
    current_words: List[str] = []
    prev_end: Optional[float] = None

    for w in words_data:
        word_text = (w.get("word") or "").strip()
        if not word_text:
            continue

        # Decide if we should break before adding this word
        if prev_end is not None:
            gap = float(w.get("start", prev_end)) - float(prev_end)
            if gap > max_pause or len(current_words) >= max_words:
                if current_words:
                    sentences.append(" ".join(current_words).strip())
                    current_words = []

        current_words.append(word_text)
        prev_end = float(w.get("end", prev_end if prev_end is not None else 0.0))

    if current_words:
        sentences.append(" ".join(current_words).strip())

    # Filter any empties
    return [s for s in sentences if s]

def _fix_spelling_with_llm(sentences: List[Dict]) -> List[Dict]:
    """
    Post-processing step: Fixes spelling errors in sentence text while preserving timestamps.
    This is done AFTER alignment to avoid breaking word matching.
    """
    if not _safe_openrouter_call:
        print("Warning: Cannot fix spelling - OpenRouter unavailable.")
        return sentences
    
    # Batch process all sentences for efficiency
    sentence_texts = [s["text"] for s in sentences]
    combined_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(sentence_texts)])
    
    prompt = (
        "You are an expert editor. Fix spelling and grammar errors in these sentences.\n"
        "IMPORTANT RULES:\n"
        "1. Fix spelling mistakes and grammar errors\n"
        "2. Maintain the same sentence structure and meaning\n"
        "3. Keep the sentence numbers\n"
        "4. Return ONLY a JSON object with key 'sentences' containing a list of corrected sentences\n"
        "5. Each sentence should be complete and properly capitalized\n\n"
        "EXAMPLE INPUT:\n"
        "1. BubbleSoar is one of the most populer algorithms.\n"
        "2. It has terrable performance.\n"
        "EXAMPLE OUTPUT:\n"
        '{"sentences": ["BubbleSort is one of the most popular algorithms.", "It has terrible performance."]}\n\n'
        "SENTENCES TO FIX:\n"
        f"{combined_text}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    print(f"Fixing spelling with LLM: {SEGMENT_MODEL}")
    content = _safe_openrouter_call(SEGMENT_MODEL, messages)
    
    if not content:
        print("Warning: LLM spelling correction failed. Keeping original text.")
        return sentences
    
    try:
        clean_content = content.strip().lstrip("```json").lstrip("```").rstrip("```")
        parsed = json.loads(clean_content)
        
        if "sentences" in parsed and isinstance(parsed["sentences"], list):
            corrected = parsed["sentences"]
            if len(corrected) == len(sentences):
                # Update the text while keeping timestamps and words intact
                for i, sent in enumerate(sentences):
                    sent["text"] = corrected[i]
                print(f"Successfully corrected spelling for {len(sentences)} sentences.")
            else:
                print(f"Warning: Mismatch in sentence count. Expected {len(sentences)}, got {len(corrected)}. Keeping original.")
        else:
            print("Warning: Invalid JSON response for spelling correction.")
    except Exception as e:
        print(f"Warning: Could not parse spelling correction response: {e}")
    
    return sentences

def _clean_word(word: str) -> str:
    """
    Normalizes a word for robust matching.
    - Converts to lowercase.
    - Removes all punctuation except for internal apostrophes.
    """
    # 1. Lowercase the word
    word = word.lower()
    
    # 2. Remove all punctuation using regex, *except* apostrophes
    # This keeps "don't" as "don't" and "it's" as "it's"
    # but turns "word." into "word" and "(word)" into "word"
    word = re.sub(r"^[^\w']+|[^\w']+$", "", word)
    
    # 3. Handle specific cases if needed, but this is usually enough.
    # For example, Whisper sometimes produces "don't" and LLM "dont".
    # A simpler, more aggressive rule:
    word = re.sub(r"[^\w\s']|_", "", word) # Keep only letters, numbers, spaces, apostrophes
    word = word.strip("'") # Remove any leading/trailing apostrophes
    
    return word
# --- UPDATED Alignment Function (No spaCy) ---

def _realign_sentences_to_words(
    sentences_list: List[str], 
    words_data: List[Dict]
) -> List[Dict]:
    """
    Realigns a list of sentences (from an LLM) with Whisper's word timestamps.
    """
    sentences = []
    word_idx = 0 # This is the main "high-water mark"

    for sent_text in sentences_list:
        if not sent_text:
            continue

        # Get the list of words in the LLM's sentence
        sent_words = sent_text.split()
        sent_start, sent_end = None, None
        temp_word_idx = word_idx # This is the search-start for *this sentence*
        matched_words = []

        for target_word in sent_words:
            found_match = False
            
            # Create a "search window" to prevent a catastrophic failure.
            # We'll look ahead a maximum of 20 words from our current spot
            # to find a match. This helps resync if the LLM skips a word.
            search_window_end = min(temp_word_idx + 20, len(words_data))

            for i in range(temp_word_idx, search_window_end):
                
                # --- THIS IS THE FIX ---
                # Use the new, more robust cleaning function
                whisper_word_cleaned = _clean_word(words_data[i]["word"])
                target_word_cleaned = _clean_word(target_word)
                # -----------------------

                if whisper_word_cleaned == target_word_cleaned and whisper_word_cleaned != "":
                    if sent_start is None:
                        sent_start = words_data[i]["start"] # Set start time
                    
                    sent_end = words_data[i]["end"] # *Always* update end time
                    matched_words.append(words_data[i])
                    temp_word_idx = i + 1  # Advance the search position for the *next word*
                    found_match = True
                    break # Found the match, move to the next target_word
            
            if not found_match:
                # print(f"Warning: Could not match target word: '{target_word}'")
                pass # Just skip this word and hope to resync on the next one

        # --- End of Sentence ---
        if sent_start is not None and sent_end is not None:
            sentences.append({
                "text": sent_text, # Use the LLM's clean, punctuated text
                "start": sent_start,
                "end": sent_end,
                "words": matched_words
            })
            # --- THIS IS THE CRITICAL LINE ---
            # Lock in the progress. The *next sentence* must start its search
            # from *after* the last word we just matched.
            word_idx = temp_word_idx 
        
        # else:
            # This sentence had 0 matched words.
            # We don't advance word_idx, so the next sentence
            # can try to match from the same spot.

    return sentences

# --- Main Transcribe Function (Updated) ---

def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe audio, segment sentences with an LLM, and realign timestamps.
    """
    if not _WHISPER_MODEL:
        raise Exception("Whisper model is not loaded. Please call load_models() first.")
    
    if not _safe_openrouter_call:
        raise Exception("OpenRouter call function not loaded. Cannot segment sentences.")

    try:
        # --- STAGE 1: Whisper Transcription (Same as before) ---
        segments, info = _WHISPER_MODEL.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        
        all_text = []
        words_data = []
        for segment in segments:
            all_text.append(segment.text)
            if segment.words:
                for word in segment.words:
                    words_data.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                    })

        transcription_data = {
            "text": " ".join(all_text).strip(),
            "words": words_data,
            "language": info.language
        }
        with open("D:\\major-project\\aaa\\transcription.json", "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=4)
        if not transcription_data["text"]:
            print("No speech detected in audio.")
            return {"sentences": [], "srt": "", "language": "unknown"}

        # --- STAGE 2: LLM Sentence Segmentation (without spelling fixes) ---
        llm_sentences = _segment_text_with_llm(transcription_data["text"])
        # Rescue fallback: if segmentation produced a single very long sentence,
        # or nothing at all, split by pauses to avoid treating entire transcript as one sentence.
        if not llm_sentences or (len(llm_sentences) == 1 and len(llm_sentences[0].split()) > 40):
            print("Segmentation fallback: splitting by pauses and length heuristics")
            llm_sentences = _segment_text_fallback_by_pauses(transcription_data["words"], max_pause=0.8, max_words=30)
        with open("D:\\major-project\\aaa\\llm_sentences.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(llm_sentences))
        
        # --- STAGE 3: Re-alignment (with original spelling) ---
        sentences = _realign_sentences_to_words(llm_sentences, transcription_data["words"])
        with open("D:\\major-project\\aaa\\realigned_sentences.json", "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=4)
        # Preserve a copy of the original sentence text before any spelling correction
        for s in sentences:
            if "original_text" not in s:
                s["original_text"] = s.get("text", "")
        
        # --- STAGE 4: Fix Spelling (AFTER alignment to preserve timestamps) ---
        # Only attempt spelling correction on the final sentence texts; timestamps and per-word data remain unchanged.
        sentences = _fix_spelling_with_llm(sentences)
        with open("D:\\major-project\\aaa\\spelling_corrected_sentences.json", "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=4)
        
        # --- STAGE 5: Build SRT ---
        srt_text = build_srt(sentences)
        with open("D:\\major-project\\Outputs\\transcript.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([f"{sentence['text']},{sentence['start']},{sentence['end']}" for sentence in sentences]))
        print("Transcription and LLM segmentation is complete")
        return {
            "sentences": sentences,
            "srt": srt_text,
            "language": info.language
        }
        
    except Exception as e:
        print(f"Error During Transcription: {e}")
        raise

if __name__ == "__main__":
    load_models()
    results = transcribe_audio("D:\\major-project\\Inputs\\Video.mp4")
