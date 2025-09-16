from faster_whisper import WhisperModel
from typing import Dict, List
import os
import datetime
from dotenv import load_dotenv
import spacy

load_dotenv()
whisper_model = os.getenv("WHISPER_MODEL")
spacy_model = os.getenv("SPACY_MODEL", "en_core_web_sm")


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def segment_and_realign_sentences(transcription_data: Dict) -> List[Dict]:
    """Segment transcript into sentences and realign word timestamps with spaCy"""
    nlp = spacy.load(spacy_model)
    doc = nlp(transcription_data["text"])

    sentences = []
    words = transcription_data["words"]
    word_idx = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sent_words = sent_text.lower().split()
        sent_start, sent_end = None, None
        temp_word_idx = word_idx
        matched_words = []

        for target_word in sent_words:
            for i in range(temp_word_idx, len(words)):
                if target_word in words[i]["word"].lower():
                    if sent_start is None:
                        sent_start = words[i]["start"]
                    sent_end = words[i]["end"]
                    matched_words.append(words[i])
                    temp_word_idx = i + 1
                    break

        if sent_start is not None and sent_end is not None:
            sentences.append({
                "text": sent_text,
                "start": sent_start,
                "end": sent_end,
                "words": matched_words
            })
            word_idx = temp_word_idx

    return sentences


def build_srt(sentences: List[Dict]) -> str:
    """Convert sentences to SRT string"""
    srt_lines = []
    for idx, sent in enumerate(sentences, start=1):
        start_time = seconds_to_srt_time(sent["start"])
        end_time = seconds_to_srt_time(sent["end"])
        srt_lines.append(f"{idx}\n{start_time} --> {end_time}\n{sent['text']}\n")
    return "\n".join(srt_lines)


def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe audio using Faster-Whisper with word-level timestamps.
    Returns:
        {
          "sentences": [...],
          "srt": str,
          "language": str
        }
    """
    try:
        model = WhisperModel(whisper_model, device="cuda", compute_type="float16")

        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
        )

        all_text = []
        words_data = []
        segments_data = []

        for segment in segments:
            all_text.append(segment.text)

            seg_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": []
            }

            if segment.words:
                for word in segment.words:
                    word_entry = {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    seg_dict["words"].append(word_entry)
                    words_data.append(word_entry)

            segments_data.append(seg_dict)

        transcription_data = {
            "text": " ".join(all_text).strip(),
            "words": words_data,
            "segments": segments_data,
            "language": info.language
        }

        # Segment + build SRT
        sentences = segment_and_realign_sentences(transcription_data)
        srt_text = build_srt(sentences)

        print("Transcription is complete")
        return {
            "sentences": sentences,
            "srt": srt_text,
            "language": info.language
        }

    except Exception as e:
        print("Error During Transcription:", e)
        raise
