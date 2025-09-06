from faster_whisper import WhisperModel
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()

whisper_model=os.getenv("WHISPER_MODEL")

def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe audio using Faster-Whisper with word-level timestamps.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing transcription results with word timestamps
    """
    # logger.info(f"Transcribing audio with Faster-Whisper model: {WHISPER_MODEL}")
    
    try:
        # Load Faster-Whisper model (GPU if available, else CPU)
        model = WhisperModel(whisper_model, device="cuda", compute_type="float16")

        # Run transcription
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5 ,
            
        )

        all_text = []
        words_data = []
        segments_data = []

        for segment in segments:
            all_text.append(segment.text)

            seg_dict = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
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

        result = {
            "text": " ".join(all_text).strip(),
            "words": words_data,
            "segments": segments_data,
            "language": info.language
        }

        # logger.info(f"Transcription complete. Found {len(segments_data)} segments")
        print("Transcription is complete")
        return result

    except Exception as e:
        # logger.error(f"Error during transcription: {e}")
        print("Error During Transcription",e)
        raise
