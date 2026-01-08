"""
utils/translator.py

Multilingual translation and TTS using Sarvam AI for translation and gTTS for TTS.
Provides safe fallbacks to ensure pipeline stability.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load .env from project root (parent of utils/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# Supported language codes for Sarvam AI (translation)
SUPPORTED_LANGUAGES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "bengali": "bn-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "odia": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "gujarati": "gu-IN"
}

# gTTS language codes (for TTS)
GTTS_LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",  # gTTS uses 'or' for Odia
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "gujarati": "gu"
}

def get_language_code(language: str) -> Optional[str]:
    """
    Get Sarvam AI language code from user-friendly language name.
    Returns None if language not supported.
    """
    if not language:
        return None
    lang = language.lower().strip()
    # If user provided full Sarvam code already (e.g., 'hi-IN'), accept it
    if lang in SUPPORTED_LANGUAGES.values():
        return lang
    # If user provided 2-letter code (e.g., 'hi'), try to map to the supported value
    for name, code in SUPPORTED_LANGUAGES.items():
        if lang == code.split('-')[0]:
            return code
    # Otherwise, treat input as name (e.g., 'hindi')
    return SUPPORTED_LANGUAGES.get(lang)

def translate_text(
    text: str,
    target_language: str,
    source_language: str = "en-IN"
) -> str:
    """
    Translate text using Sarvam AI.
    
    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'hi-IN', 'ta-IN') or name (e.g., 'hindi', 'tamil')
        source_language: Source language code (default: 'en-IN')
    
    Returns:
        Translated text, or original text if translation fails
    """
    if not text or not text.strip():
        return text
    
    # Skip translation if target is English
    target_code = get_language_code(target_language) if target_language not in SUPPORTED_LANGUAGES.values() else target_language
    if not target_code or target_code == "en-IN":
        logger.info("Target language is English or not specified, skipping translation")
        return text
    
    if not SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set, skipping translation")
        return text
    
    try:
        headers = {
            "Content-Type": "application/json",
            "API-Subscription-Key": SARVAM_API_KEY
        }
        
        payload = {
            "input": text,
            "source_language_code": source_language,
            "target_language_code": target_code,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True
        }
        
        logger.info(f"Translating text to {target_code}")
        response = requests.post(
            SARVAM_TRANSLATE_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            translated = result.get("translated_text", text)
            logger.info(f"Translation successful: {len(text)} -> {len(translated)} chars")
            return translated
        else:
            logger.warning(f"Translation API returned {response.status_code}: {response.text}")
            return text
            
    except requests.exceptions.Timeout:
        logger.warning("Translation request timed out, using original text")
        return text
    except Exception as e:
        logger.warning(f"Translation failed: {e}, using original text")
        return text

def translate_batch(
    texts: List[str],
    target_language: str,
    source_language: str = "en-IN"
) -> List[str]:
    """
    Translate multiple texts. Processes one at a time for reliability.
    
    Args:
        texts: List of texts to translate
        target_language: Target language code or name
        source_language: Source language code (default: 'en-IN')
    
    Returns:
        List of translated texts (original text on failure)
    """
    translated = []
    for text in texts:
        translated.append(translate_text(text, target_language, source_language))
    return translated

def generate_tts_gtts(
    text: str,
    output_path: str,
    target_language: str = "english"
) -> bool:
    """
    Generate speech using gTTS (Google Text-to-Speech).
    More reliable than Sarvam AI TTS.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file (.mp3)
        target_language: Language name (e.g., 'hindi', 'kannada', 'english')
    
    Returns:
        True if successful, False otherwise
    """
    if not text or not text.strip():
        logger.warning("Empty text for TTS generation")
        return False
    
    try:
        from gtts import gTTS
        
        # Resolve target_language which may be a name ('hindi'), a short code ('hi'), or a full code ('hi-IN')
        t = target_language.lower() if target_language else "english"
        # If input is already a GTTS code (2-letter), use it
        if t in GTTS_LANGUAGE_CODES.values():
            lang_code = t
        else:
            # If they passed a 2-letter code like 'hi', map by matching keys' values prefix
            mapped = None
            for name, code in SUPPORTED_LANGUAGES.items():
                if t == code.split('-')[0]:
                    mapped = GTTS_LANGUAGE_CODES.get(name, None)
                    break
            if mapped:
                lang_code = mapped
            else:
                # Try mapping from friendly name
                lang_code = GTTS_LANGUAGE_CODES.get(t, "en")
        
        logger.info(f"Generating TTS for '{target_language}' (code: {lang_code}) using gTTS")
        
        # Create gTTS object
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to file (gTTS saves as MP3)
        tts.save(output_path)
        
        logger.info(f"TTS generated successfully: {output_path}")
        return True
        
    except ImportError:
        logger.error("gTTS not installed. Install with: pip install gtts")
        return False
    except Exception as e:
        logger.error(f"gTTS generation failed: {e}")
        return False

def generate_tts_sarvam(
    text: str,
    output_path: str,
    target_language: str = "en-IN",
    speaker: str = "meera"
) -> bool:
    """
    Generate speech using Sarvam AI TTS.
    DEPRECATED: Kept for backwards compatibility, but gTTS is recommended.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file
        target_language: Language code or name
        speaker: Voice speaker name (default: 'meera')
    
    Returns:
        True if successful, False otherwise
    """
    logger.warning("Sarvam TTS is deprecated. Please use generate_tts_gtts() instead.")
    return False  # Disabled, fallback to gTTS

def is_translation_available() -> bool:
    """Check if translation service is available."""
    return bool(SARVAM_API_KEY)

def get_supported_languages() -> Dict[str, str]:
    """Get dictionary of supported languages."""
    return SUPPORTED_LANGUAGES.copy()
