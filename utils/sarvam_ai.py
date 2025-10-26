"""
Sarvam AI Integration Module
============================
Handles multilingual translation and text-to-speech using Sarvam AI API.
Supports all Indian languages that Sarvam AI offers.

Author: VidBrain Team
Date: October 2025
"""

import os
import logging
import requests
import base64
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

class SarvamAI:
    """Sarvam AI client for translation and TTS services"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("Sarvam API key is required. Set SARVAM_API_KEY environment variable.")
        
        self.base_url = "https://api.sarvam.ai"
        self.headers = {
            "Content-Type": "application/json",
            "API-Subscription-Key": self.api_key
        }
        
        # Sarvam AI supported languages with their codes and names
        self.supported_languages = {
            "hi": "Hindi (हिंदी)",
            "bn": "Bengali (বাংলা)", 
            "ta": "Tamil (தமিழ્)",
            "te": "Telugu (తెలుగు)",
            "ml": "Malayalam (മലയാളം)",
            "kn": "Kannada (ಕನ್ನಡ)",
            "gu": "Gujarati (ગુજરાતી)",
            "mr": "Marathi (मराठी)",
            "pa": "Punjabi (ਪੰਜਾਬੀ)",
            "or": "Odia (ଓଡ଼ିଆ)",
            "as": "Assamese (অসমীয়া)",
            "en": "English"
        }
        
        # Mapping from our simple codes to Sarvam API region-specific codes
        self.sarvam_language_map = {
            "hi": "hi-IN",
            "bn": "bn-IN", 
            "ta": "ta-IN",
            "te": "te-IN",
            "ml": "ml-IN",
            "kn": "kn-IN",
            "gu": "gu-IN",
            "mr": "mr-IN",
            "pa": "pa-IN",
            "or": "od-IN",  # Note: Sarvam uses 'od-IN' for Odia
            "as": "as-IN",
            "en": "en-IN"
        }
        
        logger.info(f"Sarvam AI initialized with {len(self.supported_languages)} supported languages")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages with their codes and names"""
        return self.supported_languages.copy()
    
    def _chunk_text(self, text: str, max_chars: int = 800) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentences first
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 2 > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If any chunk is still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                current_word_chunk = ""
                for word in words:
                    if current_word_chunk and len(current_word_chunk) + len(word) + 1 > max_chars:
                        final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word + " "
                    else:
                        current_word_chunk += word + " "
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())
        
        return final_chunks

    def translate_text(self, text: str, target_language: str, source_language: str = "en") -> str:
        """
        Translate text using Sarvam AI Translation API with automatic chunking for long texts
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'hi', 'ta', 'bn')
            source_language: Source language code (default: 'en')
        
        Returns:
            Translated text
        """
        if target_language not in self.supported_languages:
            raise ValueError(f"Language '{target_language}' not supported. Supported: {list(self.supported_languages.keys())}")
        
        if target_language == source_language:
            logger.info(f"Source and target languages are the same ({target_language}), returning original text")
            return text
        
        # Check if text needs chunking (Sarvam API limit ~800 chars for reliable translation)
        if len(text) > 800:
            logger.info(f"📝 Text is {len(text)} chars, chunking for translation...")
            chunks = self._chunk_text(text, 800)
            logger.info(f"📦 Split into {len(chunks)} chunks")
            
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🌐 Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                translated_chunk = self._translate_single_chunk(chunk, target_language, source_language)
                translated_chunks.append(translated_chunk)
            
            full_translation = " ".join(translated_chunks)
            logger.info(f"✅ Full translation completed: {len(full_translation)} chars")
            logger.info(f"📝 Full translated text (first 100 chars): {full_translation[:100]}...")
            logger.info(f"📝 Full translated text (last 100 chars): ...{full_translation[-100:]}")
            
            return full_translation
        else:
            return self._translate_single_chunk(text, target_language, source_language)

    def _translate_single_chunk(self, text: str, target_language: str, source_language: str = "en") -> str:
        """
        Translate a single chunk of text (internal method)
        """
        # Convert our simple language codes to Sarvam API region-specific codes
        sarvam_source = self.sarvam_language_map.get(source_language, source_language)
        sarvam_target = self.sarvam_language_map.get(target_language, target_language)
        
        url = f"{self.base_url}/translate"
        payload = {
            "input": text,
            "source_language_code": sarvam_source,  # Use region-specific code
            "target_language_code": sarvam_target,  # Use region-specific code
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1"
        }
        
        try:
            logger.info(f"Using Sarvam codes: {sarvam_source} → {sarvam_target}")
            logger.info(f"Chunk to translate ({len(text)} chars): {text[:100]}...")
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result.get("translated_text", "")
            
            if not translated_text:
                logger.error(f"Translation failed: {result}")
                raise ValueError("Translation returned empty text")
            
            logger.info(f"✅ Chunk translation successful ({len(translated_text)} chars)")
            
            return translated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam AI translation API error: {e}")
            raise ValueError(f"Translation failed: {e}")
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
    
    def generate_speech(self, text: str, language: str, speaker: str = "anushka") -> bytes:
        """
        Generate speech using Sarvam AI TTS API with automatic chunking for long texts
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'hi', 'ta', 'bn')
            speaker: Speaker name (default: 'anushka') - valid for v2: 'anushka', 'abhilash', 'manisha', 'vidya', etc.
        
        Returns:
            Audio data as bytes
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language '{language}' not supported for TTS. Supported: {list(self.supported_languages.keys())}")
        
        # Check if text needs chunking (Sarvam API limit ~500 chars for reliable TTS)
        if len(text) > 500:
            logger.info(f"🎤 Text is {len(text)} chars, chunking for TTS...")
            chunks = self._chunk_text(text, 500)  # Smaller chunks for TTS
            logger.info(f"📦 Split into {len(chunks)} chunks for TTS")
            
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🎤 Generating speech for chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                chunk_audio = self._generate_speech_single_chunk(chunk, language, speaker)
                audio_chunks.append(chunk_audio)
            
            # Concatenate all audio chunks
            full_audio = b''.join(audio_chunks)
            logger.info(f"✅ Full TTS completed: {len(full_audio)} bytes from {len(chunks)} chunks")
            
            return full_audio
        else:
            return self._generate_speech_single_chunk(text, language, speaker)

    def _generate_speech_single_chunk(self, text: str, language: str, speaker: str = "anushka") -> bytes:
        """
        Generate speech for a single chunk of text (internal method)
        """
        # Convert our simple language code to Sarvam API region-specific code
        sarvam_language = self.sarvam_language_map.get(language, language)
        
        # Valid speaker names for bulbul:v2 model (these work for both v2 and v3-beta)
        valid_speakers = [
            'anushka', 'abhilash', 'manisha', 'vidya', 'arya', 'karun', 'hitesh', 'aditya',
            'isha', 'ritu', 'chirag', 'harsh', 'sakshi', 'priya', 'neha', 'rahul', 'pooja',
            'rohan', 'simran', 'kavya', 'anjali', 'sneha', 'kiran', 'vikram', 'rajesh',
            'sunita', 'tara', 'anirudh', 'kriti', 'ishaan'
        ]
        
        if speaker not in valid_speakers:
            logger.warning(f"Speaker '{speaker}' not valid, using 'anushka' instead")
            speaker = "anushka"
        
        url = f"{self.base_url}/text-to-speech"
        payload = {
            "inputs": [text],
            "target_language_code": sarvam_language,  # Use region-specific code
            "speaker": speaker,
            "pace": 1.0,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v2"  # Use stable v2 model (no beta access required)
        }
        
        try:
            logger.info(f"Using Sarvam code: {sarvam_language}")
            logger.info(f"Chunk for TTS ({len(text)} chars): {text[:100]}...")
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=120)  # Increased timeout for TTS
            response.raise_for_status()
            
            result = response.json()
            audio_base64 = result.get("audios", [None])[0]
            
            if not audio_base64:
                logger.error(f"TTS failed: {result}")
                raise ValueError("TTS returned no audio data")
            
            # Decode base64 audio data
            audio_data = base64.b64decode(audio_base64)
            
            logger.info(f"✅ Chunk TTS successful - Generated {len(audio_data)} bytes of audio")
            
            return audio_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam AI TTS API error: {e}")
            raise ValueError(f"TTS generation failed: {e}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
    
    def translate_and_generate_speech(self, text: str, target_language: str, source_language: str = "en") -> Tuple[str, bytes]:
        """
        Translate text and generate speech in one call
        
        Args:
            text: Text to translate and convert to speech
            target_language: Target language code
            source_language: Source language code (default: 'en')
        
        Returns:
            Tuple of (translated_text, audio_data)
        """
        logger.info(f"🌐 Starting translation and TTS pipeline: {source_language} → {target_language}")
        
        # Step 1: Translate (if needed)
        if target_language != source_language:
            translated_text = self.translate_text(text, target_language, source_language)
        else:
            translated_text = text
        
        # Step 2: Generate speech
        audio_data = self.generate_speech(translated_text, target_language)
        
        logger.info(f"🎉 Translation and TTS pipeline completed successfully")
        
        return translated_text, audio_data
    
    def save_audio(self, audio_data: bytes, file_path: str) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio bytes
            file_path: Path to save the file
        
        Returns:
            Path to saved file
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"💾 Audio saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

# Convenience functions for easy integration
def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages"""
    sarvam = SarvamAI()
    return sarvam.get_supported_languages()

def translate_summary(summary: str, target_language: str, source_language: str = "en") -> str:
    """Translate summary text"""
    sarvam = SarvamAI()
    return sarvam.translate_text(summary, target_language, source_language)

def generate_multilingual_voiceover(summary: str, target_language: str, output_path: str, source_language: str = "en") -> Tuple[str, str]:
    """
    Generate multilingual voiceover
    
    Returns:
        Tuple of (translated_text, audio_file_path)
    """
    sarvam = SarvamAI()
    translated_text, audio_data = sarvam.translate_and_generate_speech(summary, target_language, source_language)
    audio_path = sarvam.save_audio(audio_data, output_path)
    return translated_text, audio_path

# Test function
def test_sarvam_integration():
    """Test Sarvam AI integration"""
    logger.info("🧪 Testing Sarvam AI integration...")
    
    try:
        sarvam = SarvamAI()
        languages = sarvam.get_supported_languages()
        
        logger.info(f"✅ Supported languages: {len(languages)}")
        for code, name in languages.items():
            logger.info(f"   {code}: {name}")
        
        # Test translation
        test_text = "Hello, this is a test message for translation."
        translated = sarvam.translate_text(test_text, "hi", "en")
        logger.info(f"✅ Translation test successful: {translated}")
        
        # Test TTS (small text)
        audio_data = sarvam.generate_speech("नमस्ते", "hi")
        logger.info(f"✅ TTS test successful: {len(audio_data)} bytes")
        
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_sarvam_integration()