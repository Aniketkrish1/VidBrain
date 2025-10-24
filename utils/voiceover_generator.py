"""
utils/voiceover_generator.py

Generate high-quality voiceovers from text summaries.
Supports pyttsx3 (offline) and edge-tts (online, better quality).
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import asyncio

logger = logging.getLogger(__name__)

# Try to import both TTS engines
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not available")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.info("edge-tts not available (optional, for better quality)")


class VoiceoverGenerator:
    """Generate voiceovers from text using available TTS engines."""
    
    def __init__(self, engine: str = "auto", voice: str = None):
        """
        Initialize voiceover generator.
        
        Args:
            engine: TTS engine to use ("pyttsx3", "edge-tts", or "auto")
            voice: Voice to use (engine-specific)
        """
        self.engine = engine
        self.voice = voice
        
        if engine == "auto":
            if EDGE_TTS_AVAILABLE:
                self.engine = "edge-tts"
                logger.info("Using edge-tts for high-quality voiceovers")
            elif PYTTSX3_AVAILABLE:
                self.engine = "pyttsx3"
                logger.info("Using pyttsx3 for voiceovers")
            else:
                raise RuntimeError("No TTS engine available")
        
        logger.info(f"VoiceoverGenerator initialized with engine: {self.engine}")
    
    def generate_voiceover_pyttsx3(self, text: str, output_path: str) -> str:
        """
        Generate voiceover using pyttsx3 (offline).
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        if not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 not available")
        
        logger.info(f"Generating voiceover with pyttsx3: {len(text)} chars")
        
        try:
            engine = pyttsx3.init()
            
            # Configure voice settings
            voices = engine.getProperty('voices')
            if voices:
                # Try to use a good quality voice
                if self.voice:
                    # Use specified voice
                    for voice in voices:
                        if self.voice.lower() in voice.name.lower() or self.voice in voice.id:
                            engine.setProperty('voice', voice.id)
                            break
                else:
                    # Try to find a female voice (often clearer)
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
            
            # Set speech properties for clarity
            engine.setProperty('rate', 165)  # Slower, clearer speech
            engine.setProperty('volume', 0.95)
            
            # Generate audio
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            engine.stop()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Voiceover created: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                raise RuntimeError("Audio file not created or empty")
                
        except Exception as e:
            logger.error(f"pyttsx3 generation failed: {e}")
            raise
        finally:
            try:
                engine.stop()
                del engine
            except:
                pass
    
    async def generate_voiceover_edge_async(self, text: str, output_path: str) -> str:
        """
        Generate voiceover using edge-tts (online, better quality).
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not available")
        
        logger.info(f"Generating voiceover with edge-tts: {len(text)} chars")
        
        try:
            # Use a high-quality voice
            voice = self.voice or "en-US-AriaNeural"  # Clear, professional female voice
            
            # Create TTS communicate object
            communicate = edge_tts.Communicate(text, voice)
            
            # Save audio
            await communicate.save(output_path)
            
            # Verify file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Voiceover created: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                raise RuntimeError("Audio file not created or empty")
                
        except Exception as e:
            logger.error(f"edge-tts generation failed: {e}")
            raise
    
    def generate_voiceover_edge(self, text: str, output_path: str) -> str:
        """
        Synchronous wrapper for edge-tts generation.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async function
            return loop.run_until_complete(
                self.generate_voiceover_edge_async(text, output_path)
            )
        except Exception as e:
            logger.error(f"Failed to run edge-tts async: {e}")
            raise
    
    def generate(self, text: str, output_path: str) -> str:
        """
        Generate voiceover using configured engine.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Clean text for better speech
        text = self._clean_text(text)
        
        if self.engine == "edge-tts":
            return self.generate_voiceover_edge(text, output_path)
        elif self.engine == "pyttsx3":
            return self.generate_voiceover_pyttsx3(text, output_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better speech synthesis.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove markdown formatting that might be in summaries
        text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        
        # Ensure text ends with proper punctuation
        if text and text[-1] not in ".!?":
            text += "."
        
        return text


def generate_voiceover(text: str, 
                      output_path: str, 
                      engine: str = "auto",
                      voice: str = None) -> str:
    """
    Convenience function to generate a single voiceover.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file
        engine: TTS engine to use ("pyttsx3", "edge-tts", or "auto")
        voice: Optional specific voice to use
    
    Returns:
        Path to generated audio file
    """
    generator = VoiceoverGenerator(engine=engine, voice=voice)
    return generator.generate(text, output_path)


def generate_multiple_voiceovers(texts: List[str], 
                                 output_dir: str = "temp_processing",
                                 prefix: str = "voiceover",
                                 engine: str = "auto") -> List[str]:
    """
    Generate multiple voiceovers efficiently.
    
    Args:
        texts: List of texts to convert
        output_dir: Directory to save audio files
        prefix: Filename prefix
        engine: TTS engine to use
    
    Returns:
        List of paths to generated audio files
    """
    logger.info(f"Generating {len(texts)} voiceovers")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = VoiceoverGenerator(engine=engine)
    
    audio_paths = []
    
    for idx, text in enumerate(texts):
        if not text or not text.strip():
            logger.warning(f"Skipping empty text at index {idx}")
            continue
        
        output_path = os.path.join(output_dir, f"{prefix}_{idx}.mp3")
        
        try:
            audio_path = generator.generate(text, output_path)
            audio_paths.append(audio_path)
            logger.info(f"Generated voiceover {idx+1}/{len(texts)}")
        except Exception as e:
            logger.error(f"Failed to generate voiceover {idx}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(audio_paths)}/{len(texts)} voiceovers")
    return audio_paths


if __name__ == "__main__":
    # Test voiceover generation
    test_text = "This is a test of the voiceover generation system. It should create clear, natural-sounding speech."
    test_output = "test_voiceover.mp3"
    
    try:
        print("Testing voiceover generation...")
        result = generate_voiceover(test_text, test_output)
        print(f"Success! Generated: {result}")
    except Exception as e:
        print(f"Failed: {e}")
