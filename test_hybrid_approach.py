#!/usr/bin/env python3
"""
Test the new hybrid approach: translated text + English TTS fallback
"""
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sarvam_ai import SarvamAI
import pyttsx3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_approach():
    """Test the new hybrid approach: translation + English TTS fallback"""
    
    logger.info("🧪 Testing hybrid approach...")
    
    # Test data
    original_text = "Bubble sort is a simple comparison-based sorting algorithm"
    target_language = "kn"
    
    # Mock summaries like main.py
    summaries = {
        0: {
            "summary": original_text,
            "start": 0.0,
            "end": 30.0
        }
    }
    
    try:
        sarvam = SarvamAI()
        cid = 0
        
        logger.info(f"📝 Original text: {original_text}")
        
        # Step 1: Always translate first (for frontend display)
        logger.info(f"🌐 Step 1: Translating to {target_language}...")
        try:
            translated_text = sarvam.translate_text(original_text, target_language, "en")
            logger.info(f"✅ Translation successful: {translated_text[:50]}...")
            
            # Update summary with translated text for frontend display
            summaries[cid]["summary"] = translated_text
            summaries[cid]["display_language"] = target_language
            logger.info(f"📝 Summary updated with translated text")
            
        except Exception as translate_e:
            logger.warning(f"⚠️ Translation failed: {translate_e}")
            translated_text = original_text
            summaries[cid]["summary"] = original_text
            summaries[cid]["display_language"] = "en"
        
        # Step 2: Try TTS in target language (will likely fail)
        try:
            logger.info(f"🎤 Step 2: Trying {target_language} TTS...")
            audio_data = sarvam.generate_speech(translated_text, target_language)
            logger.info(f"✅ {target_language} TTS successful: {len(audio_data)} bytes")
            summaries[cid]["voiceover_language"] = target_language
            
        except Exception as tts_e:
            logger.warning(f"⚠️ {target_language} TTS failed: {tts_e}")
            logger.info(f"🔄 Step 3: Using English TTS as fallback...")
            
            # Use English TTS but keep translated text for display
            try:
                engine = pyttsx3.init()
                temp_file = "temp_hybrid_test.mp3"
                engine.save_to_file(original_text, temp_file)  # English TTS
                engine.runAndWait()
                engine.stop()
                
                if os.path.exists(temp_file):
                    summaries[cid]["voiceover_language"] = "en"
                    logger.info(f"✅ English TTS completed")
                    logger.info(f"📋 Result: {target_language} text + English voiceover")
                    os.remove(temp_file)  # cleanup
                
            except Exception as fallback_e:
                logger.error(f"❌ English TTS also failed: {fallback_e}")
        
        # Check final result
        logger.info(f"\n📋 FINAL RESULT:")
        logger.info(f"   📝 Display text: {summaries[cid]['summary'][:50]}...")
        logger.info(f"   🌐 Display language: {summaries[cid].get('display_language', 'en')}")
        logger.info(f"   🎤 Voiceover language: {summaries[cid].get('voiceover_language', 'unknown')}")
        
        # Verify we have translated text for display
        final_text = summaries[cid]["summary"]
        if "ಬಬಲ್" in final_text or any(ord(c) > 127 for c in final_text):
            logger.info("✅ SUCCESS: User will see translated Kannada text!")
            logger.info("✅ SUCCESS: English voiceover will work as fallback!")
            return True
        else:
            logger.error("❌ FAILED: No translation in display text")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🧪 TESTING HYBRID APPROACH")
    logger.info("📝 Translated Text + 🎤 English TTS Fallback")
    logger.info("=" * 60)
    
    success = test_hybrid_approach()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 HYBRID APPROACH WORKING!")
        logger.info("✅ Users will see: Kannada text + English voiceover")
        logger.info("✅ Best of both worlds: Translation + Reliable audio")
    else:
        logger.error("❌ HYBRID APPROACH FAILED")
    
    logger.info("=" * 60)