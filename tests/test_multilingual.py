"""
Test script for multilingual translation and TTS functionality.
Run this to verify Sarvam AI integration works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.translator import (
    translate_text,
    generate_tts_sarvam,
    get_supported_languages,
    is_translation_available,
    get_language_code
)

load_dotenv()

def test_translation():
    """Test text translation to various languages."""
    print("=" * 60)
    print("Testing Sarvam AI Translation")
    print("=" * 60)
    
    if not is_translation_available():
        print("❌ SARVAM_API_KEY not set in environment!")
        print("   Please add SARVAM_API_KEY to your .env file")
        return False
    
    test_text = "Hello, this is a test of the translation system. Sorting algorithms are fundamental to computer science."
    
    print(f"\nOriginal text: {test_text}\n")
    
    # Test a few languages
    test_languages = ["hindi", "tamil", "kannada"]
    
    for lang in test_languages:
        lang_code = get_language_code(lang)
        print(f"Translating to {lang.capitalize()} ({lang_code})...")
        
        translated = translate_text(test_text, lang)
        
        if translated and translated != test_text:
            print(f"✓ Success: {translated[:100]}...")
        else:
            print(f"✗ Failed or returned original text")
        print()
    
    return True

def test_tts():
    """Test TTS generation."""
    print("=" * 60)
    print("Testing Sarvam AI TTS")
    print("=" * 60)
    
    if not is_translation_available():
        print("❌ SARVAM_API_KEY not set!")
        return False
    
    test_text = "यह परीक्षण है"  # "This is a test" in Hindi
    output_path = "test_tts_output.wav"
    
    print(f"\nGenerating TTS for Hindi text: {test_text}")
    print(f"Output: {output_path}")
    
    success = generate_tts_sarvam(test_text, output_path, target_language="hi-IN")
    
    if success and os.path.exists(output_path):
        print(f"✓ TTS generated successfully!")
        print(f"  File size: {os.path.getsize(output_path)} bytes")
        # Clean up
        try:
            os.remove(output_path)
            print("  Test file cleaned up")
        except:
            pass
        return True
    else:
        print("✗ TTS generation failed")
        return False

def test_fallback():
    """Test that pipeline doesn't break when translation fails."""
    print("=" * 60)
    print("Testing Graceful Fallback")
    print("=" * 60)
    
    # Test with invalid language
    test_text = "This should return unchanged"
    result = translate_text(test_text, "invalid-language")
    
    if result == test_text:
        print("✓ Fallback working: Invalid language returns original text")
        return True
    else:
        print("✗ Fallback failed")
        return False

def main():
    print("\n🌍 Multilingual Support Test Suite\n")
    
    # Show supported languages
    print("Supported languages:")
    for lang_name, lang_code in get_supported_languages().items():
        print(f"  - {lang_name.capitalize()} ({lang_code})")
    print()
    
    # Run tests
    results = []
    
    results.append(("Translation", test_translation()))
    results.append(("TTS Generation", test_tts()))
    results.append(("Graceful Fallback", test_fallback()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Multilingual support is ready.")
    else:
        print("\n⚠ Some tests failed. Check your SARVAM_API_KEY and network connection.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
