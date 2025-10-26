"""
Test script to verify multilingual functionality with Sarvam AI integration.
"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multilingual_integration():
    """Test the complete multilingual integration"""
    
    logger.info("🧪 Testing multilingual integration...")
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Check if Sarvam API key is set
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        logger.error("❌ SARVAM_API_KEY not found in environment variables")
        logger.info("📝 Please add your Sarvam AI API key to .env file:")
        logger.info("   SARVAM_API_KEY=your_api_key_here")
        return False
    
    logger.info(f"✅ Sarvam API key found: {sarvam_key[:10]}...")
    
    # Test Sarvam AI import
    try:
        from utils.sarvam_ai import SarvamAI, get_supported_languages
        logger.info("✅ Sarvam AI module imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Sarvam AI module: {e}")
        return False
    
    # Test supported languages
    try:
        languages = get_supported_languages()
        logger.info(f"✅ Supported languages loaded: {len(languages)} languages")
        for code, name in languages.items():
            logger.info(f"   {code}: {name}")
    except Exception as e:
        logger.error(f"❌ Failed to get supported languages: {e}")
        return False
    
    # Test API connection (simple test)
    try:
        sarvam = SarvamAI()
        logger.info("✅ Sarvam AI client initialized successfully")
        
        # Test with a simple Hindi translation
        test_text = "Hello, this is a test message."
        logger.info(f"🔄 Testing translation: '{test_text}' → Hindi")
        
        # This will test the API connection
        translated = sarvam.translate_text(test_text, "hi", "en")
        logger.info(f"✅ Translation successful: {translated}")
        
        logger.info("🎉 All multilingual integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sarvam AI API test failed: {e}")
        logger.info("💡 This might be due to:")
        logger.info("   1. Invalid API key")
        logger.info("   2. Network connectivity issues")
        logger.info("   3. Sarvam AI service unavailable")
        return False

def test_frontend_integration():
    """Test frontend integration"""
    logger.info("🧪 Testing frontend integration...")
    
    # Check if language dropdown was added to frontend
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for language selection elements
        if 'name="language"' in content and 'Hindi (हिंदी)' in content:
            logger.info("✅ Language dropdown found in frontend")
            
            # Count language options
            language_count = content.count('<option value="') - 1  # Subtract 1 for other options
            logger.info(f"✅ Found {language_count} language options in dropdown")
            
            return True
        else:
            logger.error("❌ Language dropdown not found in frontend")
            return False
            
    except Exception as e:
        logger.error(f"❌ Frontend test failed: {e}")
        return False

def test_backend_integration():
    """Test backend integration"""
    logger.info("🧪 Testing backend integration...")
    
    try:
        # Test app.py modifications
        with open("app.py", "r", encoding="utf-8") as f:
            app_content = f.read()
        
        if 'language: str = Form("en")' in app_content:
            logger.info("✅ Backend API endpoints updated with language parameter")
        else:
            logger.error("❌ Language parameter not found in API endpoints")
            return False
        
        # Test main.py modifications
        with open("main.py", "r", encoding="utf-8") as f:
            main_content = f.read()
        
        if 'from utils.sarvam_ai import' in main_content and 'language: str = "en"' in main_content:
            logger.info("✅ Main pipeline updated with Sarvam AI integration")
            return True
        else:
            logger.error("❌ Main pipeline not properly updated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Backend test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting comprehensive multilingual integration tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Frontend Integration", test_frontend_integration),
        ("Backend Integration", test_backend_integration),
        ("Sarvam AI Integration", test_multilingual_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Multilingual integration is ready!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Add your Sarvam AI API key to .env file")
        logger.info("   2. Test with a real video and Hindi/Tamil translation")
        logger.info("   3. Verify voice quality and translation accuracy")
    else:
        logger.error("❌ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)