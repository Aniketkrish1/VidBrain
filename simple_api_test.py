#!/usr/bin/env python3
"""
Simple test to verify OpenRouter API key from .env is working
"""

import os
from dotenv import load_dotenv

# Force reload from .env
load_dotenv(override=True)

print("🔑 OpenRouter API Key Status")
print("=" * 40)

api_key = os.getenv("OPENROUTER_API_KEY", "")

if api_key:
    print(f"✅ API Key loaded from .env")
    print(f"   Key starts with: {api_key[:15]}...")
    print(f"   Key ends with: ...{api_key[-10:]}")
    
    # Test if it's the expected key from .env file
    if api_key.startswith("sk-or-v1-d92967"):
        print("✅ Correct .env key is being used!")
    else:
        print("❌ Wrong key - environment variable may still be interfering")
        
    # Test basic OpenRouter connection
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Simple test call
        print("\n🧪 Testing OpenRouter connection...")
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct",  # Small, fast model
            messages=[{"role": "user", "content": "Say 'API key works!'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API Response: {result}")
        print("✅ OpenRouter API key is working correctly!")
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower():
            print("❌ Rate limit error - Your API key has exceeded usage limits")
            print("   Please check your OpenRouter account or try a different key")
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            print("❌ Invalid API key - Please check your key in .env file")
        else:
            print(f"❌ API Error: {error_msg}")
            
else:
    print("❌ No API key found in .env file")
    
print("\n" + "=" * 40)