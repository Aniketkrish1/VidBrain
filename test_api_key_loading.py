#!/usr/bin/env python3
"""
Test script to verify OpenRouter API key loading from .env file
"""

import os
from dotenv import load_dotenv

print("🔑 Testing OpenRouter API Key Loading")
print("=" * 50)

# Test 1: Load environment without override
print("\n1. Loading environment (without override):")
load_dotenv()
api_key_1 = os.getenv("OPENROUTER_API_KEY", "")
print(f"   API Key found: {'✅ Yes' if api_key_1 else '❌ No'}")
if api_key_1:
    print(f"   Key starts with: {api_key_1[:15]}...")

# Test 2: Load environment with override
print("\n2. Loading environment (with override=True):")
load_dotenv(override=True)
api_key_2 = os.getenv("OPENROUTER_API_KEY", "")
print(f"   API Key found: {'✅ Yes' if api_key_2 else '❌ No'}")
if api_key_2:
    print(f"   Key starts with: {api_key_2[:15]}...")

# Test 3: Test module imports
print("\n3. Testing module imports:")
try:
    from utils.topic_query_processor import refresh_openrouter_client
    success = refresh_openrouter_client()
    print(f"   Topic Query Processor: {'✅ Connected' if success else '❌ Failed'}")
except Exception as e:
    print(f"   Topic Query Processor: ❌ Error - {e}")

try:
    from utils.summarizer import refresh_openrouter_client as refresh_summarizer
    success = refresh_summarizer()
    print(f"   Summarizer: {'✅ Connected' if success else '❌ Failed'}")
except Exception as e:
    print(f"   Summarizer: ❌ Error - {e}")

# Test 4: Check .env file exists
print("\n4. Checking .env file:")
env_path = ".env"
if os.path.exists(env_path):
    print(f"   .env file: ✅ Found")
    with open(env_path, 'r') as f:
        content = f.read()
        if "OPENROUTER_API_KEY" in content:
            print(f"   OPENROUTER_API_KEY in file: ✅ Yes")
        else:
            print(f"   OPENROUTER_API_KEY in file: ❌ No")
else:
    print(f"   .env file: ❌ Not found")

print("\n" + "=" * 50)
print("✅ Test completed! Check results above.")