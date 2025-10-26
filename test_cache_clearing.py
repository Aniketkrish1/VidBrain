#!/usr/bin/env python3
"""
Test vector database cache clearing behavior on application restart.
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def test_cache_clearing():
    """Test that vector database cache clears properly"""
    
    print("🧪 Testing Vector Database Cache Clearing")
    print("=" * 50)
    
    # Check current environment setting
    clear_cache = os.getenv("CLEAR_VECTOR_CACHE", "true").lower() == "true"
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db.pkl")
    
    print(f"📋 Configuration:")
    print(f"   CLEAR_VECTOR_CACHE: {clear_cache}")
    print(f"   VECTOR_DB_PATH: {vector_db_path}")
    
    # Check if database file exists
    db_exists_before = os.path.exists(vector_db_path)
    print(f"\n📁 Database file status:")
    print(f"   Exists before test: {db_exists_before}")
    
    if db_exists_before:
        file_size = os.path.getsize(vector_db_path)
        mod_time = os.path.getmtime(vector_db_path)
        print(f"   File size: {file_size} bytes")
        print(f"   Last modified: {time.ctime(mod_time)}")
    
    # Create a dummy database file if it doesn't exist
    if not db_exists_before:
        print(f"\n🔧 Creating dummy database file for testing...")
        with open(vector_db_path, "wb") as f:
            f.write(b"dummy_database_content")
        print(f"   ✅ Created {vector_db_path}")
    
    print(f"\n🔄 Testing cache clearing behavior...")
    
    # Test the VectorDB initialization with clear_on_startup
    try:
        from utils.database import VectorDB
        
        print(f"   Initializing VectorDB with clear_on_startup={clear_cache}")
        db = VectorDB(db_path=vector_db_path, clear_on_startup=clear_cache)
        
        # Check if file still exists after initialization
        db_exists_after = os.path.exists(vector_db_path)
        print(f"   Database exists after init: {db_exists_after}")
        
        if clear_cache:
            if not db_exists_after:
                print(f"   ✅ SUCCESS: Cache was cleared as expected")
                result = "CLEARED"
            else:
                print(f"   ❌ FAILURE: Cache was NOT cleared (file still exists)")
                result = "NOT_CLEARED"
        else:
            if db_exists_after:
                print(f"   ✅ SUCCESS: Cache was preserved as expected")
                result = "PRESERVED"
            else:
                print(f"   ❌ FAILURE: Cache was cleared unexpectedly")
                result = "UNEXPECTEDLY_CLEARED"
                
    except Exception as e:
        print(f"   ❌ ERROR: Failed to test VectorDB: {e}")
        result = "ERROR"
    
    print(f"\n📊 Test Results:")
    print(f"   Cache clearing enabled: {clear_cache}")
    print(f"   Result: {result}")
    
    # Provide recommendations
    print(f"\n💡 Recommendations:")
    if clear_cache:
        print(f"   ✅ Vector database will reset on every app restart")
        print(f"   ⚡ Fresh embeddings built for each new video")
        print(f"   🔄 No stale cache issues")
        print(f"   ⚠️  Slower startup (needs to rebuild embeddings)")
    else:
        print(f"   ⚡ Vector database persists between restarts")
        print(f"   🚀 Faster startup (reuses existing embeddings)")
        print(f"   ⚠️  May have stale cache for different videos")
        print(f"   💡 Good for development/testing with same video")
    
    # Cleanup test file if we created it
    if not db_exists_before and os.path.exists(vector_db_path):
        try:
            os.remove(vector_db_path)
            print(f"\n🗑️  Cleaned up test database file")
        except:
            pass
    
    return result == "CLEARED" if clear_cache else result == "PRESERVED"

if __name__ == "__main__":
    success = test_cache_clearing()
    if success:
        print(f"\n🎉 Cache clearing test PASSED!")
    else:
        print(f"\n💥 Cache clearing test FAILED!")