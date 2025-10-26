#!/usr/bin/env python3
import pickle
import os
import sys
sys.path.append('.')

print("🔍 Checking VidBrain state...")

# Check vector DB
if os.path.exists('vector_db.pkl'):
    with open('vector_db.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f'📊 Vector DB contains {len(data["metadata"])} segments')

        print('📝 Sample segments:')
        for i in range(min(5, len(data['metadata']))):
            meta = data['metadata'][i]
            print(f'  {i}: {meta["text"][:50]}...')
else:
    print('❌ Vector DB missing')

# Check temp files
temp_dir = 'temp_processing'
if os.path.exists(temp_dir):
    print(f'📁 Temp directory contains {len(os.listdir(temp_dir))} files:')
    for f in os.listdir(temp_dir):
        if os.path.isfile(os.path.join(temp_dir, f)):
            size = os.path.getsize(os.path.join(temp_dir, f))
            print(f'  {f}: {size / 1024:.1f} KB')

# Check outputs
outputs_dir = 'outputs'
if os.path.exists(outputs_dir):
    print(f'📁 Outputs directory contains {len(os.listdir(outputs_dir))} files')
else:
    print('❌ Outputs directory missing')

