#!/usr/bin/env python3
"""
Optimized AI Video Condenser
============================
High-performance pipeline for creating AI-generated video summaries with minimal data loss.

Key Optimizations:
- Parallel processing where possible
- Shared model instances and embeddings
- Streaming and chunked processing
- GPU memory optimization
- Intelligent caching
- Scene-aware summarization

Author: AI Video Processing Expert
Date: 2025
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# External libraries
import numpy as np
import torch
import cv2
import spacy
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from TTS.api import TTS
import google.generativeai as genai
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import pickle
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Configuration
@dataclass
class Config:
    youtube_url: str = os.getenv("YOUTUBE_URL")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    embedding_model: str = "all-MiniLM-L6-v2"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    temp_dir: str = os.getenv("TEMP_DIR", "./temp")
    output_name: str = os.getenv("OUTPUT_VIDEO_NAME", "summary.mp4")
    scene_threshold: float = 30.0
    max_summary_length: int = 150
    cache_dir: str = "./cache"
    max_workers: int = min(4, os.cpu_count())
    gpu_memory_fraction: float = 0.7

config = Config()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model management with GPU memory optimization"""
    
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_gpu_memory()
        
    def _setup_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
            
    def get_whisper_model(self):
        if 'whisper' not in self.models:
            self.models['whisper'] = WhisperModel(
                config.whisper_model,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
        return self.models['whisper']
    
    def get_embedding_model(self):
        if 'embeddings' not in self.models:
            self.models['embeddings'] = SentenceTransformer(
                config.embedding_model,
                device=self.device
            )
        return self.models['embeddings']
    
    def get_tts_model(self):
        if 'tts' not in self.models:
            self.models['tts'] = TTS(
                model_name=config.tts_model,
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
        return self.models['tts']
    
    def cleanup(self):
        """Release GPU memory"""
        for model_name in self.models:
            del self.models[model_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class CacheManager:
    """Intelligent caching system"""
    
    def __init__(self, cache_dir: str = config.cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[any]:
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def set(self, key: str, data: any):
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

class OptimizedVideoProcessor:
    """Main processing class with optimized pipeline"""
    
    def __init__(self):
        self.models = ModelManager()
        self.cache = CacheManager()
        self.setup_directories()
        
        # Setup Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        
    def setup_directories(self):
        Path(config.temp_dir).mkdir(exist_ok=True)
        Path(config.cache_dir).mkdir(exist_ok=True)
        
    def download_video(self, url: str) -> Tuple[str, str]:
        """Download video with caching"""
        cache_key = self.cache.get_cache_key(url)
        cached = self.cache.get(f"download_{cache_key}")
        
        if cached:
            video_path, audio_path = cached
            if os.path.exists(video_path) and os.path.exists(audio_path):
                logger.info("Using cached video/audio files")
                return video_path, audio_path
        
        # Download logic (your existing download code)
        from pytubefix import YouTube
        
        yt = YouTube(url)
        title = yt.title.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        # Download video and audio in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            video_future = executor.submit(self._download_video_stream, yt, title)
            audio_future = executor.submit(self._download_audio_stream, yt, title)
            
            video_path = video_future.result()
            audio_path = audio_future.result()
        
        # Cache the paths
        self.cache.set(f"download_{cache_key}", (video_path, audio_path))
        return video_path, audio_path
    
    def _download_video_stream(self, yt, title):
        video_stream = yt.streams.filter(res='1080p', file_extension='mp4').first()
        if not video_stream:
            video_stream = yt.streams.filter(only_video=True, file_extension='mp4').order_by('resolution').desc().first()
        
        video_dir = os.path.join(config.temp_dir, title)
        os.makedirs(video_dir, exist_ok=True)
        video_stream.download(output_path=video_dir, filename="video.mp4")
        return os.path.join(video_dir, "video.mp4")
    
    def _download_audio_stream(self, yt, title):
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        audio_dir = os.path.join(config.temp_dir, title)
        os.makedirs(audio_dir, exist_ok=True)
        audio_stream.download(output_path=audio_dir, filename="audio.mp3")
        return os.path.join(audio_dir, "audio.mp3")
    
    def transcribe_and_segment(self, audio_path: str) -> List[Dict]:
        """Optimized transcription with sentence segmentation"""
        cache_key = self.cache.get_cache_key(audio_path + str(os.path.getmtime(audio_path)))
        cached = self.cache.get(f"transcribe_{cache_key}")
        
        if cached:
            logger.info("Using cached transcription")
            return cached
        
        logger.info("Transcribing audio...")
        model = self.models.get_whisper_model()
        
        # Transcribe with word timestamps
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5
        )
        
        # Process segments and create sentence-level data
        sentences = []
        nlp = spacy.load("en_core_web_sm")
        
        full_text = ""
        all_words = []
        
        for segment in segments:
            full_text += segment.text + " "
            if segment.words:
                all_words.extend([{
                    'word': word.word.strip(),
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                } for word in segment.words])
        
        # Sentence segmentation with word alignment
        doc = nlp(full_text.strip())
        word_idx = 0
        
        for sent in doc.sents:
            if not sent.text.strip():
                continue
                
            sent_words = []
            sent_start = None
            sent_end = None
            
            # Align words to sentences
            for token in sent:
                for i in range(word_idx, len(all_words)):
                    if token.text.lower() in all_words[i]['word'].lower():
                        if sent_start is None:
                            sent_start = all_words[i]['start']
                        sent_end = all_words[i]['end']
                        sent_words.append(all_words[i])
                        word_idx = i + 1
                        break
            
            if sent_start is not None and sent_end is not None:
                sentences.append({
                    'text': sent.text.strip(),
                    'start': sent_start,
                    'end': sent_end,
                    'words': sent_words
                })
        
        # Cache results
        self.cache.set(f"transcribe_{cache_key}", sentences)
        logger.info(f"Transcription complete: {len(sentences)} sentences")
        return sentences
    
    def detect_scenes_parallel(self, video_path: str) -> List[Tuple[float, float]]:
        """Parallel scene detection with caching"""
        cache_key = self.cache.get_cache_key(video_path + str(os.path.getmtime(video_path)))
        cached = self.cache.get(f"scenes_{cache_key}")
        
        if cached:
            logger.info("Using cached scene detection")
            return cached
        
        logger.info("Detecting scenes...")
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=config.scene_threshold))
        
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scenes = scene_manager.get_scene_list()
        
        result = [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
        
        # Cache results
        self.cache.set(f"scenes_{cache_key}", result)
        logger.info(f"Scene detection complete: {len(result)} scenes")
        return result
    
    def extract_frames_batch(self, video_path: str, timestamps: List[float]) -> Dict[float, bytes]:
        """Extract multiple frames efficiently"""
        frames = {}
        cap = cv2.VideoCapture(video_path)
        
        for timestamp in sorted(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if ret:
                _, img_bytes = cv2.imencode(".jpg", frame)
                frames[timestamp] = img_bytes.tobytes()
        
        cap.release()
        return frames
    
    async def summarize_scene_async(self, scene_text: str, frame_bytes: Optional[bytes]) -> str:
        """Async scene summarization with Gemini"""
        parts = [
            {"text": f"Summarize this transcript concisely (max {config.max_summary_length} words), focusing on key information:"},
            {"text": scene_text}
        ]
        
        if frame_bytes:
            import base64
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(frame_bytes).decode("utf-8")
                }
            })
        
        try:
            response = self.gemini_model.generate_content(parts)
            return response.text.strip() if response and response.text else scene_text[:config.max_summary_length]
        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            # Fallback to text truncation
            words = scene_text.split()
            return " ".join(words[:config.max_summary_length])
    
    async def process_scenes_parallel(self, video_path: str, sentences: List[Dict], scenes: List[Tuple[float, float]]) -> List[Dict]:
        """Process all scenes in parallel"""
        logger.info(f"Processing {len(scenes)} scenes in parallel...")
        
        # Extract all needed frames at once
        mid_timestamps = [(start + end) / 2 for start, end in scenes]
        frames = self.extract_frames_batch(video_path, mid_timestamps)
        
        # Create tasks for parallel processing
        tasks = []
        scene_data = []
        
        for idx, (start, end) in enumerate(scenes):
            # Get transcript for this scene
            scene_sentences = [s["text"] for s in sentences if start <= s["start"] <= end or start <= s["end"] <= end]
            if not scene_sentences:
                continue
                
            scene_text = " ".join(scene_sentences)
            mid_time = (start + end) / 2
            frame_bytes = frames.get(mid_time)
            
            # Create async task
            task = self.summarize_scene_async(scene_text, frame_bytes)
            tasks.append(task)
            scene_data.append({
                "scene_id": idx,
                "start": start,
                "end": end,
                "original_text": scene_text
            })
        
        # Execute all summarization tasks in parallel
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results = []
        for i, summary in enumerate(summaries):
            if not isinstance(summary, Exception):
                scene_info = scene_data[i].copy()
                scene_info["summary"] = summary
                results.append(scene_info)
            else:
                logger.warning(f"Scene {i} summarization failed: {summary}")
        
        logger.info(f"Scene processing complete: {len(results)} scenes processed")
        return results
    
    def generate_voiceovers_parallel(self, summaries: List[Dict]) -> Dict[int, str]:
        """Generate TTS voiceovers in parallel"""
        logger.info("Generating voiceovers in parallel...")
        tts = self.models.get_tts_model()
        voiceover_paths = {}
        
        def generate_single_voiceover(scene_data):
            scene_id = scene_data["scene_id"]
            summary = scene_data["summary"]
            audio_path = os.path.join(config.temp_dir, f"voiceover_{scene_id}.wav")
            
            try:
                tts.tts_to_file(text=summary, file_path=audio_path)
                return scene_id, audio_path
            except Exception as e:
                logger.error(f"TTS failed for scene {scene_id}: {e}")
                return scene_id, None
        
        # Process in parallel batches to avoid memory issues
        batch_size = 3
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=min(len(batch), 2)) as executor:
                futures = [executor.submit(generate_single_voiceover, scene) for scene in batch]
                for future in as_completed(futures):
                    scene_id, audio_path = future.result()
                    if audio_path:
                        voiceover_paths[scene_id] = audio_path
        
        logger.info(f"Generated {len(voiceover_paths)} voiceovers")
        return voiceover_paths
    
    def assemble_video_optimized(self, video_path: str, scene_summaries: List[Dict], voiceover_paths: Dict[int, str]) -> str:
        """Optimized video assembly with better memory management"""
        logger.info("Assembling final video...")
        
        video_segments = []
        original_video = VideoFileClip(video_path)
        
        try:
            for scene_data in scene_summaries:
                scene_id = scene_data["scene_id"]
                start = scene_data["start"]
                end = scene_data["end"]
                
                if scene_id not in voiceover_paths:
                    continue
                
                # Extract video segment
                video_segment = original_video.subclip(start, min(end, original_video.duration))
                voiceover_audio = AudioFileClip(voiceover_paths[scene_id])
                
                # Adjust timing
                target_duration = voiceover_audio.duration
                if video_segment.duration < target_duration:
                    video_segment = video_segment.loop(duration=target_duration)
                else:
                    video_segment = video_segment.subclip(0, target_duration)
                
                # Replace audio
                video_segment = video_segment.set_audio(
                    voiceover_audio.audio_fadein(0.02).audio_fadeout(0.02)
                )
                
                video_segments.append(video_segment)
                logger.info(f"Processed scene {scene_id}")
            
            # Concatenate and save
            if video_segments:
                final_video = concatenate_videoclips(video_segments, method="compose")
                output_path = os.path.join(config.temp_dir, config.output_name)
                
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None,
                    threads=config.max_workers
                )
                
                # Cleanup
                for segment in video_segments:
                    segment.close()
                final_video.close()
                
                return output_path
            else:
                raise ValueError("No video segments created")
                
        finally:
            original_video.close()
    
    async def process_video(self, url: str) -> str:
        """Main processing pipeline"""
        logger.info("=== Starting Optimized Video Condenser ===")
        
        try:
            # Step 1: Download (cached)
            video_path, audio_path = self.download_video(url)
            
            # Step 2: Parallel transcription and scene detection
            transcription_task = asyncio.create_task(
                asyncio.to_thread(self.transcribe_and_segment, audio_path)
            )
            scene_task = asyncio.create_task(
                asyncio.to_thread(self.detect_scenes_parallel, video_path)
            )
            
            sentences, scenes = await asyncio.gather(transcription_task, scene_task)
            
            # Step 3: Parallel scene processing
            scene_summaries = await self.process_scenes_parallel(video_path, sentences, scenes)
            
            # Step 4: Generate voiceovers
            voiceover_paths = self.generate_voiceovers_parallel(scene_summaries)
            
            # Step 5: Assemble final video
            final_video_path = self.assemble_video_optimized(video_path, scene_summaries, voiceover_paths)
            
            logger.info(f"✅ Success! Video condensed: {final_video_path}")
            logger.info(f"📊 Stats: {len(scenes)} scenes, {len(scene_summaries)} summaries")
            
            return final_video_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.models.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.models.cleanup()

async def main():
    """Main entry point"""
    if not config.youtube_url:
        logger.error("Please set YOUTUBE_URL environment variable")
        return
    
    processor = OptimizedVideoProcessor()
    
    try:
        result = await processor.process_video(config.youtube_url)
        print(f"\n🎉 Video processing complete!")
        print(f"📁 Output: {result}")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
