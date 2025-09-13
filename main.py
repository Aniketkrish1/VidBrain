#!/usr/bin/env python3
"""
AI Video Summarizer
====================
A complete pipeline for creating AI-generated video summaries from YouTube videos.

This script downloads a YouTube video, transcribes it, identifies key topics,
generates summaries with AI voiceovers, and assembles a final summary video.

Author: AI Video Processing Expert
Date: 2025
"""

import os
import sys
import json
import shutil
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()
# External libraries
import numpy as np
import whisper
import spacy
import torch
import hdbscan
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from TTS.api import TTS
os.environ["TQDM_DISABLE"] = "1"

from utils import topic_clustering as tc
from utils import downloader as dl
from utils import transcriber as tb
from utils import summarizer as sz
from utils.database import VectorDB, parse_srt

# ==================== CONFIGURATION ====================
# User-configurable variables
YOUTUBE_URL = os.getenv("YOUTUBE_URL")  # Replace with actual URL
WHISPER_MODEL =os.getenv("WHISPER_MODEL") # Options: tiny, base, small, medium, large
SPACY_MODEL =os.getenv("SPACY_MODEL") # English language model for sentence segmentation
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Hugging Face summarization model
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui TTS model
MIN_CLUSTER_SIZE = 3  # Minimum sentences per topic cluster
MAX_SUMMARY_LENGTH = 500  # Maximum length for topic summaries
OUTPUT_VIDEO_NAME = os.getenv("OUTPUT_VIDEO_NAME")
TEMP_DIR = os.getenv("TEMP_DIR")

LOG_LEVEL = logging.INFO

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app.log")

# ==================== HELPER FUNCTIONS ====================

def setup_directories():
    """Create necessary directories for processing."""
    Path(TEMP_DIR).mkdir(exist_ok=True)
    logger.info(f"Created temporary directory: {TEMP_DIR}")

def cleanup_files():
    """Remove temporary files and directories."""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.warning(f"Could not fully clean up temporary files: {e}")

# ==================== MAIN PROCESSING FUNCTIONS ====================
# def transcribe_audio(audio_path: str) -> Dict:
#     """
#     Transcribe audio using OpenAI Whisper with word-level timestamps.
    
#     Args:
#         audio_path: Path to the audio file
        
#     Returns:
#         Dictionary containing transcription results with word timestamps
#     """
#     logger.info(f"Transcribing audio with Whisper model: {WHISPER_MODEL}")
    
#     try:
#         # Load Whisper model
#         model = whisper.load_model(WHISPER_MODEL)
        
#         # Transcribe with word timestamps
#         result = model.transcribe(
#             audio_path,
#             word_timestamps=True,
#             verbose=None,
            
#         )
        
#         logger.info(f"Transcription complete. Found {len(result['segments'])} segments")
        
#         # Extract word-level information
#         words_data = []
#         for segment in result['segments']:
#             if 'words' in segment:
#                 for word in segment['words']:
#                     words_data.append({
#                         'word': word['word'].strip(),
#                         'start': word['start'],
#                         'end': word['end']
#                     })
        
#         return {
#             'text': result['text'],
#             'words': words_data,
#             'segments': result['segments']
#         }
        
#     except Exception as e:
#         logger.error(f"Error during transcription: {e}")
#         raise

import datetime

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def segment_and_realign_sentences(transcription_data: Dict) -> List[Dict]:
    """
    Segment transcript into sentences and realign word timestamps.
    
    Args:
        transcription_data: Dictionary with transcription and word timestamps
        
    Returns:
        List of sentences with aligned timestamps
    """
    logger.info("Segmenting text into sentences with spaCy")
    
    try:
        # Load spaCy model
        nlp = spacy.load(SPACY_MODEL)
        
        # Process full text
        doc = nlp(transcription_data['text'])
        
        # Extract sentences
        sentences = []
        words = transcription_data['words']
        word_idx = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Find corresponding words for this sentence
            sent_words = sent_text.lower().split()
            sent_start = None
            sent_end = None
            
            # Match words to get timestamps
            temp_word_idx = word_idx
            matched_words = []
            
            for target_word in sent_words:
                for i in range(temp_word_idx, len(words)):
                    if target_word in words[i]['word'].lower():
                        if sent_start is None:
                            sent_start = words[i]['start']
                        sent_end = words[i]['end']
                        matched_words.append(words[i])
                        temp_word_idx = i + 1
                        break
            
            if sent_start is not None and sent_end is not None:
                sentences.append({
                    'text': sent_text,
                    'start': sent_start,
                    'end': sent_end,
                    'words': matched_words
                })
                word_idx = temp_word_idx
        
        logger.info(f"Segmented into {len(sentences)} sentences")
        return sentences
        
    except Exception as e:
        logger.error(f"Error during sentence segmentation: {e}")
        raise


def write_srt(sentences: List[Dict], filename: str = "transcript.srt"):
    """Write segmented sentences to SRT file"""
    with open(filename, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(sentences, start=1):
            start_time = seconds_to_srt_time(sent["start"])
            end_time = seconds_to_srt_time(sent["end"])
            
            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{sent['text']}\n\n")
    logger.info(f"SRT file saved to {filename}")

# def cluster_topics(sentences: List[Dict]) -> Dict[int, List[Dict]]:
#     """
#     Cluster sentences into topics using embeddings and HDBSCAN.
    
#     Args:
#         sentences: List of sentence dictionaries
        
#     Returns:
#         Dictionary mapping cluster IDs to sentences
#     """
#     logger.info("Generating sentence embeddings and clustering topics")
    
#     try:
#         # Load sentence transformer model
#         encoder = SentenceTransformer(EMBEDDING_MODEL)
        
#         # Generate embeddings
#         sentence_texts = [s['text'] for s in sentences]
#         embeddings = encoder.encode(sentence_texts, show_progress_bar=False)
        
#         # Perform clustering with HDBSCAN
#         clusterer = hdbscan.HDBSCAN(
#             min_cluster_size=MIN_CLUSTER_SIZE,
#             min_samples=1,
#             metric='cosine'
#         )
        
#         cluster_labels = clusterer.fit_predict(embeddings)
        
#         # Group sentences by cluster
#         clusters = {}
#         for idx, label in enumerate(cluster_labels):
#             if label == -1:  # Noise points
#                 continue
#             if label not in clusters:
#                 clusters[label] = []
#             clusters[label].append(sentences[idx])
        
#         # Sort clusters by temporal order (first sentence appearance)
#         sorted_clusters = dict(sorted(
#             clusters.items(),
#             key=lambda x: x[1][0]['start']
#         ))
        
#         logger.info(f"Found {len(sorted_clusters)} topic clusters")
#         with open("sentence.txt",'w') as f:
#             for cluster in sorted_clusters.values():

#                 print(cluster[0],file=f)
#         return sorted_clusters
        
#     except Exception as e:
#         logger.error(f"Error during topic clustering: {e}")
#         raise

# def summarize_topics(topic_clusters: Dict[int, List[Dict]]) -> Dict[int, str]:
#     """
#     Generate summaries for each topic cluster.
    
#     Args:
#         topic_clusters: Dictionary mapping cluster IDs to sentences
        
#     Returns:
#         Dictionary mapping cluster IDs to summary text
#     """
#     logger.info("Generating summaries for topic clusters")
    
#     try:
#         # Initialize summarization pipeline
#         summarizer = pipeline(
#             "summarization",
#             model=SUMMARIZATION_MODEL,
#             device=0 if torch.cuda.is_available() else -1
#         )
        
#         summaries = {}
        
#         for cluster_id, sentences in topic_clusters.items():
#             # Combine sentences in cluster
#             cluster_text = " ".join([s['text'] for s in sentences])
            
#             # Generate summary
#             if len(cluster_text.split()) > 40:  # Only summarize if substantial
#                 summary = summarizer(
#                     cluster_text,
#                     max_length=MAX_SUMMARY_LENGTH,
#                     min_length=30,
#                     do_sample=False
#                 )[0]['summary_text']
#             else:
#                 summary = cluster_text  # Use original if too short
            
#             summaries[cluster_id] = summary
#             logger.info(f"Generated summary for cluster {cluster_id}")
        
#         return summaries
        
#     except Exception as e:
#         logger.error(f"Error during summarization: {e}")
#         raise

def generate_voiceovers(summaries: Dict[int, str]) -> Dict[int, str]:
    """
    Generate TTS voiceovers for each summary.
    
    Args:
        summaries: Dictionary mapping cluster IDs to summary text
        
    Returns:
        Dictionary mapping cluster IDs to audio file paths
    """
    logger.info("Generating TTS voiceovers")
    
    try:
        # Initialize TTS model
        tts = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=torch.cuda.is_available())
        
        voiceover_paths = {}
        
        for cluster_id, summary_text in summaries.items():
            # Generate audio file path
            audio_path = os.path.join(TEMP_DIR, f"voiceover_{cluster_id}.wav")
            
            # Generate speech
            tts.tts_to_file(
                text=summary_text,
                file_path=audio_path
            )
            
            voiceover_paths[cluster_id] = audio_path
            logger.info(f"Generated voiceover for cluster {cluster_id}")
        
        return voiceover_paths
        
    except Exception as e:
        logger.error(f"Error during TTS generation: {e}")
        raise

def assemble_video(
    video_path: str,
    topic_clusters: Dict[int, List[Dict]],
    voiceover_paths: Dict[int, str],
    output_path: str
) -> str:
    """
    Assemble final summary video with topic clips and voiceovers.
    
    Args:
        video_path: Path to original video
        topic_clusters: Dictionary mapping cluster IDs to sentences
        voiceover_paths: Dictionary mapping cluster IDs to audio paths
        output_path: Path for output video
        
    Returns:
        Path to final video
    """
    logger.info("Assembling final summary video")
    
    try:
        # Load original video
        original_video = VideoFileClip(video_path)
        
        video_segments = []
        
        for cluster_id in sorted(topic_clusters.keys()):
            sentences = topic_clusters[cluster_id]
            voiceover_path = voiceover_paths[cluster_id]
            
            # Get time range for this cluster
            start_time = sentences[0]['start']
            end_time = sentences[-1]['end']
            
            # Extract video segment
            video_segment = original_video.subclip(start_time, end_time)
            
            # Load voiceover audio
            voiceover_audio = AudioFileClip(voiceover_path)

            
            # Adjust video segment duration to match voiceover
            if voiceover_audio.duration < video_segment.duration:
                video_segment = video_segment.subclip(0, voiceover_audio.duration)
            elif voiceover_audio.duration > video_segment.duration:
                # Loop or extend video if voiceover is longer
                video_segment = video_segment.loop(duration=voiceover_audio.duration)
            
            # Replace audio with voiceover
            video_segment = video_segment.set_audio(voiceover_audio.audio_fadein(0.02).audio_fadeout(0.02))
            
            video_segments.append(video_segment)
            logger.info(f"Processed segment for cluster {cluster_id}")
        
        # Concatenate all segments
        final_video = concatenate_videoclips(video_segments, method="compose")
        
        # Write final video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=False,
            verbose=False,
            logger=None
        )
        
        # Clean up
        original_video.close()
        for segment in video_segments:
            segment.close()
        final_video.close()
        
        logger.info(f"Final video saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during video assembly: {e}")
        raise


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function that orchestrates the entire pipeline."""
    
    logger.info("="*50)
    logger.info("Starting AI Video Summarizer Pipeline")
    logger.info("="*50)
    
    try:
        # Setup
        setup_directories()
        
        # Step 1: Download and extract audio
        logger.info("Step 1/7: Downloading video and extracting audio...")
        video_path, audio_path = dl.download_and_extract_audio(YOUTUBE_URL)
        
        # Step 2: Transcribe audio
        logger.info("Step 2/7: Transcribing audio with Whisper...")
        print(audio_path)
        transcription_data = tb.transcribe_audio(audio_path)
        
        # Step 3: Segment sentences
        logger.info("Step 3/7: Segmenting and aligning sentences...")
        sentences = segment_and_realign_sentences(transcription_data)
        write_srt(sentences, "transcript.srt")
        # Step 3b: Build vector database from transcript
        logger.info("Step 3b: Building vector database from transcript...")
        segments = parse_srt("transcript.srt")
        db = VectorDB(db_path="vector_db.pkl")
        db.build(segments)
        logger.info("Vector database created with transcript embeddings")

        # Step 4: Cluster topics
        logger.info("Step 4/7: Clustering sentences into topics...")
        topic_clusters = tc.cluster_topics(sentences)
        
        if not topic_clusters:
            logger.error("No topic clusters found. Exiting.")
            return
        
        # Step 5: Generate summaries
        logger.info("Step 5/7: Generating topic summaries...")
        summaries = sz.summarize_topics(topic_clusters)
        
        # Step 6: Generate voiceovers
        logger.info("Step 6/7: Generating TTS voiceovers...")
        voiceover_paths = generate_voiceovers(summaries)
        
        # Step 7: Assemble final video
        logger.info("Step 7/7: Assembling final summary video...")
        final_video_path = assemble_video(
            video_path,
            topic_clusters,
            voiceover_paths,
            OUTPUT_VIDEO_NAME
        )
        
        # Cleanup
        cleanup_files()
        
        logger.info("="*50)
        logger.info(f"✅ Success! Summary video created: {final_video_path}")
        logger.info(f"Topics identified: {len(topic_clusters)}")
        logger.info("="*50)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        cleanup_files()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        cleanup_files()
        sys.exit(1)

if __name__ == "__main__":
    main()
