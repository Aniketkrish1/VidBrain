from transformers import pipeline
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
load_dotenv()
import torch
summary_length=int(os.getenv("MAX_SUMMARY_LENGTH"))
model = os.getenv("SUMMARIZATION_MODEL")
def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> list:
    """Splits long text into chunks with overlap to avoid cutting context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def summarize_topics(topic_clusters: Dict[int, List[Dict]]) -> Dict[int, str]:
    """
    Generate summaries for each topic cluster with chunking for long texts.
    """
    # logger.info("Generating summaries for topic clusters")

    try:
        summarizer = pipeline(
            "summarization",
            model=model,
            device=0 if torch.cuda.is_available() else -1
        )

        summaries = {}

        for cluster_id, sentences in topic_clusters.items():
            cluster_text = " ".join([s['text'] for s in sentences])

            if len(cluster_text.split()) > 40:
                # Chunk if too long
                if len(cluster_text.split()) > 900:  # slightly below model max
                    chunks = chunk_text(cluster_text)
                    chunk_summaries = []
                    for chunk in chunks:
                        chunk_summaries.append(
                            summarizer(
                                chunk,
                                max_length=summary_length,
                                min_length=30,
                                do_sample=False
                            )[0]['summary_text']
                        )
                    # Summarize the combined chunk summaries again (optional)
                    combined_summary = " ".join(chunk_summaries)
                    summary = summarizer(
                        combined_summary,
                        max_length=summary_length,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                else:
                    summary = summarizer(
                        cluster_text,
                        max_length=summary_length,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
            else:
                summary = cluster_text

            summaries[cluster_id] = summary
            # logger.info(f"Generated summary for cluster {cluster_id}")

        return summaries

    except Exception as e:
        # logger.error(f"Error during summarization: {e}")
        raise
