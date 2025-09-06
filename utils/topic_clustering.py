from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import hdbscan

e_model =os.getenv("EMBEDDING_MODEL")
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE"))

def cluster_topics(sentences: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Cluster sentences into topics using SBERT embeddings (cosine similarity) + dynamic temporal weighting + HDBSCAN.
    """
    # logger.info("Generating sentence embeddings and clustering topics")

    try:
        # Load embedding model
        encoder = SentenceTransformer(e_model)
        sentence_texts = [s['text'] for s in sentences]
        embeddings = encoder.encode(sentence_texts, show_progress_bar=False, normalize_embeddings=True)

        # Calculate video length
        video_length = max(s['end'] for s in sentences)
        temporal_weight = max(0.05, min(0.5, 1 - (video_length / 40)))  # dynamic scaling

        # logger.info(f"Dynamic temporal weight: {temporal_weight:.2f} for video length {video_length:.2f}s")

        # Normalize timestamps
        times = np.array([s['start'] for s in sentences]).reshape(-1, 1)
        times = (times - times.min()) / (times.max() - times.min())

        # Combine using cosine for embeddings + weighted time penalty
        # We create a custom distance matrix
        from sklearn.metrics.pairwise import cosine_distances
        cosine_dist = cosine_distances(embeddings)
        temporal_dist = np.abs(times - times.T)  # pairwise temporal difference

        # Weighted distance matrix
        combined_dist = (1 - temporal_weight) * cosine_dist + temporal_weight * temporal_dist

        # Cluster using precomputed distance
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=1,
            metric='precomputed'
        )
        cluster_labels = clusterer.fit_predict(combined_dist)

        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # noise
                continue
            clusters.setdefault(label, []).append(sentences[idx])

        # Sort clusters by first sentence time
        sorted_clusters = dict(sorted(
            clusters.items(),
            key=lambda x: x[1][0]['start']
        ))

        # logger.info(f"Found {len(sorted_clusters)} topic clusters")
        with open("sentence.txt",'w') as f:
            for cluster in sorted_clusters.values():
                print(cluster[0]['text'],file=f)
        return sorted_clusters

    except Exception as e:
        # logger.error(f"Error during topic clustering: {e}")
        raise
