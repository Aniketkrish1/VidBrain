"""
utils/topic_clustering.py

Cluster transcript sentences into topic clusters using sentence embeddings + HDBSCAN.
Returns { cluster_id: [ {text,start,end}, ... ] }
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cluster_topics(
    sentences: List[Dict],
    embedding_model: Optional[str] = None,
    min_cluster_size: int = 3,
    keep_percentile: float = 0.0
) -> Dict[int, List[Dict]]:
    """
    Cluster sentences into topics using embeddings and HDBSCAN.

    Args:
        sentences: list of dicts each with keys at least 'text','start','end'
        embedding_model: optional str name of sentence transformer model; 
                         if None defaults to 'all-MiniLM-L6-v2'
        min_cluster_size: minimum cluster size for HDBSCAN
        keep_percentile: optional, drop clusters whose total text length < percentile threshold

    Returns:
        Dict[int, List[Dict]] mapping cluster ID to list of sentences
    """
    if not sentences:
        return {}

    model_name = embedding_model or "all-MiniLM-L6-v2"
    logger.info("Embedding %d sentences with model: %s", len(sentences), model_name)
    encoder = SentenceTransformer(model_name)

    texts = [str(s.get("text", "")).strip() for s in sentences]
    embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)

    clusters: Dict[int, List[Dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[idx])

    if not clusters:
        logger.warning("No clusters found (all noise)")
        return {}

    # optional: filter out tiny clusters by text length percentile
    if keep_percentile and keep_percentile > 0.0:
        # compute total length per cluster
        lengths = {cid: sum(len(s["text"].split()) for s in segs) for cid, segs in clusters.items()}
        if lengths:
            vals = np.array(list(lengths.values()))
            threshold = np.percentile(vals, keep_percentile)
            clusters = {cid: segs for cid, segs in clusters.items() if lengths[cid] >= threshold}
            logger.info("Filtered clusters below %s percentile; kept %d clusters", keep_percentile, len(clusters))

    # sort clusters by earliest start time
    sorted_clusters = dict(sorted(clusters.items(), key=lambda kv: kv[1][0].get("start", 0.0)))
    logger.info("cluster_topics -> %d clusters found", len(sorted_clusters))
    return sorted_clusters
