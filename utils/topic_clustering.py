import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from main import EMBEDDING_MODEL, MIN_CLUSTER_SIZE, logger


def cluster_topics(sentences: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Cluster sentences into important topic groups using embeddings + HDBSCAN.

    Args:
        sentences: List of dicts with 'text', 'start', 'end', 'words'

    Returns:
        Dict[int, List[Dict]]: mapping cluster_id -> list of sentence dicts
    """
    logger.info("Clustering sentences into topics with embeddings + HDBSCAN")

    if not sentences:
        logger.warning("No sentences provided for clustering")
        return {}

    # 1. Encode sentences
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [s["text"] for s in sentences]
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # 2. Importance filtering (drop filler sentences)
    centroid = np.mean(embeddings, axis=0)
    scores = util.cos_sim(embeddings, centroid.reshape(1, -1)).cpu().numpy().flatten()
    threshold = np.percentile(scores, 20)  # keep top 60%
    keep_idxs = [i for i, s in enumerate(scores) if s >= threshold]

    if not keep_idxs:
        logger.warning("No sentences passed importance filter")
        return {}

    kept_sentences = [sentences[i] for i in keep_idxs]
    kept_emb = embeddings[keep_idxs]

    # 3. HDBSCAN clustering (adaptive #topics, no fixed k)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(kept_emb)

    # 4. Group into clusters, drop noise (-1)
    clusters = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(kept_sentences[idx])

    # 5. Sort sentences in each cluster by time
    for lbl in clusters:
        clusters[lbl] = sorted(clusters[lbl], key=lambda x: x["start"])

    logger.info(f"Found {len(clusters)} topic clusters after filtering")
    return clusters
