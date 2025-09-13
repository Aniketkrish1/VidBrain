import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load file
sentences, timestamps = [], []
with open("sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        text, start, end = line.rsplit(" ", 2)
        sentences.append(text.strip())
        timestamps.append((float(start), float(end)))

# 2. Encode
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, convert_to_numpy=True)

# 3. Cluster by topic
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
labels = clustering.fit_predict(embeddings)

# 4. Compute cluster centroids
clusters = {}
for i, label in enumerate(labels):
    clusters.setdefault(label, []).append(i)

centroids = {label: np.mean(embeddings[idxs], axis=0) for label, idxs in clusters.items()}

# 5. Importance filtering: keep only sentences close to centroid
important_sentences = []
for label, idxs in clusters.items():
    centroid = centroids[label].reshape(1, -1)
    sims = cosine_similarity(embeddings[idxs], centroid).flatten()
    threshold = np.median(sims)  # adaptive
    for idx, score in zip(idxs, sims):
        if score >= threshold:
            important_sentences.append((sentences[idx], timestamps[idx], label, score))

# 6. Merge consecutive sentences in same topic
important_sentences.sort(key=lambda x: x[1][0])  # sort by start time
result = []
current = None
for text, (start, end), label, score in important_sentences:
    if current and current["label"] == label and start <= current["end"] + 1:
        current["end"] = end
        current["text"] += " " + text
    else:
        if current: result.append(current)
        current = {"topic": label, "start": start, "end": end, "text": text, "label": label}
if current: result.append(current)

with open("topics.txt",'w') as f:
    for r in result:
        print(f"[{r['start']} - {r['end']}] {r['text']}",file=f)
