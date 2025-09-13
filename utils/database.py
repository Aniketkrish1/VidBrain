# utils/database.py
import os
import re
import torch
import pickle
from sentence_transformers import SentenceTransformer

# ------------------------------
# 1. Parse SRT file
# ------------------------------
def parse_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to capture index, timestamps, and text
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\n|\Z)",
        re.MULTILINE,
    )

    segments = []
    for match in pattern.finditer(content):
        index = int(match.group(1))
        start = match.group(2)
        end = match.group(3)
        text = match.group(4).replace("\n", " ").strip()
        segments.append({"id": index, "start": start, "end": end, "text": text})
    return segments

# ------------------------------
# 2. Embed with Torch (GPU if available)
# ------------------------------
class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2", db_path="vector_db.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.db_path = db_path

        self.embeddings = None  # torch.Tensor
        self.metadata = []      # list of dicts

        if os.path.exists(db_path):
            self.load()
        print(self.device)
    def build(self, segments):
        texts = [seg["text"] for seg in segments]
        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)

        self.embeddings = embeddings
        self.metadata = segments
        self.save()

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(
                {"embeddings": self.embeddings.cpu(), "metadata": self.metadata},
                f
            )

    def load(self):
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"].to(self.device)
            self.metadata = data["metadata"]

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)
        scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings)
        topk = torch.topk(scores, k=top_k)

        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append({
                "text": self.metadata[idx]["text"],
                "start": self.metadata[idx]["start"],
                "end": self.metadata[idx]["end"],
                "score": float(score)
            })
        return results

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    srt_file = "transcript.srt"  # change path if needed
    segments = parse_srt(srt_file)

    db = VectorDB()

    if db.embeddings is None:  # build only once
        print("Building vector database...")
        db.build(segments)
    else:
        print("Loaded existing database")

    query = "What are cookies and sessions?"
    results = db.search(query, top_k=3)

    print("\nQuery:", query)
    for r in results:
        print(f"[{r['start']} - {r['end']}] {r['text']} (score={r['score']:.4f})")
