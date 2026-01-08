# utils/database.py
import os
import re
import logging
import torch
import pickle
from typing import Optional
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
logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, model_name: str = "all-mpnet-base-v2", db_path: str = "vector_db.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.db_path = db_path

        self.embeddings = None  # torch.Tensor or None
        self.metadata = []      # list of dicts

        if os.path.exists(db_path):
            try:
                self.load()
            except Exception as e:
                logger.warning("Failed to load VectorDB from %s: %s. Will rebuild on demand.", db_path, e)
                self.embeddings = None
                self.metadata = []

        logger.info("VectorDB device: %s", self.device)

    def build(self, segments):
        texts = [seg.get("text", "") for seg in segments]
        # Ensure start/end are numeric floats in metadata
        for seg in segments:
            try:
                seg["start"] = float(seg.get("start", 0.0))
            except Exception:
                seg["start"] = 0.0
            try:
                seg["end"] = float(seg.get("end", seg["start"]))
            except Exception:
                seg["end"] = seg["start"]

        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)

        self.embeddings = embeddings
        self.metadata = segments
        self.save()

    def save(self):
        # Save embeddings as CPU tensor to avoid device mismatch on load
        data = {"embeddings": None, "metadata": self.metadata}
        if self.embeddings is not None:
            try:
                cpu_emb = self.embeddings.cpu()
                data["embeddings"] = cpu_emb
            except Exception:
                # fallback: try to save as-is
                data["embeddings"] = self.embeddings

        # WARNING: using pickle; ensure db_path is trusted. Loading arbitrary pickles is unsafe.
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        # Load in a robust way and move embeddings to current device if possible
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
            emb = data.get("embeddings")
            if emb is None:
                self.embeddings = None
            else:
                try:
                    # If saved as tensor
                    if hasattr(emb, "to"):
                        self.embeddings = emb.to(self.device)
                    else:
                        # Try converting numpy/ndarray to tensor
                        import numpy as _np
                        if isinstance(emb, _np.ndarray):
                            self.embeddings = torch.from_numpy(emb).to(self.device)
                        else:
                            # Last resort: assign as-is (may fail later)
                            self.embeddings = emb
                except Exception as e:
                    logger.warning("Could not move embeddings to device %s: %s", self.device, e)
                    self.embeddings = None

            self.metadata = data.get("metadata", [])

    def search(self, query: str, top_k: int = 10):
        # If embeddings are not built/loaded, return no results so caller falls back
        # to clustering logic.
        if self.embeddings is None:
            return []

        # Compute query embedding
        query_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)
        scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings)

        k = min(top_k, int(self.embeddings.size(0)))
        topk = torch.topk(scores, k=k)

        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append({
                "text": self.metadata[idx].get("text", ""),
                "start": float(self.metadata[idx].get("start", 0.0)),
                "end": float(self.metadata[idx].get("end", self.metadata[idx].get("start", 0.0))),
                "score": float(score)
            })

        # Best-effort debug output saved to aaa/ if present
        try:
            os.makedirs("aaa", exist_ok=True)
            with open("aaa/retrieved_results.txt", "w", encoding="utf-8") as f:
                f.write(str(results))
        except Exception:
            # not critical
            pass

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
