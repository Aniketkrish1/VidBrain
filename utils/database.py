# utils/database.py
import os
import re
import torch
import pickle
import datetime
from sentence_transformers import SentenceTransformer

# ------------------------------
# 1. Parse SRT file
# ------------------------------

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""
    try:
        parts = re.split(r'[:,]', time_str)
        if len(parts) != 4:
            return 0.0
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(parts[3])
        
        total_seconds = (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)
        return total_seconds
    except Exception:
        return 0.0

def parse_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_srt_content(content)

def parse_srt_content(content):
    """Parse SRT content string (not file) into segments"""
    # Regex to capture index, timestamps, and text
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\n|\Z)",
        re.MULTILINE,
    )

    segments = []
    for match in pattern.finditer(content):
        index = int(match.group(1))
        start_str = match.group(2)
        end_str = match.group(3)
        text = match.group(4).replace("\n", " ").strip()
        
        # --- FIX: Convert timestamps to seconds here ---
        segments.append({
            "id": index, 
            "start": srt_time_to_seconds(start_str), # Use helper
            "end": srt_time_to_seconds(end_str),     # Use helper
            "text": text
        })
    return segments

# ------------------------------
# 2. Embed with Torch (GPU if available)
# ------------------------------
class VectorDB:
    def __init__(self, model_name="multi-qa-mpnet-base-dot-v1", db_path="vector_db.pkl"):
        """
        --- IMPROVEMENT: Changed default model to one better for search/Q&A ---
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.db_path = db_path

        self.embeddings = None  # torch.Tensor
        self.metadata = []      # list of dicts

        if os.path.exists(db_path):
            self.load()
        print(f"VectorDB using device: {self.device}")
        
    def build(self, segments):
        texts = [seg["text"] for seg in segments]
        print(f"Building embeddings for {len(texts)} segments...")
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True, 
            device=self.device, 
            show_progress_bar=True
        )

        # --- FIX: Timestamp conversion bug is GONE ---
        # The 'segments' data already has float seconds from parse_srt
        # We can just assign them directly.
        self.embeddings = embeddings
        self.metadata = segments
        self.save()
        print("Database built and saved.")

    def save(self):
        with open(self.db_path, "wb") as f:
            data = {"embeddings": None, "metadata": self.metadata}
            if self.embeddings is not None:
                try:
                    # Move to CPU for pickling, as it's safer
                    data["embeddings"] = self.embeddings.cpu()
                except Exception as e:
                    print(f"Warning: Could not move embeddings to CPU. Saving as is. Error: {e}")
                    data["embeddings"] = self.embeddings
            pickle.dump(data, f)

    def load(self):
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                if data["embeddings"] is not None:
                    self.embeddings = data["embeddings"].to(self.device)
                    self.metadata = data["metadata"]
                    print(f"Loaded existing database with {len(self.metadata)} entries.")
                else:
                    print("Loaded database, but no embeddings found.")
        except Exception as e:
            print(f"Error loading database file. It might be corrupt. Error: {e}")
            print("Will attempt to rebuild.")
            self.embeddings = None
            self.metadata = []

    def search(self, query, top_k=5):
        if self.embeddings is None:
            print("Error: No embeddings loaded. Cannot perform search.")
            return []

        query_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)
        
        # Use dot_score for models tuned with dot product (like multi-qa-mpnet)
        # Use cosine_similarity for others
        if "mpnet" in str(self.model).lower():
             scores = torch.mm(query_emb, self.embeddings.T)[0]
        else:
             scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings)

        # Ensure k is not larger than the number of embeddings
        k = min(top_k, self.embeddings.size(0))
        topk = torch.topk(scores, k=k)

        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append({
                "text": self.metadata[idx]["text"],
                "start": float(self.metadata[idx]["start"]),
                "end": float(self.metadata[idx]["end"]),
                "score": float(score)
            })
        return results

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    srt_file = "transcript.srt"  # Make sure this file exists
    
    if not os.path.exists(srt_file):
        print(f"Error: {srt_file} not found. Please create it first (e.g., using transcriber.py)")
    else:
        db = VectorDB()

        # Check if DB is empty or if user wants to force rebuild
        if db.embeddings is None:  
            print("No database found. Building new one...")
            segments = parse_srt(srt_file)
            if segments:
                db.build(segments)
            else:
                print("SRT file seems to be empty or invalid.")
        else:
            print("Loaded existing database.")

        if db.embeddings is not None:
            query = "What are cookies and sessions?"
            results = db.search(query, top_k=3)

            print("\nQuery:", query)
            for r in results:
                # Format start/end times back to readable format
                start_time = str(datetime.timedelta(seconds=r['start'])).split('.')[0]
                end_time = str(datetime.timedelta(seconds=r['end'])).split('.')[0]
                print(f"[{start_time} - {end_time}] {r['text']} (score={r['score']:.4f})")
