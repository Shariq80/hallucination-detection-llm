# src/retrieval.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path="index.faiss", chunk_path="index_chunks.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)

        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, claim, top_k=5):
        claim_embedding = self.model.encode([claim])
        claim_embedding = np.array(claim_embedding).astype("float32")

        distances, indices = self.index.search(claim_embedding, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            results.append({
                "score": float(score),
                "text": self.chunks[idx]
            })

        return results


if __name__ == "__main__":
    retriever = Retriever()
    claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."

    results = retriever.retrieve(claim)

    for r in results:
        print(r["score"])
        print(r["text"])
        print()