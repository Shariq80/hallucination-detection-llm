import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class EvidenceRetriever:

    def __init__(self, config):
        model_name = config.get("models", "embedding_model")
        self.model = SentenceTransformer(model_name)

        self.top_k = config.get("retrieval", "top_k")
        self.index_path = config.get("retrieval", "index_path")
        self.metadata_path = config.get("retrieval", "metadata_path")

        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query):

        query_embedding = self.model.encode([query]).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, self.top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "score": float(score),
                "title": self.metadata[idx]["title"],
                "text": self.metadata[idx]["text"]
            })

        return results