import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

class EvidenceRetriever:

    def __init__(self, config):
        # 1. Embedding model
        model_name = config.get("models", "embedding_model")
        self.model = SentenceTransformer(model_name)

        # 2. Cross-encoder for re-ranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        # 3. Retrieval settings
        # Adjusted for the original config.py which doesn't support defaults in .get()
        try:
            self.top_k = config.get("retrieval", "top_k")
        except KeyError:
            # Fallback if 'top_k' is missing from your default.yaml
            self.top_k = 10
            
        self.index_path = config.get("retrieval", "index_path")
        self.metadata_path = config.get("retrieval", "metadata_path")

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Prepare BM25 for hybrid retrieval
        self.bm25_corpus = [self._preprocess_text(m["text"]) for m in self.metadata]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def _preprocess_text(self, text):
        # Lowercase and remove non-alphanumeric for BM25
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return tokens

    def retrieve(self, query, rerank=True, hybrid=True, bm25_weight=0.3):
        query_embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Step 1: Get more candidates from FAISS
        dense_k = self.top_k * 5
        scores, indices = self.index.search(query_embedding, dense_k)

        dense_results = []
        for score, idx in zip(scores[0], indices[0]):
            dense_results.append({
                "idx": idx,
                "score": float(score),
                "title": self.metadata[idx]["title"],
                "text": self.metadata[idx]["text"]
            })

        # Step 2: BM25 retrieval (top_k * 5)
        if hybrid:
            query_tokens = self._preprocess_text(query)
            bm25_scores = self.bm25.get_scores(query_tokens)

            bm25_indices = np.argsort(bm25_scores)[::-1][:dense_k]

            bm25_results = []
            for idx in bm25_indices:
                bm25_results.append({
                    "idx": idx,
                    "bm25_score": float(bm25_scores[idx]),
                    "title": self.metadata[idx]["title"],
                    "text": self.metadata[idx]["text"]
                })

            # Step 3: Merge results
            combined = {}

            for r in dense_results:
                combined[r["idx"]] = r

            for r in bm25_results:
                if r["idx"] in combined:
                    combined[r["idx"]]["bm25_score"] = r["bm25_score"]
                else:
                    combined[r["idx"]] = {
                        "idx": r["idx"],
                        "score": 0.0,
                        "bm25_score": r["bm25_score"],
                        "title": r["title"],
                        "text": r["text"]
                    }

            results = list(combined.values())

            # Step 4: Normalize scores
            dense_scores = [r.get("score", 0) for r in results]
            bm25_scores = [r.get("bm25_score", 0) for r in results]

            max_dense = max(dense_scores) if dense_scores else 1
            max_bm25 = max(bm25_scores) if bm25_scores else 1

            for r in results:
                r["score"] = r.get("score", 0) / max_dense
                r["bm25_score"] = r.get("bm25_score", 0) / max_bm25

                r["hybrid_score"] = (
                    (1 - bm25_weight) * r["score"] +
                    bm25_weight * r["bm25_score"]
                )

            results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        else:
            results = dense_results

        # Step 5: Cross-encoder reranking
        if rerank:
            reranked_results = results[:dense_k]

            pairs = [(query, r["text"]) for r in reranked_results]
            rerank_scores = self.reranker.predict(pairs)

            for r, s in zip(reranked_results, rerank_scores):
                r["rerank_score"] = float(s)

            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

            results = reranked_results + results[dense_k:]
        
        return results[:self.top_k] if results else []