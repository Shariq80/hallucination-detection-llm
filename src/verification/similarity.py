from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def score(self, claim, evidence):
        embeddings = self.model.encode([claim, evidence])
        sim = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(sim)