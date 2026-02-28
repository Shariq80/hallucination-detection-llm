from sentence_transformers import SentenceTransformer, util
import torch

class SimilarityScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the semantic similarity scoring pipeline."""
        self.model = SentenceTransformer(model_name)
    
    def score(self, claim, evidence):
        """Calculate similarity score for a single pair."""
        embeddings = self.model.encode([claim, evidence], convert_to_tensor=True)
        sim = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()
        return float(sim)
        
    def score_batch(self, claims, evidences, batch_size=32):
        """Calculate similarity scores for a batch of claim-evidence pairs."""
        if len(claims) != len(evidences):
            raise ValueError("Number of claims must match number of evidences.")
            
        claim_embeddings = self.model.encode(claims, batch_size=batch_size, convert_to_tensor=True)
        evidence_embeddings = self.model.encode(evidences, batch_size=batch_size, convert_to_tensor=True)
        
        # Calculate pairwise cosine similarity for corresponding pairs efficiently
        similarities = torch.nn.functional.cosine_similarity(claim_embeddings, evidence_embeddings)
        return similarities.tolist()
        
    def predict(self, claim, evidence, threshold=0.5):
        """Predict whether a claim is supported or hallucinated."""
        sim = self.score(claim, evidence)
        return {
            "score": sim,
            "prediction": "Supported" if sim >= threshold else "Hallucinated"
        }
        
    def predict_batch(self, claims, evidences, threshold=0.5, batch_size=32):
        """Predict whether claims are supported or hallucinated for a batch."""
        scores = self.score_batch(claims, evidences, batch_size=batch_size)
        return [
            {
                "score": score,
                "prediction": "Supported" if score >= threshold else "Hallucinated"
            }
            for score in scores
        ]