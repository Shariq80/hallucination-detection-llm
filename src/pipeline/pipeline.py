# src/pipeline/pipeline.py
from src.verification.nli import NLIVerifier
from src.similarity.similarity import SimilarityScorer
from src.verification.aggregator import ScoreAggregator
from src.retrieval.retrieval import EvidenceRetriever
from src.utils.config import Config

class HallucinationPipeline:

    def __init__(self, config_path="configs/default.yaml", nli_model_override=None):
        self.config = Config(config_path)

        # Pass override model directly to NLIVerifier
        self.nli = NLIVerifier(self.config, model_name=nli_model_override)
        self.similarity = SimilarityScorer(self.config)
        self.aggregator = ScoreAggregator(self.config)
        self.retriever = EvidenceRetriever(self.config)

    def verify(self, claim):
        print("\n==============================")
        print("CLAIM:", claim)
        print("==============================")

        # Step 0: Split into atomic claims (simplified)
        atomic_claims = [claim]
        print("Atomic Claims:", atomic_claims)

        atomic_results = []

        for atomic_claim in atomic_claims:
            print("\n--- Retrieving Evidence ---")
            retrieved = self.retriever.retrieve(atomic_claim)

            evidence_results = []

            for i, evidence in enumerate(retrieved):
                text = evidence["text"]

                print(f"\nEvidence {i+1}")
                print("Title:", evidence["title"])
                print("Text:", text[:200], "...")
                print("Retriever Score:", evidence.get("score", 0.0))

                # Similarity
                sim_score = self.similarity.score(atomic_claim, text)
                print("Similarity Score:", sim_score)

                # NLI
                nli_scores = self.nli.predict(atomic_claim, text)
                print("NLI Scores:", nli_scores)

                evidence_results.append({
                    "title": evidence["title"],
                    "text": text,
                    "retriever_score": evidence.get("score", 0.0),
                    "similarity_score": sim_score,
                    "nli_scores": nli_scores
                })

            # Aggregate evidence for this atomic claim
            print("\n--- Aggregating Evidence ---")
            aggregated_result = self.aggregator.aggregate(evidence_results)
            print("\nFinal Decision:", aggregated_result)

            atomic_results.append({
                "atomic_claim": atomic_claim,
                "evidence": evidence_results,
                "final_result": aggregated_result
            })

        return {
            "original_claim": claim,
            "atomic_results": atomic_results
        }