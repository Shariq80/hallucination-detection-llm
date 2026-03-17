from src.verification.nli import NLIVerifier
from src.similarity.similarity import SimilarityScorer
from src.verification.aggregator import ScoreAggregator
from src.retrieval.retrieval import EvidenceRetriever
from src.utils.config import Config


class HallucinationPipeline:

    def __init__(self, config_path="configs/default.yaml"):

        self.config = Config(config_path)

        self.nli = NLIVerifier(self.config)
        self.similarity = SimilarityScorer(self.config)
        self.aggregator = ScoreAggregator(self.config)

        self.retriever = EvidenceRetriever(self.config)

    def verify(self, claim):

        print("\n==============================")
        print("CLAIM:", claim)
        print("==============================")

        print("\n--- Retrieving Evidence ---")

        retrieved = self.retriever.retrieve(claim)

        evidence_results = []

        for i, evidence in enumerate(retrieved):

            text = evidence["text"]

            print(f"\nEvidence {i+1}")
            print("Title:", evidence["title"])
            print("Text:", text[:200], "...")   # shorten display
            print("Retriever Score:", evidence["score"])

            print("\nRunning Similarity...")
            sim_score = self.similarity.score(claim, text)

            print("Similarity Score:", sim_score)

            print("Running NLI...")
            nli_scores = self.nli.predict(claim, text)

            print("NLI Scores:", nli_scores)

            evidence_results.append({
                "text": text,
                "similarity": sim_score,
                "nli": nli_scores
            })

        print("\n--- Aggregating Evidence ---")

        result = self.aggregator.aggregate(evidence_results)

        return result