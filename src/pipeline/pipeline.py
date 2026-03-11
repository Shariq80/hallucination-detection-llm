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

        print("\n--- Retrieving Evidence ---")

        retrieved = self.retriever.retrieve(claim)

        # Convert retrieved evidence list → text
        evidence_text = " ".join([item["text"] for item in retrieved])

        print("\n--- Running NLI ---")

        nli_scores = self.nli.predict(claim, evidence_text)

        print("\n--- Computing Similarity ---")

        sim_score = self.similarity.score(claim, evidence_text)

        print("\n--- Aggregating Scores ---")

        result = self.aggregator.aggregate(sim_score, nli_scores)

        return result