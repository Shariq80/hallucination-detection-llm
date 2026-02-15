from verification.nli import NLIVerifier
from verification.similarity import SimilarityScorer
from verification.aggregator import ScoreAggregator

class HallucinationPipeline:
    def __init__(self):
        self.nli = NLIVerifier()
        self.similarity = SimilarityScorer()
        self.aggregator = ScoreAggregator()

    def verify(self, claim, evidence):
        sim_score = self.similarity.score(claim, evidence)
        nli_scores = self.nli.predict(claim, evidence)

        result = self.aggregator.aggregate(sim_score, nli_scores)

        return result